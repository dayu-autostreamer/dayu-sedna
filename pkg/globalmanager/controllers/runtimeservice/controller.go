/*
Copyright 2021 The KubeEdge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package runtimeservice

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
	sednaclient "github.com/dayu-autostreamer/dayu-sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	sednalisters "github.com/dayu-autostreamer/dayu-sedna/pkg/client/listers/sedna/v1alpha1"
	globalruntime "github.com/dayu-autostreamer/dayu-sedna/pkg/globalmanager/runtime"
	activation "github.com/dayu-autostreamer/dayu-sedna/pkg/runtimeservice"
)

const (
	// Name is the controller registry name.
	Name = "RuntimeService"
	// KindName is the Kubernetes API kind handled by this controller.
	KindName = activation.KindName

	runtimeServiceUIDIndex = "runtimeServiceUID"
	targetNodeIndex        = "targetNode"

	notReadyRequeueInterval = 5 * time.Second
	activationRetryInterval = 30 * time.Second
	ackCacheRetryInterval   = time.Second
)

// Controller reconciles RuntimeService resources from shared informer caches.
type Controller struct {
	kubeClient    kubernetes.Interface
	runtimeClient sednaclient.SednaV1alpha1Interface

	runtimeLister    sednalisters.RuntimeServiceLister
	runtimeIndexer   cache.Indexer
	deploymentLister appslisters.DeploymentLister
	podLister        corelisters.PodLister
	serviceLister    corelisters.ServiceLister
	endpointsLister  corelisters.EndpointsLister
	nodeLister       corelisters.NodeLister

	cacheSyncs []cache.InformerSynced
	queue      workqueue.RateLimitingInterface

	downstreamSend globalruntime.DownstreamSendFunc

	ackMu        sync.Mutex
	pendingAcks  map[string]activation.ActivationAck
	acceptedAcks map[string]activation.ActivationAck
}

var (
	_ globalruntime.FeatureControllerI            = (*Controller)(nil)
	_ globalruntime.SourceAwareFeatureControllerI = (*Controller)(nil)
)

func (c *Controller) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting %s controller", Name)
	defer klog.Infof("Shutting down %s controller", Name)

	if !cache.WaitForNamedCacheSync(Name, stopCh, c.cacheSyncs...) {
		klog.Errorf("failed to wait for %s caches to sync", Name)
		return
	}

	go wait.Until(c.worker, time.Second, stopCh)
	<-stopCh
}

func (c *Controller) worker() {
	for c.processNextWorkItem() {
	}
}

func (c *Controller) processNextWorkItem() bool {
	item, shutdown := c.queue.Get()
	if shutdown {
		return false
	}
	defer c.queue.Done(item)

	key, ok := item.(string)
	if !ok {
		c.queue.Forget(item)
		runtime.HandleError(fmt.Errorf("RuntimeService queue item has unexpected type %T", item))
		return true
	}

	if err := c.sync(key); err != nil {
		runtime.HandleError(fmt.Errorf("failed to reconcile RuntimeService %q: %w", key, err))
		c.queue.AddRateLimited(key)
		return true
	}
	c.queue.Forget(item)
	return true
}

func (c *Controller) sync(key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	shared, err := c.runtimeLister.RuntimeServices(namespace).Get(name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			c.clearActivationAcks(key)
			return nil
		}
		return err
	}
	service := shared.DeepCopy()
	if service.DeletionTimestamp != nil {
		c.sendCancellation(service)
		c.clearActivationAcks(key)
		return nil
	}

	specHash, hashErr := runtimeSpecHash(service)
	validationErr := validateRuntimeService(service)
	if validationErr == nil {
		validationErr = hashErr
	}
	if validationErr == nil && service.Status.ObservedSpecHash != "" &&
		(service.Status.ObservedRevision != service.Spec.DeploymentRevision || service.Status.ObservedSpecHash != specHash) {
		validationErr = fmt.Errorf("RuntimeService spec and revision are immutable; create a new revision-scoped resource")
	}
	if validationErr == nil {
		validationErr = c.validateChildSpecHash(service, specHash)
	}
	if validationErr != nil {
		status := invalidRuntimeStatus(service, validationErr)
		return c.updateStatus(service.Namespace, service.Name, status)
	}

	deployment, deploymentErr := c.ensureDeployment(service)
	if deploymentErr != nil {
		return c.finishResourceError(service, specHash, deploymentErr)
	}
	networkService, serviceErr := c.ensureService(service)
	if serviceErr != nil {
		return c.finishResourceError(service, specHash, serviceErr)
	}

	observation := c.observe(service, deployment, networkService)
	working := service.DeepCopy()
	c.applyPendingAck(working, observation)
	status := buildRuntimeStatus(working, specHash, observation)

	dispatchErr := c.maybeDispatchActivation(working, observation, &status)
	if dispatchErr != nil {
		setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionActivated,
			false, "ActivationDispatchFailed", dispatchErr.Error())
		setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionReady,
			false, "ReadinessGatesPending", "node-local mesh activation could not be dispatched")
	}

	if err := c.updateStatus(service.Namespace, service.Name, status); err != nil {
		return err
	}
	if c.hasAcceptedAck(key) {
		// The informer may still expose the pre-ACK status. Reconcile once more
		// so the persisted status can retire the short-lived accepted ACK cache.
		c.queue.AddAfter(key, ackCacheRetryInterval)
	} else if !runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionReady) {
		c.queue.AddAfter(key, notReadyRequeueInterval)
	}
	return dispatchErr
}

func invalidRuntimeStatus(service *sednav1.RuntimeService, validationErr error) sednav1.RuntimeServiceStatus {
	status := service.DeepCopy().Status
	status.ObservedGeneration = service.Generation
	status.DeploymentRef = nil
	status.PodRef = nil
	status.Endpoint = nil
	setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionSpecAccepted,
		false, "InvalidSpec", validationErr.Error())
	for _, conditionType := range []sednav1.RuntimeServiceConditionType{
		sednav1.RuntimeServiceConditionResourcesReconciled,
		sednav1.RuntimeServiceConditionNodeReady,
		sednav1.RuntimeServiceConditionWorkloadReady,
		sednav1.RuntimeServiceConditionEndpointReady,
		sednav1.RuntimeServiceConditionActivated,
		sednav1.RuntimeServiceConditionReady,
	} {
		setRuntimeCondition(&status, service.Generation, conditionType, false, "InvalidSpec", validationErr.Error())
	}
	return status
}

func (c *Controller) finishResourceError(service *sednav1.RuntimeService, specHash string, resourceErr error) error {
	observation := c.observe(service, nil, nil)
	observation.resourcesReconciled = false
	observation.resourcesReason = "ReconcileFailed"
	observation.resourcesMessage = resourceErr.Error()
	status := buildRuntimeStatus(service, specHash, observation)
	if statusErr := c.updateStatus(service.Namespace, service.Name, status); statusErr != nil {
		return fmt.Errorf("resource error: %v; status error: %w", resourceErr, statusErr)
	}
	return resourceErr
}

func (c *Controller) observe(service *sednav1.RuntimeService, deployment *appsv1.Deployment, networkService *corev1.Service) runtimeObservation {
	observation := runtimeObservation{
		resourcesReconciled: true,
		resourcesReason:     "Reconciled",
		resourcesMessage:    "deployment and optional cluster service match the desired state",
		deployment:          deployment,
		service:             networkService,
	}

	if observation.deployment == nil {
		if value, err := c.deploymentLister.Deployments(service.Namespace).Get(service.Name); err == nil {
			observation.deployment = value
		}
	}
	if pods, err := c.podLister.Pods(service.Namespace).List(podSelector(service)); err == nil {
		observation.pods = pods
	}
	if node, err := c.nodeLister.Get(service.Spec.TargetNode); err == nil {
		observation.node = node
	}
	if service.Spec.Endpoint != nil {
		if observation.service == nil {
			if value, err := c.serviceLister.Services(service.Namespace).Get(service.Name); err == nil {
				observation.service = value
			}
		}
		if endpoints, err := c.endpointsLister.Endpoints(service.Namespace).Get(service.Name); err == nil {
			observation.endpoints = endpoints
		}
	}
	return observation
}

func (c *Controller) ensureDeployment(service *sednav1.RuntimeService) (*appsv1.Deployment, error) {
	desired := desiredDeployment(service)
	actual, err := c.deploymentLister.Deployments(service.Namespace).Get(service.Name)
	if apierrors.IsNotFound(err) {
		created, createErr := c.kubeClient.AppsV1().Deployments(service.Namespace).Create(context.TODO(), desired, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(createErr) {
			existing, getErr := c.kubeClient.AppsV1().Deployments(service.Namespace).Get(context.TODO(), service.Name, metav1.GetOptions{})
			if getErr != nil {
				return nil, getErr
			}
			if !metav1.IsControlledBy(existing, service) {
				return nil, fmt.Errorf("Deployment %s/%s already exists and is not controlled by RuntimeService UID %s", service.Namespace, service.Name, service.UID)
			}
			if hashErr := validateOwnedChildSpecHash(existing, service); hashErr != nil {
				return nil, hashErr
			}
			return existing, nil
		}
		return created, createErr
	}
	if err != nil {
		return nil, err
	}
	if !metav1.IsControlledBy(actual, service) {
		return nil, fmt.Errorf("Deployment %s/%s is not controlled by RuntimeService UID %s", service.Namespace, service.Name, service.UID)
	}
	if hashErr := validateOwnedChildSpecHash(actual, service); hashErr != nil {
		return nil, hashErr
	}

	updated := actual.DeepCopy()
	updated.Labels = mergeStringMap(updated.Labels, desired.Labels)
	updated.Annotations = mergeStringMap(updated.Annotations, desired.Annotations)
	updated.OwnerReferences = desired.OwnerReferences
	updated.Spec = desired.Spec
	if reflect.DeepEqual(actual, updated) {
		return actual, nil
	}
	return c.kubeClient.AppsV1().Deployments(service.Namespace).Update(context.TODO(), updated, metav1.UpdateOptions{})
}

func (c *Controller) ensureService(service *sednav1.RuntimeService) (*corev1.Service, error) {
	desired := desiredService(service)
	actual, err := c.serviceLister.Services(service.Namespace).Get(service.Name)
	if desired == nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		if !metav1.IsControlledBy(actual, service) {
			return nil, fmt.Errorf("Service %s/%s is not controlled by RuntimeService UID %s", service.Namespace, service.Name, service.UID)
		}
		if err := c.kubeClient.CoreV1().Services(service.Namespace).Delete(context.TODO(), actual.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return nil, err
		}
		return nil, nil
	}
	if apierrors.IsNotFound(err) {
		created, createErr := c.kubeClient.CoreV1().Services(service.Namespace).Create(context.TODO(), desired, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(createErr) {
			existing, getErr := c.kubeClient.CoreV1().Services(service.Namespace).Get(context.TODO(), service.Name, metav1.GetOptions{})
			if getErr != nil {
				return nil, getErr
			}
			if !metav1.IsControlledBy(existing, service) {
				return nil, fmt.Errorf("Service %s/%s already exists and is not controlled by RuntimeService UID %s", service.Namespace, service.Name, service.UID)
			}
			if hashErr := validateOwnedChildSpecHash(existing, service); hashErr != nil {
				return nil, hashErr
			}
			return existing, nil
		}
		return created, createErr
	}
	if err != nil {
		return nil, err
	}
	if !metav1.IsControlledBy(actual, service) {
		return nil, fmt.Errorf("Service %s/%s is not controlled by RuntimeService UID %s", service.Namespace, service.Name, service.UID)
	}
	if hashErr := validateOwnedChildSpecHash(actual, service); hashErr != nil {
		return nil, hashErr
	}

	updated := actual.DeepCopy()
	updated.Labels = mergeStringMap(updated.Labels, desired.Labels)
	updated.Annotations = mergeStringMap(updated.Annotations, desired.Annotations)
	updated.OwnerReferences = desired.OwnerReferences
	// Only fields owned by RuntimeService are changed. API-allocated ClusterIP,
	// ClusterIPs, IPFamilies and IPFamilyPolicy remain untouched.
	// RuntimeService owns the complete routing policy. Preserve only fields allocated
	// by the API server; resetting the remainder prevents affinity/topology drift from
	// creating data-plane behavior that disagrees with the managed EdgeMesh route.
	clusterIP := actual.Spec.ClusterIP
	clusterIPs := append([]string(nil), actual.Spec.ClusterIPs...)
	ipFamilies := append([]corev1.IPFamily(nil), actual.Spec.IPFamilies...)
	var ipFamilyPolicy *corev1.IPFamilyPolicyType
	if actual.Spec.IPFamilyPolicy != nil {
		value := *actual.Spec.IPFamilyPolicy
		ipFamilyPolicy = &value
	}
	var internalTrafficPolicy *corev1.ServiceInternalTrafficPolicyType
	if actual.Spec.InternalTrafficPolicy != nil {
		value := *actual.Spec.InternalTrafficPolicy
		internalTrafficPolicy = &value
	}
	desired.Spec.DeepCopyInto(&updated.Spec)
	updated.Spec.ClusterIP = clusterIP
	updated.Spec.ClusterIPs = clusterIPs
	updated.Spec.IPFamilies = ipFamilies
	updated.Spec.IPFamilyPolicy = ipFamilyPolicy
	// This field is feature-gated and default-off in the Kubernetes 1.21 API
	// level used by this repository. Preserve the API-server representation to
	// avoid a write/default loop on clusters with different gate settings.
	updated.Spec.InternalTrafficPolicy = internalTrafficPolicy
	if reflect.DeepEqual(actual, updated) {
		return actual, nil
	}
	return c.kubeClient.CoreV1().Services(service.Namespace).Update(context.TODO(), updated, metav1.UpdateOptions{})
}

func (c *Controller) updateStatus(namespace, name string, desired sednav1.RuntimeServiceStatus) error {
	client := c.runtimeClient.RuntimeServices(namespace)
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		latest, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if reflect.DeepEqual(latest.Status, desired) {
			return nil
		}
		latest.Status = desired
		_, err = client.UpdateStatus(context.TODO(), latest, metav1.UpdateOptions{})
		return err
	})
}

func (c *Controller) validateChildSpecHash(service *sednav1.RuntimeService, specHash string) error {
	children := []metav1.Object{}
	if deployment, err := c.deploymentLister.Deployments(service.Namespace).Get(service.Name); err == nil {
		children = append(children, deployment)
	} else if !apierrors.IsNotFound(err) {
		return err
	}
	if networkService, err := c.serviceLister.Services(service.Namespace).Get(service.Name); err == nil {
		children = append(children, networkService)
	} else if !apierrors.IsNotFound(err) {
		return err
	}
	for _, child := range children {
		if !metav1.IsControlledBy(child, service) {
			continue
		}
		anchoredHash := child.GetAnnotations()[AnnotationRuntimeSpecHash]
		if anchoredHash != specHash {
			return fmt.Errorf("RuntimeService spec is immutable after child resource creation; create a new revision-scoped resource")
		}
	}
	return nil
}

func validateOwnedChildSpecHash(child metav1.Object, service *sednav1.RuntimeService) error {
	expected, err := runtimeSpecHash(service)
	if err != nil {
		return err
	}
	if child.GetAnnotations()[AnnotationRuntimeSpecHash] != expected {
		return fmt.Errorf("%T %s/%s has a missing or mismatched RuntimeService spec hash", child, child.GetNamespace(), child.GetName())
	}
	return nil
}

func (c *Controller) hasAcceptedAck(key string) bool {
	c.ackMu.Lock()
	defer c.ackMu.Unlock()
	_, found := c.acceptedAcks[key]
	return found
}

func (c *Controller) applyPendingAck(service *sednav1.RuntimeService, observation runtimeObservation) {
	if observation.service == nil || observation.endpoints == nil {
		return
	}
	key, _ := cache.MetaNamespaceKeyFunc(service)
	c.ackMu.Lock()
	ack, pending := c.pendingAcks[key]
	found := pending
	if !pending {
		ack, found = c.acceptedAcks[key]
		if found && activationMatchesAck(service.Status.Activation, ack) {
			delete(c.acceptedAcks, key)
			c.ackMu.Unlock()
			return
		}
	}
	c.ackMu.Unlock()
	if !found {
		return
	}

	readyPod, readyCount := uniqueReadyPod(observation.pods, service)
	if readyCount != 1 {
		return
	}
	if ack.RuntimeServiceUID != service.UID || ack.ServiceUID != observation.service.UID ||
		ack.EndpointPodUID != readyPod.UID ||
		ack.DeploymentRevision != service.Spec.DeploymentRevision || ack.RuntimeID != service.Name ||
		ack.TargetNode != service.Spec.TargetNode {
		c.clearActivationAcks(key)
		klog.Warningf("Ignoring stale RuntimeService activation ack for %s/%s", service.Namespace, service.Name)
		return
	}

	activationStatus := service.Status.Activation
	if activationMatchesAck(activationStatus, ack) {
		c.clearActivationAcks(key)
		return
	}
	c.ackMu.Lock()
	delete(c.pendingAcks, key)
	c.acceptedAcks[key] = ack
	c.ackMu.Unlock()
	ackTime := metav1.Now()
	if !ack.ObservedAt.IsZero() {
		ackTime = metav1.NewTime(ack.ObservedAt)
	}
	if activationStatus == nil {
		activationStatus = &sednav1.RuntimeServiceActivationStatus{}
	}
	activationStatus.TargetNode = service.Spec.TargetNode
	activationStatus.ActivatedRevision = ack.DeploymentRevision
	activationStatus.ActivatedServiceUID = ack.ServiceUID
	activationStatus.ActivatedEndpointPodUID = ack.EndpointPodUID
	activationStatus.ActivatedLocalSequence = ack.LocalSequence
	activationStatus.LastAckTime = &ackTime
	service.Status.Activation = activationStatus
}

func activationMatchesAck(status *sednav1.RuntimeServiceActivationStatus, ack activation.ActivationAck) bool {
	return status != nil && status.TargetNode == ack.TargetNode &&
		status.ActivatedRevision == ack.DeploymentRevision &&
		status.ActivatedServiceUID == ack.ServiceUID &&
		status.ActivatedEndpointPodUID == ack.EndpointPodUID &&
		status.ActivatedLocalSequence == ack.LocalSequence
}

func (c *Controller) clearActivationAcks(key string) {
	c.ackMu.Lock()
	delete(c.pendingAcks, key)
	delete(c.acceptedAcks, key)
	c.ackMu.Unlock()
}

func (c *Controller) maybeDispatchActivation(service *sednav1.RuntimeService, observation runtimeObservation, status *sednav1.RuntimeServiceStatus) error {
	if service.Spec.Endpoint == nil || !runtimeConditionTrue(status, sednav1.RuntimeServiceConditionEndpointReady) ||
		status.Endpoint == nil || status.PodRef == nil || observation.endpoints == nil {
		return nil
	}
	if runtimeConditionTrue(status, sednav1.RuntimeServiceConditionActivated) {
		return nil
	}
	if c.downstreamSend == nil {
		return fmt.Errorf("RuntimeService downstream sender is not configured")
	}

	now := metav1.Now()
	activationStatus := status.Activation
	if activationStatus != nil && activationStatus.LastRequestTime != nil &&
		activationStatus.RequestedRevision == service.Spec.DeploymentRevision &&
		activationStatus.RequestedServiceUID == status.Endpoint.ServiceRef.UID &&
		activationStatus.RequestedEndpointPodUID == status.PodRef.UID &&
		now.Sub(activationStatus.LastRequestTime.Time) < activationRetryInterval {
		return nil
	}

	request := &activation.ActivationRequest{
		TypeMeta:           metav1.TypeMeta{APIVersion: sednav1.SchemeGroupVersion.String(), Kind: KindName},
		ObjectMeta:         metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		RuntimeServiceUID:  service.UID,
		ServiceUID:         status.Endpoint.ServiceRef.UID,
		EndpointPodUID:     status.PodRef.UID,
		DeploymentRevision: service.Spec.DeploymentRevision,
		RuntimeID:          service.Name,
		TargetNode:         service.Spec.TargetNode,
	}
	if err := c.downstreamSend(service.Spec.TargetNode, watch.Added, request); err != nil {
		return err
	}
	if activationStatus == nil {
		activationStatus = &sednav1.RuntimeServiceActivationStatus{}
	}
	activationStatus.TargetNode = service.Spec.TargetNode
	activationStatus.RequestedRevision = service.Spec.DeploymentRevision
	activationStatus.RequestedServiceUID = status.Endpoint.ServiceRef.UID
	activationStatus.RequestedEndpointPodUID = status.PodRef.UID
	activationStatus.LastRequestTime = &now
	status.Activation = activationStatus
	return nil
}

func (c *Controller) updateFromEdge(sourceNode, name, namespace, _ string, content []byte) error {
	var ack activation.ActivationAck
	if err := json.Unmarshal(content, &ack); err != nil {
		return fmt.Errorf("decode RuntimeService activation ack: %w", err)
	}
	if namespace == "" || ack.RuntimeID != name || ack.RuntimeServiceUID == "" ||
		ack.ServiceUID == "" || ack.EndpointPodUID == "" || ack.DeploymentRevision < 1 ||
		ack.TargetNode == "" || ack.LocalSequence == 0 || ack.ObservedAt.IsZero() {
		return fmt.Errorf("activation ack identity does not match message header")
	}
	if sourceNode == "" {
		return fmt.Errorf("activation ack is missing the websocket source node")
	}
	if sourceNode != ack.TargetNode {
		return fmt.Errorf("activation ack target node %q does not match websocket source node %q", ack.TargetNode, sourceNode)
	}
	key := namespace + "/" + name
	c.ackMu.Lock()
	c.pendingAcks[key] = ack
	c.ackMu.Unlock()
	c.queue.Add(key)
	return nil
}

func (c *Controller) SetDownstreamSendFunc(send globalruntime.DownstreamSendFunc) error {
	c.downstreamSend = send
	return nil
}

func (c *Controller) SetUpstreamHandler(add globalruntime.UpstreamHandlerAddFunc) error {
	return add(KindName, func(name, namespace, operation string, content []byte) error {
		return c.updateFromEdge("", name, namespace, operation, content)
	})
}

func (c *Controller) SetSourceAwareUpstreamHandler(add globalruntime.SourceAwareUpstreamHandlerAddFunc) error {
	return add(KindName, c.updateFromEdge)
}
