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
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
	globalruntime "github.com/dayu-autostreamer/dayu-sedna/pkg/globalmanager/runtime"
	activation "github.com/dayu-autostreamer/dayu-sedna/pkg/runtimeservice"
)

func objectFromEvent(obj interface{}) (metav1.Object, bool) {
	if object, ok := obj.(metav1.Object); ok {
		return object, true
	}
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		return nil, false
	}
	object, ok := tombstone.Obj.(metav1.Object)
	return object, ok
}

func (c *Controller) enqueueRuntime(obj interface{}) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		klog.Warningf("failed to get RuntimeService key: %v", err)
		return
	}
	c.queue.Add(key)
}

func (c *Controller) enqueueRuntimeUpdate(oldObj, newObj interface{}) {
	oldService, oldOK := oldObj.(*sednav1.RuntimeService)
	newService, newOK := newObj.(*sednav1.RuntimeService)
	if !oldOK || !newOK {
		return
	}
	if oldService.Generation == newService.Generation &&
		reflectTimeEqual(oldService.DeletionTimestamp, newService.DeletionTimestamp) {
		return
	}
	if newService.DeletionTimestamp != nil {
		c.sendCancellation(newService)
	}
	c.enqueueRuntime(newService)
}

func reflectTimeEqual(left, right *metav1.Time) bool {
	if left == nil || right == nil {
		return left == right
	}
	return left.Equal(right)
}

func (c *Controller) enqueueOwned(obj interface{}) {
	object, ok := objectFromEvent(obj)
	if !ok {
		return
	}
	uid := object.GetLabels()[LabelRuntimeServiceUID]
	if uid != "" {
		items, err := c.runtimeIndexer.ByIndex(runtimeServiceUIDIndex, uid)
		if err == nil {
			for _, item := range items {
				c.enqueueRuntime(item)
			}
			if len(items) > 0 {
				return
			}
		}
	}
	for _, owner := range object.GetOwnerReferences() {
		if owner.Controller != nil && *owner.Controller && owner.Kind == KindName {
			c.queue.Add(object.GetNamespace() + "/" + owner.Name)
			return
		}
	}
}

func (c *Controller) enqueueOwnedUpdate(oldObj, newObj interface{}) {
	oldMeta, oldOK := objectFromEvent(oldObj)
	newMeta, newOK := objectFromEvent(newObj)
	if !oldOK || !newOK || oldMeta.GetResourceVersion() == newMeta.GetResourceVersion() {
		return
	}
	// An update can remove or replace both the identity label and ownerRef.
	// The old object must still wake the previous RuntimeService owner.
	c.enqueueOwned(oldObj)
	c.enqueueOwned(newObj)
}

func (c *Controller) enqueueNode(obj interface{}) {
	object, ok := objectFromEvent(obj)
	if !ok {
		return
	}
	items, err := c.runtimeIndexer.ByIndex(targetNodeIndex, object.GetName())
	if err != nil {
		return
	}
	for _, item := range items {
		c.enqueueRuntime(item)
	}
}

func (c *Controller) enqueueNodeUpdate(oldObj, newObj interface{}) {
	oldMeta, oldOK := objectFromEvent(oldObj)
	newMeta, newOK := objectFromEvent(newObj)
	if !oldOK || !newOK || oldMeta.GetResourceVersion() == newMeta.GetResourceVersion() {
		return
	}
	c.enqueueNode(newObj)
}

func (c *Controller) deleteRuntime(obj interface{}) {
	object, ok := objectFromEvent(obj)
	if ok {
		if service, serviceOK := object.(*sednav1.RuntimeService); serviceOK {
			c.sendCancellation(service)
			key, _ := cache.MetaNamespaceKeyFunc(service)
			c.clearActivationAcks(key)
		}
	}
	c.enqueueRuntime(obj)
}

func (c *Controller) sendCancellation(service *sednav1.RuntimeService) {
	if c.downstreamSend == nil || service.Spec.Endpoint == nil {
		return
	}
	request := &activation.ActivationRequest{
		TypeMeta:           metav1.TypeMeta{APIVersion: sednav1.SchemeGroupVersion.String(), Kind: KindName},
		ObjectMeta:         metav1.ObjectMeta{Name: service.Name, Namespace: service.Namespace},
		RuntimeServiceUID:  service.UID,
		DeploymentRevision: service.Spec.DeploymentRevision,
		RuntimeID:          service.Name,
		TargetNode:         service.Spec.TargetNode,
	}
	if service.Status.Endpoint != nil {
		request.ServiceUID = service.Status.Endpoint.ServiceRef.UID
	}
	if service.Status.PodRef != nil {
		request.EndpointPodUID = service.Status.PodRef.UID
	}
	if err := c.downstreamSend(service.Spec.TargetNode, watch.Deleted, request); err != nil {
		klog.Warningf("failed to cancel RuntimeService activation %s/%s: %v", service.Namespace, service.Name, err)
	}
}

// New constructs a RuntimeService controller backed entirely by shared informer listers.
func New(cc *globalruntime.ControllerContext) (globalruntime.FeatureControllerI, error) {
	runtimeInformer := cc.SednaInformerFactory.Sedna().V1alpha1().RuntimeServices()
	if err := runtimeInformer.Informer().AddIndexers(cache.Indexers{
		runtimeServiceUIDIndex: func(obj interface{}) ([]string, error) {
			service, ok := obj.(*sednav1.RuntimeService)
			if !ok {
				return nil, fmt.Errorf("expected RuntimeService, got %T", obj)
			}
			return []string{string(service.UID)}, nil
		},
		targetNodeIndex: func(obj interface{}) ([]string, error) {
			service, ok := obj.(*sednav1.RuntimeService)
			if !ok {
				return nil, fmt.Errorf("expected RuntimeService, got %T", obj)
			}
			return []string{service.Spec.TargetNode}, nil
		},
	}); err != nil {
		return nil, err
	}

	deploymentInformer := cc.KubeInformerFactory.Apps().V1().Deployments()
	podInformer := cc.KubeInformerFactory.Core().V1().Pods()
	serviceInformer := cc.KubeInformerFactory.Core().V1().Services()
	endpointsInformer := cc.KubeInformerFactory.Core().V1().Endpoints()
	nodeInformer := cc.KubeInformerFactory.Core().V1().Nodes()

	controller := &Controller{
		kubeClient:       cc.KubeClient,
		runtimeClient:    cc.SednaClient.SednaV1alpha1(),
		runtimeLister:    runtimeInformer.Lister(),
		runtimeIndexer:   runtimeInformer.Informer().GetIndexer(),
		deploymentLister: deploymentInformer.Lister(),
		podLister:        podInformer.Lister(),
		serviceLister:    serviceInformer.Lister(),
		endpointsLister:  endpointsInformer.Lister(),
		nodeLister:       nodeInformer.Lister(),
		queue: workqueue.NewNamedRateLimitingQueue(
			workqueue.NewItemExponentialFailureRateLimiter(globalruntime.DefaultBackOff, globalruntime.MaxBackOff),
			"runtimeservice",
		),
		pendingAcks:  make(map[string]activation.ActivationAck),
		acceptedAcks: make(map[string]activation.ActivationAck),
	}
	controller.cacheSyncs = []cache.InformerSynced{
		runtimeInformer.Informer().HasSynced,
		deploymentInformer.Informer().HasSynced,
		podInformer.Informer().HasSynced,
		serviceInformer.Informer().HasSynced,
		endpointsInformer.Informer().HasSynced,
		nodeInformer.Informer().HasSynced,
	}

	runtimeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    controller.enqueueRuntime,
		UpdateFunc: controller.enqueueRuntimeUpdate,
		DeleteFunc: controller.deleteRuntime,
	})
	ownedHandlers := cache.ResourceEventHandlerFuncs{
		AddFunc:    controller.enqueueOwned,
		UpdateFunc: controller.enqueueOwnedUpdate,
		DeleteFunc: controller.enqueueOwned,
	}
	deploymentInformer.Informer().AddEventHandler(ownedHandlers)
	podInformer.Informer().AddEventHandler(ownedHandlers)
	serviceInformer.Informer().AddEventHandler(ownedHandlers)
	endpointsInformer.Informer().AddEventHandler(ownedHandlers)
	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    controller.enqueueNode,
		UpdateFunc: controller.enqueueNodeUpdate,
		DeleteFunc: controller.enqueueNode,
	})

	return controller, nil
}
