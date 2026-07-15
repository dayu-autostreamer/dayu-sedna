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
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	appstyped "k8s.io/client-go/kubernetes/typed/apps/v1"
	coretyped "k8s.io/client-go/kubernetes/typed/core/v1"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

type deploymentClientStub struct {
	appstyped.DeploymentInterface
	object  *appsv1.Deployment
	actions int
}

func (client *deploymentClientStub) Create(_ context.Context, deployment *appsv1.Deployment, _ metav1.CreateOptions) (*appsv1.Deployment, error) {
	client.actions++
	if client.object != nil {
		return nil, apierrors.NewAlreadyExists(schema.GroupResource{Group: "apps", Resource: "deployments"}, deployment.Name)
	}
	client.object = deployment.DeepCopy()
	return client.object.DeepCopy(), nil
}

func (client *deploymentClientStub) Get(_ context.Context, name string, _ metav1.GetOptions) (*appsv1.Deployment, error) {
	client.actions++
	if client.object == nil || client.object.Name != name {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: "apps", Resource: "deployments"}, name)
	}
	return client.object.DeepCopy(), nil
}

func (client *deploymentClientStub) Update(_ context.Context, deployment *appsv1.Deployment, _ metav1.UpdateOptions) (*appsv1.Deployment, error) {
	client.actions++
	client.object = deployment.DeepCopy()
	return client.object.DeepCopy(), nil
}

type appsClientStub struct {
	appstyped.AppsV1Interface
	deployments *deploymentClientStub
}

func (client *appsClientStub) Deployments(_ string) appstyped.DeploymentInterface {
	return client.deployments
}

type serviceClientStub struct {
	coretyped.ServiceInterface
	object  *corev1.Service
	actions int
}

func (client *serviceClientStub) Create(_ context.Context, service *corev1.Service, _ metav1.CreateOptions) (*corev1.Service, error) {
	client.actions++
	if client.object != nil {
		return nil, apierrors.NewAlreadyExists(schema.GroupResource{Resource: "services"}, service.Name)
	}
	client.object = service.DeepCopy()
	return client.object.DeepCopy(), nil
}

func (client *serviceClientStub) Get(_ context.Context, name string, _ metav1.GetOptions) (*corev1.Service, error) {
	client.actions++
	if client.object == nil || client.object.Name != name {
		return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "services"}, name)
	}
	return client.object.DeepCopy(), nil
}

func (client *serviceClientStub) Update(_ context.Context, service *corev1.Service, _ metav1.UpdateOptions) (*corev1.Service, error) {
	client.actions++
	client.object = service.DeepCopy()
	return client.object.DeepCopy(), nil
}

type coreClientStub struct {
	coretyped.CoreV1Interface
	services *serviceClientStub
}

func (client *coreClientStub) Services(_ string) coretyped.ServiceInterface {
	return client.services
}

type kubeClientStub struct {
	kubernetes.Interface
	apps *appsClientStub
	core *coreClientStub
}

func (client *kubeClientStub) AppsV1() appstyped.AppsV1Interface { return client.apps }
func (client *kubeClientStub) CoreV1() coretyped.CoreV1Interface { return client.core }

func newKubeClientStub(deployment *appsv1.Deployment, service *corev1.Service) (*kubeClientStub, *deploymentClientStub, *serviceClientStub) {
	deployments := &deploymentClientStub{object: deployment}
	services := &serviceClientStub{object: service}
	return &kubeClientStub{
		apps: &appsClientStub{deployments: deployments},
		core: &coreClientStub{services: services},
	}, deployments, services
}

func testRuntimeService() *sednav1.RuntimeService {
	return &sednav1.RuntimeService{
		TypeMeta: metav1.TypeMeta{APIVersion: sednav1.SchemeGroupVersion.String(), Kind: KindName},
		ObjectMeta: metav1.ObjectMeta{
			Name:       "processor-face-edge1-r7",
			Namespace:  "dayu",
			UID:        types.UID("runtime-uid"),
			Generation: 1,
		},
		Spec: sednav1.RuntimeServiceSpec{
			InstallID:          "dayu",
			DeploymentRevision: 7,
			Component:          "processor",
			LogicalService:     "face/detection",
			TargetNode:         "edge1",
			PodTemplate: sednav1.RuntimePodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "processor",
						Image: "example/processor:v1",
						Ports: []corev1.ContainerPort{{ContainerPort: 9000, Protocol: corev1.ProtocolTCP}},
					}},
				},
			},
			Endpoint: &sednav1.RuntimeServiceEndpointSpec{Port: 9000},
		},
	}
}

func TestDesiredResourcesAreOneToOneAndMeshManaged(t *testing.T) {
	runtimeService := testRuntimeService()
	runtimeService.Spec.PodTemplate.Labels = map[string]string{
		"app.kubernetes.io/managed-by": "dayu-backend",
		LabelMeshManaged:               "true",
	}
	runtimeService.Spec.PodTemplate.Annotations = map[string]string{
		"dayu.io/user-note": "preserved",
	}
	if err := validateRuntimeService(runtimeService); err != nil {
		t.Fatalf("valid caller pod-template metadata was rejected: %v", err)
	}
	deployment := desiredDeployment(runtimeService)
	networkService := desiredService(runtimeService)

	if deployment.Name != runtimeService.Name || networkService.Name != runtimeService.Name {
		t.Fatalf("child names must equal RuntimeService name")
	}
	if deployment.Spec.Replicas == nil || *deployment.Spec.Replicas != 1 {
		t.Fatalf("deployment must have exactly one replica")
	}
	if deployment.Spec.Strategy.Type != appsv1.RecreateDeploymentStrategyType {
		t.Fatalf("deployment strategy must prevent multiple revision endpoints")
	}
	if deployment.Spec.Template.Spec.NodeName != runtimeService.Spec.TargetNode {
		t.Fatalf("deployment did not bind target node")
	}
	if deployment.Spec.Template.Spec.AutomountServiceAccountToken == nil ||
		*deployment.Spec.Template.Spec.AutomountServiceAccountToken {
		t.Fatalf("runtime pod must not automount a Kubernetes token")
	}
	if !metav1.IsControlledBy(deployment, runtimeService) || !metav1.IsControlledBy(networkService, runtimeService) {
		t.Fatalf("children must have a valid RuntimeService controller owner reference")
	}
	if !hasIdentityLabels(deployment.Labels, identityLabels(runtimeService)) ||
		!hasIdentityLabels(deployment.Spec.Template.Labels, identityLabels(runtimeService)) ||
		!hasIdentityLabels(networkService.Labels, identityLabels(runtimeService)) {
		t.Fatalf("children are missing the mesh identity labels")
	}
	if deployment.Spec.Template.Labels["app.kubernetes.io/managed-by"] != "dayu-backend" ||
		deployment.Spec.Template.Annotations["dayu.io/user-note"] != "preserved" {
		t.Fatalf("caller pod-template metadata was not preserved: labels=%v annotations=%v",
			deployment.Spec.Template.Labels, deployment.Spec.Template.Annotations)
	}
	if networkService.Spec.Type != corev1.ServiceTypeClusterIP || len(networkService.Spec.Ports) != 1 {
		t.Fatalf("endpoint must be one ClusterIP Service port")
	}
	port := networkService.Spec.Ports[0]
	if port.Name != runtimePortName || port.Protocol != corev1.ProtocolTCP ||
		port.Port != 9000 || port.TargetPort.IntVal != 9000 || port.NodePort != 0 {
		t.Fatalf("unexpected fixed endpoint contract: %#v", port)
	}
	if networkService.Annotations[AnnotationLogicalService] != "face/detection" ||
		networkService.Annotations[AnnotationTargetNode] != "edge1" {
		t.Fatalf("full identities must be preserved as annotations")
	}
}

func TestDesiredDeploymentAlwaysOverlaysControllerIdentity(t *testing.T) {
	runtimeService := testRuntimeService()
	runtimeService.Spec.PodTemplate.Labels = map[string]string{
		LabelMeshManaged:       "false",
		LabelRuntimeServiceUID: "caller-value",
	}
	templateLabels := desiredDeployment(runtimeService).Spec.Template.Labels
	if templateLabels[LabelMeshManaged] != "true" ||
		templateLabels[LabelRuntimeServiceUID] != string(runtimeService.UID) ||
		!hasIdentityLabels(templateLabels, identityLabels(runtimeService)) {
		t.Fatalf("controller identity did not override caller labels: %v", templateLabels)
	}
}

func TestValidateRuntimeServiceRejectsUnsafeOrUnreachableSpecs(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*sednav1.RuntimeService)
	}{
		{name: "long runtime name", mutate: func(service *sednav1.RuntimeService) {
			service.Name = "a-runtime-service-name-that-is-intentionally-longer-than-sixty-three-characters"
		}},
		{name: "invalid label identity", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.InstallID = "contains/a/slash"
		}},
		{name: "empty install identity", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.InstallID = ""
		}},
		{name: "empty component", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.Component = ""
		}},
		{name: "conflicting node", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.PodTemplate.Spec.NodeName = "edge2"
		}},
		{name: "service account token", mutate: func(service *sednav1.RuntimeService) {
			value := true
			service.Spec.PodTemplate.Spec.AutomountServiceAccountToken = &value
		}},
		{name: "projected service account token", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.PodTemplate.Spec.Volumes = []corev1.Volume{{
				Name: "api-token",
				VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{Sources: []corev1.VolumeProjection{{
					ServiceAccountToken: &corev1.ServiceAccountTokenProjection{Path: "token"},
				}}}},
			}}
		}},
		{name: "reserved label conflict", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.PodTemplate.Labels = map[string]string{LabelRuntimeID: "other"}
		}},
		{name: "invalid pod template label key", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.PodTemplate.Labels = map[string]string{"bad key": "value"}
		}},
		{name: "invalid pod template label value", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.PodTemplate.Labels = map[string]string{"example.com/key": "bad/value"}
		}},
		{name: "invalid pod template annotation key", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.PodTemplate.Annotations = map[string]string{"bad key": "value"}
		}},
		{name: "undeclared endpoint port", mutate: func(service *sednav1.RuntimeService) {
			service.Spec.Endpoint.Port = 9100
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service := testRuntimeService()
			test.mutate(service)
			if err := validateRuntimeService(service); err == nil {
				t.Fatalf("expected validation error")
			}
		})
	}
}

func TestEnsureResourcesIsIdempotentAndPreservesClusterIP(t *testing.T) {
	runtimeService := testRuntimeService()
	deployment := desiredDeployment(runtimeService)
	deployment.UID = types.UID("deployment-uid")
	networkService := desiredService(runtimeService)
	networkService.UID = types.UID("service-uid")
	networkService.Spec.ClusterIP = "10.96.0.44"
	networkService.Spec.ClusterIPs = []string{"10.96.0.44"}

	kubeClient, deploymentClient, serviceClient := newKubeClientStub(deployment.DeepCopy(), networkService.DeepCopy())
	deploymentIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	serviceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	if err := deploymentIndexer.Add(deployment); err != nil {
		t.Fatal(err)
	}
	if err := serviceIndexer.Add(networkService); err != nil {
		t.Fatal(err)
	}
	controller := &Controller{
		kubeClient:       kubeClient,
		deploymentLister: appslisters.NewDeploymentLister(deploymentIndexer),
		serviceLister:    corelisters.NewServiceLister(serviceIndexer),
	}

	if _, err := controller.ensureDeployment(runtimeService); err != nil {
		t.Fatalf("idempotent deployment reconcile failed: %v", err)
	}
	reconciledService, err := controller.ensureService(runtimeService)
	if err != nil {
		t.Fatalf("idempotent service reconcile failed: %v", err)
	}
	if reconciledService.Spec.ClusterIP != "10.96.0.44" {
		t.Fatalf("allocated ClusterIP was not preserved")
	}
	if actions := deploymentClient.actions + serviceClient.actions; actions != 0 {
		t.Fatalf("second reconcile must perform zero writes, actions=%d", actions)
	}
}

func TestEnsureServiceResetsRoutingPolicyDriftAndPreservesAPIAllocation(t *testing.T) {
	runtimeService := testRuntimeService()
	actual := desiredService(runtimeService)
	actual.UID = "service-uid"
	actual.Spec.ClusterIP = "10.96.0.45"
	actual.Spec.ClusterIPs = []string{"10.96.0.45"}
	actual.Spec.IPFamilies = []corev1.IPFamily{corev1.IPv4Protocol}
	policy := corev1.IPFamilyPolicySingleStack
	actual.Spec.IPFamilyPolicy = &policy
	timeout := int32(600)
	actual.Spec.SessionAffinity = corev1.ServiceAffinityClientIP
	actual.Spec.SessionAffinityConfig = &corev1.SessionAffinityConfig{ClientIP: &corev1.ClientIPConfig{TimeoutSeconds: &timeout}}
	local := corev1.ServiceInternalTrafficPolicyLocal
	actual.Spec.InternalTrafficPolicy = &local
	actual.Spec.TopologyKeys = []string{"kubernetes.io/hostname"}
	actual.Spec.ExternalIPs = []string{"192.0.2.10"}

	kubeClient, _, serviceClient := newKubeClientStub(nil, actual.DeepCopy())
	serviceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	if err := serviceIndexer.Add(actual); err != nil {
		t.Fatal(err)
	}
	controller := &Controller{kubeClient: kubeClient, serviceLister: corelisters.NewServiceLister(serviceIndexer)}
	reconciled, err := controller.ensureService(runtimeService)
	if err != nil {
		t.Fatal(err)
	}
	if serviceClient.actions != 1 {
		t.Fatalf("routing drift must cause exactly one update, actions=%d", serviceClient.actions)
	}
	if reconciled.Spec.SessionAffinity != corev1.ServiceAffinityNone || reconciled.Spec.SessionAffinityConfig != nil ||
		len(reconciled.Spec.TopologyKeys) != 0 || len(reconciled.Spec.ExternalIPs) != 0 {
		t.Fatalf("controller-owned routing policy was not restored: %#v", reconciled.Spec)
	}
	if reconciled.Spec.InternalTrafficPolicy == nil || *reconciled.Spec.InternalTrafficPolicy != corev1.ServiceInternalTrafficPolicyLocal {
		t.Fatal("feature-gated internalTrafficPolicy should preserve the API-server representation")
	}
	if reconciled.Spec.ClusterIP != "10.96.0.45" || len(reconciled.Spec.ClusterIPs) != 1 ||
		len(reconciled.Spec.IPFamilies) != 1 || reconciled.Spec.IPFamilyPolicy == nil || *reconciled.Spec.IPFamilyPolicy != policy {
		t.Fatalf("API-allocated service identity was not preserved: %#v", reconciled.Spec)
	}
}

func TestChildSpecHashAnchorsImmutabilityBeforeFirstStatusUpdate(t *testing.T) {
	runtimeService := testRuntimeService()
	deployment := desiredDeployment(runtimeService)
	deploymentIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	serviceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	if err := deploymentIndexer.Add(deployment); err != nil {
		t.Fatal(err)
	}
	controller := &Controller{
		deploymentLister: appslisters.NewDeploymentLister(deploymentIndexer),
		serviceLister:    corelisters.NewServiceLister(serviceIndexer),
	}

	mutated := runtimeService.DeepCopy()
	mutated.Spec.PodTemplate.Spec.Containers[0].Image = "example/processor:v2"
	mutatedHash, err := runtimeSpecHash(mutated)
	if err != nil {
		t.Fatal(err)
	}
	if err := controller.validateChildSpecHash(mutated, mutatedHash); err == nil {
		t.Fatal("child spec-hash did not reject mutation before status was persisted")
	}

	missingHash := deployment.DeepCopy()
	delete(missingHash.Annotations, AnnotationRuntimeSpecHash)
	if err := deploymentIndexer.Update(missingHash); err != nil {
		t.Fatal(err)
	}
	originalHash, _ := runtimeSpecHash(runtimeService)
	if err := controller.validateChildSpecHash(runtimeService, originalHash); err == nil {
		t.Fatal("controlled child without a spec-hash annotation was accepted")
	}
}

func TestEnsureHandlesInformerCreateLag(t *testing.T) {
	runtimeService := testRuntimeService()
	deployment := desiredDeployment(runtimeService)
	networkService := desiredService(runtimeService)
	kubeClient, _, _ := newKubeClientStub(deployment, networkService)
	emptyDeploymentIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	emptyServiceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	controller := &Controller{
		kubeClient:       kubeClient,
		deploymentLister: appslisters.NewDeploymentLister(emptyDeploymentIndexer),
		serviceLister:    corelisters.NewServiceLister(emptyServiceIndexer),
	}

	if _, err := controller.ensureDeployment(runtimeService); err != nil {
		t.Fatalf("AlreadyExists deployment during cache lag must converge: %v", err)
	}
	if _, err := controller.ensureService(runtimeService); err != nil {
		t.Fatalf("AlreadyExists service during cache lag must converge: %v", err)
	}
}

func TestEnsureRejectsUnanchoredChildDuringInformerCreateLag(t *testing.T) {
	for _, anchoredHash := range []string{"", "stale-spec-hash"} {
		name := anchoredHash
		if name == "" {
			name = "missing"
		}
		t.Run(name, func(t *testing.T) {
			runtimeService := testRuntimeService()
			deployment := desiredDeployment(runtimeService)
			networkService := desiredService(runtimeService)
			if anchoredHash == "" {
				delete(deployment.Annotations, AnnotationRuntimeSpecHash)
				delete(networkService.Annotations, AnnotationRuntimeSpecHash)
			} else {
				deployment.Annotations[AnnotationRuntimeSpecHash] = anchoredHash
				networkService.Annotations[AnnotationRuntimeSpecHash] = anchoredHash
			}
			kubeClient, _, _ := newKubeClientStub(deployment, networkService)
			emptyDeploymentIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			emptyServiceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			controller := &Controller{
				kubeClient:       kubeClient,
				deploymentLister: appslisters.NewDeploymentLister(emptyDeploymentIndexer),
				serviceLister:    corelisters.NewServiceLister(emptyServiceIndexer),
			}

			if _, err := controller.ensureDeployment(runtimeService); err == nil {
				t.Fatal("AlreadyExists Deployment without the exact spec hash was accepted")
			}
			if _, err := controller.ensureService(runtimeService); err == nil {
				t.Fatal("AlreadyExists Service without the exact spec hash was accepted")
			}
		})
	}
}

func TestEnsureRejectsUnownedCreateCollisionDuringInformerLag(t *testing.T) {
	runtimeService := testRuntimeService()
	deployment := desiredDeployment(runtimeService)
	deployment.OwnerReferences = nil
	networkService := desiredService(runtimeService)
	networkService.OwnerReferences = nil
	kubeClient, _, _ := newKubeClientStub(deployment, networkService)
	emptyDeploymentIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	emptyServiceIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	controller := &Controller{
		kubeClient:       kubeClient,
		deploymentLister: appslisters.NewDeploymentLister(emptyDeploymentIndexer),
		serviceLister:    corelisters.NewServiceLister(emptyServiceIndexer),
	}

	if _, err := controller.ensureDeployment(runtimeService); err == nil {
		t.Fatalf("an unowned Deployment collision must not be adopted")
	}
	if _, err := controller.ensureService(runtimeService); err == nil {
		t.Fatalf("an unowned Service collision must not be adopted")
	}
}
