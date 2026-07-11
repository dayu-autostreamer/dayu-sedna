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
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

var runtimeServiceGVK = schema.GroupVersionKind{
	Group:   sednav1.SchemeGroupVersion.Group,
	Version: sednav1.SchemeGroupVersion.Version,
	Kind:    KindName,
}

const runtimePortName = "runtime"

func desiredDeployment(service *sednav1.RuntimeService) *appsv1.Deployment {
	replicas := int32(1)
	automountToken := false
	template := *service.Spec.PodTemplate.DeepCopy()
	template.Labels = mergeStringMap(template.Labels, identityLabels(service))
	template.Annotations = mergeStringMap(template.Annotations, identityAnnotations(service))
	template.Spec.NodeName = service.Spec.TargetNode
	template.Spec.RestartPolicy = corev1.RestartPolicyAlways
	template.Spec.AutomountServiceAccountToken = &automountToken

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:            service.Name,
			Namespace:       service.Namespace,
			Labels:          identityLabels(service),
			Annotations:     identityAnnotations(service),
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(service, runtimeServiceGVK)},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: selectorLabels(service)},
			Template: template,
			Strategy: appsv1.DeploymentStrategy{Type: appsv1.RecreateDeploymentStrategyType},
		},
	}
	clientgoscheme.Scheme.Default(deployment)
	return deployment
}

func desiredService(service *sednav1.RuntimeService) *corev1.Service {
	if service.Spec.Endpoint == nil {
		return nil
	}

	endpoint := service.Spec.Endpoint
	networkService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:            service.Name,
			Namespace:       service.Namespace,
			Labels:          identityLabels(service),
			Annotations:     identityAnnotations(service),
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(service, runtimeServiceGVK)},
		},
		Spec: corev1.ServiceSpec{
			Type:            corev1.ServiceTypeClusterIP,
			Selector:        selectorLabels(service),
			SessionAffinity: corev1.ServiceAffinityNone,
			Ports: []corev1.ServicePort{{
				Name:       runtimePortName,
				Protocol:   corev1.ProtocolTCP,
				Port:       endpoint.Port,
				TargetPort: intstr.FromInt(int(endpoint.Port)),
			}},
		},
	}
	clientgoscheme.Scheme.Default(networkService)
	return networkService
}
