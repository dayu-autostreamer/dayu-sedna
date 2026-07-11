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
	"net"
	"reflect"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiMeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

type runtimeObservation struct {
	resourcesReconciled bool
	resourcesReason     string
	resourcesMessage    string

	deployment *appsv1.Deployment
	pods       []*corev1.Pod
	node       *corev1.Node
	service    *corev1.Service
	endpoints  *corev1.Endpoints
}

func conditionStatus(value bool) metav1.ConditionStatus {
	if value {
		return metav1.ConditionTrue
	}
	return metav1.ConditionFalse
}

func setRuntimeCondition(status *sednav1.RuntimeServiceStatus, generation int64, conditionType sednav1.RuntimeServiceConditionType, value bool, reason, message string) {
	apiMeta.SetStatusCondition(&status.Conditions, metav1.Condition{
		Type:               string(conditionType),
		Status:             conditionStatus(value),
		ObservedGeneration: generation,
		Reason:             reason,
		Message:            message,
	})
}

func runtimeConditionTrue(status *sednav1.RuntimeServiceStatus, conditionType sednav1.RuntimeServiceConditionType) bool {
	condition := apiMeta.FindStatusCondition(status.Conditions, string(conditionType))
	return condition != nil && condition.Status == metav1.ConditionTrue
}

func podIsReady(pod *corev1.Pod, service *sednav1.RuntimeService) bool {
	if pod == nil || pod.DeletionTimestamp != nil || pod.Spec.NodeName != service.Spec.TargetNode || pod.Status.Phase != corev1.PodRunning {
		return false
	}
	if !hasIdentityLabels(pod.Labels, identityLabels(service)) {
		return false
	}
	readyCondition := false
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			readyCondition = true
			break
		}
	}
	if !readyCondition || len(pod.Status.ContainerStatuses) != len(pod.Spec.Containers) {
		return false
	}
	for _, container := range pod.Status.ContainerStatuses {
		if !container.Ready {
			return false
		}
	}
	return true
}

func uniqueReadyPod(pods []*corev1.Pod, service *sednav1.RuntimeService) (*corev1.Pod, int) {
	var readyPod *corev1.Pod
	readyCount := 0
	for _, pod := range pods {
		if podIsReady(pod, service) {
			readyPod = pod
			readyCount++
		}
	}
	return readyPod, readyCount
}

func nodeIsReady(node *corev1.Node) bool {
	if node == nil || node.DeletionTimestamp != nil {
		return false
	}
	for _, condition := range node.Status.Conditions {
		if condition.Type == corev1.NodeReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func deploymentIsReady(deployment *appsv1.Deployment) bool {
	if deployment == nil || deployment.DeletionTimestamp != nil || deployment.Spec.Replicas == nil || *deployment.Spec.Replicas != 1 {
		return false
	}
	return deployment.Status.ObservedGeneration >= deployment.Generation &&
		deployment.Status.UpdatedReplicas == 1 &&
		deployment.Status.Replicas == 1 &&
		deployment.Status.ReadyReplicas == 1 &&
		deployment.Status.AvailableReplicas == 1 &&
		deployment.Status.UnavailableReplicas == 0
}

func serviceMatches(actual *corev1.Service, runtimeService *sednav1.RuntimeService) bool {
	desired := desiredService(runtimeService)
	if actual == nil || desired == nil || actual.DeletionTimestamp != nil {
		return false
	}
	if !metav1.IsControlledBy(actual, runtimeService) || !hasIdentityLabels(actual.Labels, identityLabels(runtimeService)) {
		return false
	}
	return actual.Spec.Type == corev1.ServiceTypeClusterIP &&
		actual.Spec.ClusterIP != "" && actual.Spec.ClusterIP != corev1.ClusterIPNone &&
		net.ParseIP(actual.Spec.ClusterIP) != nil &&
		reflect.DeepEqual(actual.Spec.Selector, desired.Spec.Selector) &&
		actual.Spec.SessionAffinity == desired.Spec.SessionAffinity &&
		reflect.DeepEqual(actual.Spec.SessionAffinityConfig, desired.Spec.SessionAffinityConfig) &&
		len(actual.Spec.Ports) == 1 &&
		actual.Spec.Ports[0].Name == desired.Spec.Ports[0].Name &&
		actual.Spec.Ports[0].Protocol == desired.Spec.Ports[0].Protocol &&
		actual.Spec.Ports[0].Port == desired.Spec.Ports[0].Port &&
		actual.Spec.Ports[0].TargetPort == desired.Spec.Ports[0].TargetPort &&
		actual.Spec.Ports[0].NodePort == 0
}

func endpointsMatch(endpoints *corev1.Endpoints, runtimeService *sednav1.RuntimeService, readyPod *corev1.Pod) (bool, int32, int32) {
	if endpoints == nil || readyPod == nil || endpoints.DeletionTimestamp != nil {
		return false, 0, 0
	}
	if !hasIdentityLabels(endpoints.Labels, identityLabels(runtimeService)) {
		return false, 0, 0
	}
	if len(endpoints.Subsets) != 1 {
		return false, 0, 0
	}

	var readyAddresses int32
	var notReadyAddresses int32
	portsValid := true
	for _, subset := range endpoints.Subsets {
		if len(subset.Ports) != 1 || len(subset.Addresses) != 1 || len(subset.NotReadyAddresses) != 0 ||
			subset.Ports[0].Name != runtimePortName ||
			subset.Ports[0].Protocol != corev1.ProtocolTCP ||
			subset.Ports[0].Port != runtimeService.Spec.Endpoint.Port {
			portsValid = false
		}
		for _, address := range subset.Addresses {
			readyAddresses++
			if net.ParseIP(address.IP) == nil || address.NodeName == nil || *address.NodeName != runtimeService.Spec.TargetNode ||
				address.TargetRef == nil || address.TargetRef.Kind != "Pod" ||
				address.TargetRef.Name != readyPod.Name || address.TargetRef.UID != readyPod.UID ||
				(address.TargetRef.Namespace != "" && address.TargetRef.Namespace != runtimeService.Namespace) {
				portsValid = false
			}
		}
		notReadyAddresses += int32(len(subset.NotReadyAddresses))
	}

	return portsValid && readyAddresses == 1 && notReadyAddresses == 0, readyAddresses, notReadyAddresses
}

func buildRuntimeStatus(service *sednav1.RuntimeService, specHash string, observation runtimeObservation) sednav1.RuntimeServiceStatus {
	status := service.DeepCopy().Status
	status.ObservedGeneration = service.Generation
	status.ObservedRevision = service.Spec.DeploymentRevision
	status.ObservedSpecHash = specHash
	status.DeploymentRef = nil
	status.PodRef = nil
	status.Endpoint = nil

	setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionSpecAccepted, true, "Accepted", "runtime specification is valid")
	setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionResourcesReconciled,
		observation.resourcesReconciled, observation.resourcesReason, observation.resourcesMessage)

	nodeReady := nodeIsReady(observation.node)
	nodeReason := "NodeNotReady"
	nodeMessage := fmt.Sprintf("target node %q is not Ready", service.Spec.TargetNode)
	if nodeReady {
		nodeReason = "NodeReady"
		nodeMessage = fmt.Sprintf("target node %q is Ready", service.Spec.TargetNode)
	}
	setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionNodeReady, nodeReady, nodeReason, nodeMessage)

	if observation.deployment != nil {
		status.DeploymentRef = &sednav1.RuntimeServiceObjectReference{Name: observation.deployment.Name, UID: observation.deployment.UID}
	}
	readyPod, readyPodCount := uniqueReadyPod(observation.pods, service)
	workloadReady := deploymentIsReady(observation.deployment) && readyPodCount == 1
	workloadReason := "WorkloadNotReady"
	workloadMessage := fmt.Sprintf("deployment and unique ready pod have not converged; ready pods=%d", readyPodCount)
	if workloadReady {
		status.PodRef = &sednav1.RuntimeServiceObjectReference{Name: readyPod.Name, UID: readyPod.UID}
		workloadReason = "WorkloadReady"
		workloadMessage = "deployment has exactly one ready pod on the target node"
	}
	setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionWorkloadReady,
		workloadReady, workloadReason, workloadMessage)

	endpointReady := service.Spec.Endpoint == nil
	endpointReason := "NotRequired"
	endpointMessage := "runtime does not expose an inbound endpoint"
	if service.Spec.Endpoint != nil {
		endpointReady = false
		endpointReason = "EndpointNotReady"
		endpointMessage = "service and core endpoints have not converged to the unique ready pod"
		if observation.service != nil {
			endpointStatus := &sednav1.RuntimeServiceEndpointStatus{
				ServiceRef: sednav1.RuntimeServiceObjectReference{Name: observation.service.Name, UID: observation.service.UID},
				DNSName:    fmt.Sprintf("%s.%s.svc.cluster.local", observation.service.Name, observation.service.Namespace),
				Port:       service.Spec.Endpoint.Port,
			}
			endpointsReady, readyAddresses, notReadyAddresses := endpointsMatch(observation.endpoints, service, readyPod)
			endpointStatus.ReadyAddresses = readyAddresses
			endpointStatus.NotReadyAddresses = notReadyAddresses
			status.Endpoint = endpointStatus
			endpointReady = serviceMatches(observation.service, service) && endpointsReady && workloadReady
			if endpointReady {
				endpointReason = "EndpointReady"
				endpointMessage = "cluster service has exactly one ready endpoint for the target pod"
			}
		}
	}
	setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionEndpointReady,
		endpointReady, endpointReason, endpointMessage)

	if service.Spec.Endpoint == nil {
		status.Activation = nil
		setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionActivated, true, "NotRequired", "runtime has no mesh endpoint")
	} else {
		activated := activationStillApplies(service, observation, status.Activation)
		activationReason := "AwaitingNodeAck"
		activationMessage := "waiting for the target node local controller to acknowledge the exact EdgeMesh route identity"
		if activated {
			activationReason = "Activated"
			activationMessage = "target node acknowledged the exact route identity for this rollout"
		}
		setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionActivated, activated, activationReason, activationMessage)
	}

	ready := observation.resourcesReconciled && nodeReady && workloadReady && endpointReady &&
		runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionActivated)
	readyReason := "ReadinessGatesPending"
	readyMessage := "one or more runtime readiness gates are false"
	if ready {
		readyReason = "Ready"
		readyMessage = "runtime resources converged and the exact rollout route was activated"
	}
	setRuntimeCondition(&status, service.Generation, sednav1.RuntimeServiceConditionReady, ready, readyReason, readyMessage)

	return status
}

// activationStillApplies makes Activated a one-time rollout barrier rather than
// a continuous health signal. Cache gaps or readiness loss do not revoke an
// exact ACK, while an observed Service or Pod replacement does.
func activationStillApplies(service *sednav1.RuntimeService, observation runtimeObservation, activationStatus *sednav1.RuntimeServiceActivationStatus) bool {
	if activationStatus == nil || activationStatus.TargetNode != service.Spec.TargetNode ||
		activationStatus.ActivatedRevision != service.Spec.DeploymentRevision ||
		activationStatus.ActivatedServiceUID == "" || activationStatus.ActivatedEndpointPodUID == "" {
		return false
	}
	if observation.service != nil && observation.service.UID != activationStatus.ActivatedServiceUID {
		return false
	}
	if pod, count := uniqueCurrentPod(observation.pods, service); count == 1 && pod.UID != activationStatus.ActivatedEndpointPodUID {
		return false
	}
	return true
}

func uniqueCurrentPod(pods []*corev1.Pod, service *sednav1.RuntimeService) (*corev1.Pod, int) {
	var current *corev1.Pod
	count := 0
	for _, pod := range pods {
		if pod == nil || pod.DeletionTimestamp != nil || pod.Spec.NodeName != service.Spec.TargetNode ||
			!hasIdentityLabels(pod.Labels, identityLabels(service)) {
			continue
		}
		current = pod
		count++
	}
	return current, count
}

func podSelector(service *sednav1.RuntimeService) labels.Selector {
	return labels.SelectorFromSet(selectorLabels(service))
}
