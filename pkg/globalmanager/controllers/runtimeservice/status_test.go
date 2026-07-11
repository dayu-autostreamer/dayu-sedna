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
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
	activation "github.com/dayu-autostreamer/dayu-sedna/pkg/runtimeservice"
)

func readyObservation(runtimeService *sednav1.RuntimeService) runtimeObservation {
	deployment := desiredDeployment(runtimeService)
	deployment.UID = types.UID("deployment-uid")
	deployment.Generation = 3
	deployment.Status = appsv1.DeploymentStatus{
		ObservedGeneration:  3,
		Replicas:            1,
		UpdatedReplicas:     1,
		ReadyReplicas:       1,
		AvailableReplicas:   1,
		UnavailableReplicas: 0,
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      runtimeService.Name + "-pod",
			Namespace: runtimeService.Namespace,
			UID:       types.UID("pod-uid"),
			Labels:    identityLabels(runtimeService),
		},
		Spec: corev1.PodSpec{
			NodeName:   runtimeService.Spec.TargetNode,
			Containers: runtimeService.Spec.PodTemplate.Spec.Containers,
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			Conditions: []corev1.PodCondition{{
				Type: corev1.PodReady, Status: corev1.ConditionTrue,
			}},
			ContainerStatuses: []corev1.ContainerStatus{{Name: "processor", Ready: true}},
		},
	}
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: runtimeService.Spec.TargetNode},
		Status: corev1.NodeStatus{Conditions: []corev1.NodeCondition{{
			Type: corev1.NodeReady, Status: corev1.ConditionTrue,
		}}},
	}
	networkService := desiredService(runtimeService)
	networkService.UID = types.UID("service-uid")
	networkService.Spec.ClusterIP = "10.96.0.42"
	networkService.Spec.ClusterIPs = []string{"10.96.0.42"}
	nodeName := runtimeService.Spec.TargetNode
	endpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:            runtimeService.Name,
			Namespace:       runtimeService.Namespace,
			Labels:          identityLabels(runtimeService),
			ResourceVersion: "55",
		},
		Subsets: []corev1.EndpointSubset{{
			Addresses: []corev1.EndpointAddress{{
				IP:       "10.0.0.7",
				NodeName: &nodeName,
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: runtimeService.Namespace,
					Name:      pod.Name,
					UID:       pod.UID,
				},
			}},
			Ports: []corev1.EndpointPort{{Name: runtimePortName, Port: 9000, Protocol: corev1.ProtocolTCP}},
		}},
	}
	return runtimeObservation{
		resourcesReconciled: true,
		resourcesReason:     "Reconciled",
		resourcesMessage:    "ok",
		deployment:          deployment,
		pods:                []*corev1.Pod{pod},
		node:                node,
		service:             networkService,
		endpoints:           endpoints,
	}
}

func TestStatusRequiresExactStickyActivationIdentity(t *testing.T) {
	runtimeService := testRuntimeService()
	observation := readyObservation(runtimeService)
	now := metav1.NewTime(time.Now())
	runtimeService.Status.Activation = &sednav1.RuntimeServiceActivationStatus{
		TargetNode:              runtimeService.Spec.TargetNode,
		ActivatedRevision:       runtimeService.Spec.DeploymentRevision,
		ActivatedServiceUID:     observation.service.UID,
		ActivatedEndpointPodUID: observation.pods[0].UID,
		LastAckTime:             &now,
	}
	hash, err := runtimeSpecHash(runtimeService)
	if err != nil {
		t.Fatal(err)
	}
	status := buildRuntimeStatus(runtimeService, hash, observation)
	if !runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionEndpointReady) ||
		!runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionActivated) ||
		!runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionReady) {
		t.Fatalf("fully converged exact identity must be Ready: %#v", status.Conditions)
	}
	if status.Endpoint == nil || status.Endpoint.DNSName != runtimeService.Name+".dayu.svc.cluster.local" ||
		status.Endpoint.ReadyAddresses != 1 || status.Endpoint.NotReadyAddresses != 0 {
		t.Fatalf("unexpected endpoint projection: %#v", status.Endpoint)
	}

	runtimeService.Status.Activation.ActivatedEndpointPodUID = types.UID("old-pod")
	staleStatus := buildRuntimeStatus(runtimeService, hash, observation)
	if runtimeConditionTrue(&staleStatus, sednav1.RuntimeServiceConditionActivated) ||
		runtimeConditionTrue(&staleStatus, sednav1.RuntimeServiceConditionReady) {
		t.Fatalf("replayed ACK for an old pod must not activate a replacement pod")
	}
}

func TestActivatedIsStickyAcrossTransientHealthLossButNotIdentityReplacement(t *testing.T) {
	runtimeService := testRuntimeService()
	observation := readyObservation(runtimeService)
	runtimeService.Status.Activation = &sednav1.RuntimeServiceActivationStatus{
		TargetNode:              runtimeService.Spec.TargetNode,
		ActivatedRevision:       runtimeService.Spec.DeploymentRevision,
		ActivatedServiceUID:     observation.service.UID,
		ActivatedEndpointPodUID: observation.pods[0].UID,
		ActivatedLocalSequence:  11,
	}
	hash, _ := runtimeSpecHash(runtimeService)

	transient := observation
	transient.pods = nil
	transient.endpoints = nil
	transient.deployment = nil
	status := buildRuntimeStatus(runtimeService, hash, transient)
	if !runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionActivated) {
		t.Fatal("a transient cache/readiness gap revoked the one-time activation barrier")
	}
	if runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionReady) {
		t.Fatal("dynamic Ready must fall when workload and endpoint health are unavailable")
	}

	replacementPod := readyObservation(runtimeService)
	replacementPod.pods[0].UID = "replacement-pod"
	replacementPod.endpoints.Subsets[0].Addresses[0].TargetRef.UID = "replacement-pod"
	status = buildRuntimeStatus(runtimeService, hash, replacementPod)
	if runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionActivated) {
		t.Fatal("an observed Pod replacement retained an ACK for the old Pod UID")
	}

	replacementService := readyObservation(runtimeService)
	replacementService.service.UID = "replacement-service"
	status = buildRuntimeStatus(runtimeService, hash, replacementService)
	if runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionActivated) {
		t.Fatal("an observed Service replacement retained an ACK for the old Service UID")
	}
}

func TestStatusRejectsNotReadyOrMismatchedEndpoints(t *testing.T) {
	runtimeService := testRuntimeService()
	observation := readyObservation(runtimeService)
	observation.endpoints.Subsets[0].NotReadyAddresses = observation.endpoints.Subsets[0].Addresses
	observation.endpoints.Subsets[0].Addresses = nil
	hash, _ := runtimeSpecHash(runtimeService)
	status := buildRuntimeStatus(runtimeService, hash, observation)
	if runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionEndpointReady) ||
		runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionReady) {
		t.Fatalf("not-ready Endpoints must block rollout readiness")
	}
}

func TestStatusRejectsAdditionalEmptyEndpointsSubset(t *testing.T) {
	runtimeService := testRuntimeService()
	observation := readyObservation(runtimeService)
	observation.endpoints.Subsets = append(observation.endpoints.Subsets, corev1.EndpointSubset{})
	hash, _ := runtimeSpecHash(runtimeService)
	status := buildRuntimeStatus(runtimeService, hash, observation)
	if runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionEndpointReady) ||
		runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionReady) {
		t.Fatalf("an additional empty Endpoints subset must fail the exact route contract")
	}
}

func TestStatusUsesTheSameStrictRouteContractAsEdgeMesh(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*runtimeObservation)
	}{
		{name: "missing cluster ip", mutate: func(observation *runtimeObservation) {
			observation.service.Spec.ClusterIP = ""
		}},
		{name: "headless service", mutate: func(observation *runtimeObservation) {
			observation.service.Spec.ClusterIP = corev1.ClusterIPNone
		}},
		{name: "invalid endpoint ip", mutate: func(observation *runtimeObservation) {
			observation.endpoints.Subsets[0].Addresses[0].IP = "not-an-ip"
		}},
		{name: "non pod target reference", mutate: func(observation *runtimeObservation) {
			observation.endpoints.Subsets[0].Addresses[0].TargetRef.Kind = "Node"
		}},
		{name: "wrong pod name", mutate: func(observation *runtimeObservation) {
			observation.endpoints.Subsets[0].Addresses[0].TargetRef.Name = "other-pod"
		}},
		{name: "wrong pod namespace", mutate: func(observation *runtimeObservation) {
			observation.endpoints.Subsets[0].Addresses[0].TargetRef.Namespace = "other"
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtimeService := testRuntimeService()
			observation := readyObservation(runtimeService)
			test.mutate(&observation)
			hash, _ := runtimeSpecHash(runtimeService)
			status := buildRuntimeStatus(runtimeService, hash, observation)
			if runtimeConditionTrue(&status, sednav1.RuntimeServiceConditionEndpointReady) {
				t.Fatal("Sedna accepted a route EdgeMesh would reject")
			}
		})
	}
}

func TestAcceptedAckSurvivesInformerStatusLag(t *testing.T) {
	runtimeService := testRuntimeService()
	observation := readyObservation(runtimeService)
	controller := &Controller{
		pendingAcks:  make(map[string]activation.ActivationAck),
		acceptedAcks: make(map[string]activation.ActivationAck),
	}
	key := runtimeService.Namespace + "/" + runtimeService.Name
	ack := activation.ActivationAck{
		RuntimeServiceUID:  runtimeService.UID,
		ServiceUID:         observation.service.UID,
		EndpointPodUID:     observation.pods[0].UID,
		DeploymentRevision: runtimeService.Spec.DeploymentRevision,
		RuntimeID:          runtimeService.Name,
		TargetNode:         runtimeService.Spec.TargetNode,
		ObservedAt:         time.Now(),
	}
	controller.pendingAcks[key] = ack

	first := runtimeService.DeepCopy()
	controller.applyPendingAck(first, observation)
	if first.Status.Activation == nil {
		t.Fatalf("pending ACK was not applied")
	}
	// Simulate an immediate second reconcile from a pre-ACK informer object.
	second := runtimeService.DeepCopy()
	controller.applyPendingAck(second, observation)
	if second.Status.Activation == nil ||
		second.Status.Activation.ActivatedEndpointPodUID != observation.pods[0].UID {
		t.Fatalf("accepted ACK was lost during informer status lag")
	}
	// Once the informer exposes the persisted ACK, the transient cache must be
	// retired instead of leaking one entry per Ready RuntimeService.
	third := second.DeepCopy()
	controller.applyPendingAck(third, observation)
	controller.ackMu.Lock()
	_, stillAccepted := controller.acceptedAcks[key]
	controller.ackMu.Unlock()
	if stillAccepted {
		t.Fatal("persisted activation ACK was not retired from acceptedAcks")
	}
}
