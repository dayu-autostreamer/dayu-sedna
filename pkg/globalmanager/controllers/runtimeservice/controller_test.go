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
	"encoding/json"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
	sednalisters "github.com/dayu-autostreamer/dayu-sedna/pkg/client/listers/sedna/v1alpha1"
	activation "github.com/dayu-autostreamer/dayu-sedna/pkg/runtimeservice"
)

func TestSyncNotFoundClearsLateActivationAcks(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	controller := &Controller{
		runtimeLister: sednalisters.NewRuntimeServiceLister(indexer),
		pendingAcks: map[string]activation.ActivationAck{
			"dayu/deleted": {RuntimeServiceUID: "late"},
		},
		acceptedAcks: map[string]activation.ActivationAck{
			"dayu/deleted": {RuntimeServiceUID: "accepted"},
		},
	}
	if err := controller.sync("dayu/deleted"); err != nil {
		t.Fatal(err)
	}
	if len(controller.pendingAcks) != 0 || len(controller.acceptedAcks) != 0 {
		t.Fatal("NotFound RuntimeService retained late ACK cache entries")
	}
}

func TestActivationAckRequiresConnectionSourceNodeAndCompleteIdentity(t *testing.T) {
	runtimeService := testRuntimeService()
	observation := readyObservation(runtimeService)
	ack := activation.ActivationAck{
		RuntimeServiceUID:  runtimeService.UID,
		ServiceUID:         observation.service.UID,
		EndpointPodUID:     observation.pods[0].UID,
		DeploymentRevision: runtimeService.Spec.DeploymentRevision,
		RuntimeID:          runtimeService.Name,
		TargetNode:         runtimeService.Spec.TargetNode,
		LocalSequence:      9,
		ObservedAt:         time.Now(),
	}
	content, err := json.Marshal(ack)
	if err != nil {
		t.Fatal(err)
	}
	queue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "runtime-ack-test")
	defer queue.ShutDown()
	controller := &Controller{
		queue:        queue,
		pendingAcks:  make(map[string]activation.ActivationAck),
		acceptedAcks: make(map[string]activation.ActivationAck),
	}
	if err := controller.updateFromEdge("", runtimeService.Name, runtimeService.Namespace, "status", content); err == nil {
		t.Fatal("ACK without a websocket source node was accepted")
	}
	if err := controller.updateFromEdge("other-node", runtimeService.Name, runtimeService.Namespace, "status", content); err == nil {
		t.Fatal("ACK from a different websocket node connection was accepted")
	}
	if len(controller.pendingAcks) != 0 {
		t.Fatal("rejected source-node ACK entered pending cache")
	}
	if err := controller.updateFromEdge(runtimeService.Spec.TargetNode, runtimeService.Name, runtimeService.Namespace, "status", content); err != nil {
		t.Fatal(err)
	}
	if len(controller.pendingAcks) != 1 || queue.Len() != 1 {
		t.Fatal("valid exact ACK was not queued")
	}

	incomplete := ack
	incomplete.LocalSequence = 0
	content, _ = json.Marshal(incomplete)
	if err := controller.updateFromEdge(runtimeService.Spec.TargetNode, runtimeService.Name, runtimeService.Namespace, "status", content); err == nil {
		t.Fatal("ACK without a local sequence was accepted")
	}
}

func TestTerminatingRuntimeOnlyCancelsAndNeverReconcilesChildren(t *testing.T) {
	runtimeService := testRuntimeService()
	now := metav1.Now()
	runtimeService.DeletionTimestamp = &now
	runtimeService.Status.Endpoint = &sednav1.RuntimeServiceEndpointStatus{
		ServiceRef: sednav1.RuntimeServiceObjectReference{UID: "service-uid"},
	}
	runtimeService.Status.PodRef = &sednav1.RuntimeServiceObjectReference{UID: "pod-uid"}
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	if err := indexer.Add(runtimeService); err != nil {
		t.Fatal(err)
	}
	cancelled := 0
	key := runtimeService.Namespace + "/" + runtimeService.Name
	controller := &Controller{
		runtimeLister: sednalisters.NewRuntimeServiceLister(indexer),
		downstreamSend: func(node string, event watch.EventType, object interface{}) error {
			if node != runtimeService.Spec.TargetNode || event != watch.Deleted {
				t.Fatalf("unexpected cancellation target/event: %s %s", node, event)
			}
			cancelled++
			return nil
		},
		pendingAcks:  map[string]activation.ActivationAck{key: {}},
		acceptedAcks: map[string]activation.ActivationAck{key: {}},
	}
	if err := controller.sync(key); err != nil {
		t.Fatal(err)
	}
	if cancelled != 1 || len(controller.pendingAcks) != 0 || len(controller.acceptedAcks) != 0 {
		t.Fatalf("terminating reconcile did not reduce to one cancellation: cancelled=%d", cancelled)
	}
}
