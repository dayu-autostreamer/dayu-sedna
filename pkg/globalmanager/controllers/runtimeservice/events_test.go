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

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

func TestEnqueueOwnedFallsBackToOwnerReferenceForStaleUIDLabel(t *testing.T) {
	runtimeService := testRuntimeService()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{
		runtimeServiceUIDIndex: func(obj interface{}) ([]string, error) {
			return []string{string(obj.(*sednav1.RuntimeService).UID)}, nil
		},
	})
	if err := indexer.Add(runtimeService); err != nil {
		t.Fatal(err)
	}
	queue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "runtime-events-test")
	defer queue.ShutDown()
	controller := &Controller{runtimeIndexer: indexer, queue: queue}
	child := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{
		Name:      runtimeService.Name,
		Namespace: runtimeService.Namespace,
		Labels:    map[string]string{LabelRuntimeServiceUID: "stale-runtime-uid"},
		OwnerReferences: []metav1.OwnerReference{
			*metav1.NewControllerRef(runtimeService, runtimeServiceGVK),
		},
	}}

	controller.enqueueOwned(child)
	item, shutdown := queue.Get()
	if shutdown {
		t.Fatalf("queue shut down unexpectedly")
	}
	defer queue.Done(item)
	if item != runtimeService.Namespace+"/"+runtimeService.Name {
		t.Fatalf("unexpected owner fallback key %v", item)
	}
}

func TestEnqueueOwnedUpdateWakesOwnerWhenNewIdentityIsRemoved(t *testing.T) {
	runtimeService := testRuntimeService()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{
		runtimeServiceUIDIndex: func(obj interface{}) ([]string, error) {
			return []string{string(obj.(*sednav1.RuntimeService).UID)}, nil
		},
	})
	if err := indexer.Add(runtimeService); err != nil {
		t.Fatal(err)
	}
	queue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "runtime-events-identity-removal-test")
	defer queue.ShutDown()
	controller := &Controller{runtimeIndexer: indexer, queue: queue}
	oldObject := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{
		Name: runtimeService.Name, Namespace: runtimeService.Namespace, ResourceVersion: "1",
		Labels:          map[string]string{LabelRuntimeServiceUID: string(runtimeService.UID)},
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(runtimeService, runtimeServiceGVK)},
	}}
	newObject := oldObject.DeepCopy()
	newObject.ResourceVersion = "2"
	newObject.Labels = nil
	newObject.OwnerReferences = nil

	controller.enqueueOwnedUpdate(oldObject, newObject)
	if queue.Len() != 1 {
		t.Fatalf("old owner identity was not enqueued exactly once, queue length=%d", queue.Len())
	}
}
