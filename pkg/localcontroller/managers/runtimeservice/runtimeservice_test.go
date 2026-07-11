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
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	clienttypes "github.com/dayu-autostreamer/dayu-sedna/pkg/localcontroller/gmclient"
	activation "github.com/dayu-autostreamer/dayu-sedna/pkg/runtimeservice"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func routeHTTPClient(t *testing.T, statusCode int, route func() activation.RouteStatus, inspect func(*http.Request)) *http.Client {
	t.Helper()
	return &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if inspect != nil {
			inspect(request)
		}
		var body []byte
		if route != nil {
			var err error
			body, err = json.Marshal(route())
			if err != nil {
				t.Fatal(err)
			}
		}
		return &http.Response{
			StatusCode: statusCode,
			Body:       io.NopCloser(bytes.NewReader(body)),
			Header:     make(http.Header),
		}, nil
	})}
}

type recordedWrite struct {
	body   interface{}
	header clienttypes.MessageHeader
}

type recordingClient struct {
	mu     sync.Mutex
	writes []recordedWrite
}

type blockingClient struct {
	started chan struct{}
	release chan struct{}
}

func (*blockingClient) Start() error                                         { return nil }
func (*blockingClient) Subscribe(_ clienttypes.MessageResourceHandler) error { return nil }
func (client *blockingClient) WriteMessage(interface{}, clienttypes.MessageHeader) error {
	close(client.started)
	<-client.release
	return nil
}

func (c *recordingClient) Start() error                                         { return nil }
func (c *recordingClient) Subscribe(_ clienttypes.MessageResourceHandler) error { return nil }
func (c *recordingClient) WriteMessage(body interface{}, header clienttypes.MessageHeader) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.writes = append(c.writes, recordedWrite{body: body, header: header})
	return nil
}
func (c *recordingClient) writeCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.writes)
}
func (c *recordingClient) firstWrite() recordedWrite {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.writes[0]
}

func testActivationRequest() activation.ActivationRequest {
	return activation.ActivationRequest{
		TypeMeta:           metav1.TypeMeta{APIVersion: "sedna.io/v1alpha1", Kind: activation.KindName},
		ObjectMeta:         metav1.ObjectMeta{Name: "processor-edge1-r7", Namespace: "dayu"},
		RuntimeServiceUID:  types.UID("runtime-uid"),
		ServiceUID:         types.UID("service-uid"),
		EndpointPodUID:     types.UID("pod-uid"),
		DeploymentRevision: 7,
		RuntimeID:          "processor-edge1-r7",
		TargetNode:         "edge1",
	}
}

func activationMessage(t *testing.T, request activation.ActivationRequest, operation string) *clienttypes.Message {
	t.Helper()
	content, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	return &clienttypes.Message{
		Header: clienttypes.MessageHeader{
			Namespace:    request.Namespace,
			ResourceKind: KindName,
			ResourceName: request.Name,
			Operation:    operation,
		},
		Content: content,
	}
}

func waitFor(t *testing.T, timeout time.Duration, predicate func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if predicate() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("condition did not become true within %s", timeout)
}

func TestManagerAcknowledgesOnlyExactAppliedSyncedRoute(t *testing.T) {
	request := testActivationRequest()
	route := activation.RouteStatus{
		ServiceUID:         request.ServiceUID,
		RuntimeServiceUID:  request.RuntimeServiceUID,
		EndpointPodUID:     request.EndpointPodUID,
		DeploymentRevision: request.DeploymentRevision,
		RuntimeID:          request.RuntimeID,
		State:              activation.RouteStateApplied,
		SourceState:        activation.RouteSourceSynced,
		TargetNode:         request.TargetNode,
		LocalSequence:      9,
		ObservedAt:         time.Now(),
	}
	httpClient := routeHTTPClient(t, http.StatusOK, func() activation.RouteStatus { return route }, func(httpRequest *http.Request) {
		if httpRequest.URL.Path != "/v1/routes/"+string(request.ServiceUID) {
			t.Errorf("unexpected route path %q", httpRequest.URL.Path)
		}
	})

	client := &recordingClient{}
	manager := New(client, "edge1")
	manager.httpClient = httpClient
	manager.statusBaseURL = "http://edgemesh.local"
	manager.pollInterval = 5 * time.Millisecond
	manager.lease = 200 * time.Millisecond
	if err := manager.Insert(activationMessage(t, request, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	waitFor(t, time.Second, func() bool { return client.writeCount() == 1 })

	write := client.firstWrite()
	ack, ok := write.body.(activation.ActivationAck)
	if !ok {
		t.Fatalf("unexpected ACK type %T", write.body)
	}
	if ack.ServiceUID != request.ServiceUID || ack.EndpointPodUID != request.EndpointPodUID ||
		ack.DeploymentRevision != request.DeploymentRevision || ack.LocalSequence != 9 {
		t.Fatalf("ACK did not preserve exact route identity: %#v", ack)
	}
	if write.header.Operation != clienttypes.StatusOperation ||
		write.header.ResourceKind != KindName || write.header.ResourceName != request.Name {
		t.Fatalf("unexpected ACK header: %#v", write.header)
	}
}

func TestManagerRejectsReplayedRouteForOldPod(t *testing.T) {
	request := testActivationRequest()
	route := activation.RouteStatus{
		ServiceUID:         request.ServiceUID,
		RuntimeServiceUID:  request.RuntimeServiceUID,
		EndpointPodUID:     types.UID("old-pod"),
		DeploymentRevision: request.DeploymentRevision,
		RuntimeID:          request.RuntimeID,
		State:              activation.RouteStateApplied,
		SourceState:        activation.RouteSourceSynced,
		TargetNode:         request.TargetNode,
	}
	client := &recordingClient{}
	manager := New(client, "edge1")
	manager.httpClient = routeHTTPClient(t, http.StatusOK, func() activation.RouteStatus { return route }, nil)
	manager.statusBaseURL = "http://edgemesh.local"
	manager.pollInterval = 5 * time.Millisecond
	manager.lease = 40 * time.Millisecond
	if err := manager.Insert(activationMessage(t, request, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	waitFor(t, time.Second, func() bool {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		return len(manager.pollers) == 0
	})
	if client.writeCount() != 0 {
		t.Fatalf("old Pod route must not be acknowledged")
	}
}

func TestManagerDeduplicatesReplacesCancelsAndExpiresPollers(t *testing.T) {
	client := &recordingClient{}
	manager := New(client, "edge1")
	manager.httpClient = routeHTTPClient(t, http.StatusServiceUnavailable, nil, nil)
	manager.statusBaseURL = "http://edgemesh.local"
	manager.pollInterval = 5 * time.Millisecond
	manager.lease = 50 * time.Millisecond

	first := testActivationRequest()
	firstMessage := activationMessage(t, first, clienttypes.InsertOperation)
	if err := manager.Insert(firstMessage); err != nil {
		t.Fatal(err)
	}
	key := activationKey(first.Namespace, first.Name)
	manager.mu.Lock()
	original := manager.pollers[key]
	manager.mu.Unlock()
	if original == nil {
		t.Fatalf("poller was not created")
	}

	if err := manager.Insert(firstMessage); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	if manager.pollers[key] != original || len(manager.pollers) != 1 {
		manager.mu.Unlock()
		t.Fatalf("duplicate Insert must renew one poller")
	}
	manager.mu.Unlock()

	replacement := first
	replacement.ServiceUID = types.UID("replacement-service")
	replacement.EndpointPodUID = types.UID("replacement-pod")
	if err := manager.Insert(activationMessage(t, replacement, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	replaced := manager.pollers[key]
	manager.mu.Unlock()
	if replaced == nil || replaced == original {
		t.Fatalf("identity change must atomically replace the poller")
	}

	if err := manager.Delete(activationMessage(t, replacement, clienttypes.DeleteOperation)); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	remaining := len(manager.pollers)
	manager.mu.Unlock()
	if remaining != 0 {
		t.Fatalf("Delete must immediately cancel the poller")
	}

	expiring := replacement
	expiring.Name = "processor-edge1-r8"
	expiring.RuntimeID = expiring.Name
	expiring.RuntimeServiceUID = "runtime-uid-r8"
	expiring.DeploymentRevision = 8
	if err := manager.Insert(activationMessage(t, expiring, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	waitFor(t, time.Second, func() bool {
		manager.mu.Lock()
		defer manager.mu.Unlock()
		return len(manager.pollers) == 0
	})
}

func TestManagerDeleteTombstoneIsIdentityAwareAndPreventsResurrection(t *testing.T) {
	client := &recordingClient{}
	manager := New(client, "edge1")
	manager.httpClient = routeHTTPClient(t, http.StatusServiceUnavailable, nil, nil)
	manager.statusBaseURL = "http://edgemesh.local"
	manager.pollInterval = 5 * time.Millisecond
	manager.lease = 100 * time.Millisecond

	oldRequest := testActivationRequest()
	newRequest := oldRequest
	newRequest.RuntimeServiceUID = "runtime-uid-new"
	newRequest.ServiceUID = "service-uid-new"
	newRequest.EndpointPodUID = "pod-uid-new"
	newRequest.DeploymentRevision = 8

	if err := manager.Insert(activationMessage(t, newRequest, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	key := activationKey(newRequest.Namespace, newRequest.Name)
	if err := manager.Delete(activationMessage(t, oldRequest, clienttypes.DeleteOperation)); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	current := manager.pollers[key]
	manager.mu.Unlock()
	if current == nil || current.request.RuntimeServiceUID != newRequest.RuntimeServiceUID {
		t.Fatal("stale Delete cancelled a replacement RuntimeService UID")
	}

	if err := manager.Delete(activationMessage(t, newRequest, clienttypes.DeleteOperation)); err != nil {
		t.Fatal(err)
	}
	// A still later Delete for the old incarnation must not overwrite the new
	// incarnation's deletion barrier.
	if err := manager.Delete(activationMessage(t, oldRequest, clienttypes.DeleteOperation)); err != nil {
		t.Fatal(err)
	}
	if err := manager.Insert(activationMessage(t, newRequest, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	_, resurrected := manager.pollers[key]
	manager.mu.Unlock()
	if resurrected {
		t.Fatal("late Insert resurrected an identity after its Delete tombstone")
	}
}

func TestManagerCancellationUsesRuntimeUIDAcrossIllegalSpecMutation(t *testing.T) {
	client := &recordingClient{}
	manager := New(client, "edge1")
	manager.httpClient = routeHTTPClient(t, http.StatusServiceUnavailable, nil, nil)
	manager.statusBaseURL = "http://edgemesh.local"
	manager.pollInterval = 5 * time.Millisecond
	manager.lease = 100 * time.Millisecond

	request := testActivationRequest()
	if err := manager.Insert(activationMessage(t, request, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	mutatedCancellation := request
	mutatedCancellation.DeploymentRevision++
	mutatedCancellation.ServiceUID = ""
	mutatedCancellation.EndpointPodUID = ""
	if err := manager.Delete(activationMessage(t, mutatedCancellation, clienttypes.DeleteOperation)); err != nil {
		t.Fatal(err)
	}
	manager.mu.Lock()
	_, exists := manager.pollers[activationKey(request.Namespace, request.Name)]
	manager.mu.Unlock()
	if exists {
		t.Fatal("same RuntimeService UID was not cancelled after an illegal revision mutation")
	}
}

func TestManagerDoesNotAckAfterPollerCancellation(t *testing.T) {
	request := testActivationRequest()
	started := make(chan struct{})
	release := make(chan struct{})
	route := activation.RouteStatus{
		ServiceUID: request.ServiceUID, RuntimeServiceUID: request.RuntimeServiceUID,
		EndpointPodUID: request.EndpointPodUID, DeploymentRevision: request.DeploymentRevision,
		RuntimeID: request.RuntimeID, TargetNode: request.TargetNode,
		State: activation.RouteStateApplied, SourceState: activation.RouteSourceSynced,
		LocalSequence: 1, ObservedAt: time.Now(),
	}
	client := &recordingClient{}
	manager := New(client, "edge1")
	manager.pollInterval = 5 * time.Millisecond
	manager.lease = time.Second
	manager.httpClient = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		close(started)
		<-release
		body, _ := json.Marshal(route)
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(body)), Header: make(http.Header)}, nil
	})}
	if err := manager.Insert(activationMessage(t, request, clienttypes.InsertOperation)); err != nil {
		t.Fatal(err)
	}
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("route poll did not start")
	}
	if err := manager.Delete(activationMessage(t, request, clienttypes.DeleteOperation)); err != nil {
		t.Fatal(err)
	}
	close(release)
	time.Sleep(20 * time.Millisecond)
	if client.writeCount() != 0 {
		t.Fatal("cancelled in-flight poller emitted a stale ACK")
	}
}

func TestBlockedAckQueueDoesNotBlockCancellation(t *testing.T) {
	request := testActivationRequest()
	client := &blockingClient{started: make(chan struct{}), release: make(chan struct{})}
	manager := New(client, "edge1")
	manager.lease = time.Second
	key := activationKey(request.Namespace, request.Name)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	poller := &activationPoller{request: request, cancel: cancel}
	poller.renew(time.Now(), manager.lease)
	manager.pollers[key] = poller

	done := make(chan error, 1)
	go func() {
		done <- manager.sendAckIfCurrent(ctx, key, poller, activation.RouteStatus{})
	}()
	select {
	case <-client.started:
	case <-time.After(time.Second):
		t.Fatal("ACK enqueue did not start")
	}

	deleteMessage := activationMessage(t, request, clienttypes.DeleteOperation)
	deleteDone := make(chan error, 1)
	go func() {
		deleteDone <- manager.Delete(deleteMessage)
	}()
	select {
	case err := <-deleteDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Delete blocked behind a full ACK send queue")
	}
	close(client.release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("blocked ACK enqueue did not finish")
	}
}

func TestManagerRejectsActivationForAnotherNode(t *testing.T) {
	request := testActivationRequest()
	request.TargetNode = "edge2"
	client := &recordingClient{}
	manager := New(client, "edge1")

	if err := manager.Insert(activationMessage(t, request, clienttypes.InsertOperation)); err == nil {
		t.Fatalf("activation targeted at a different node must be rejected")
	}
	manager.mu.Lock()
	pollerCount := len(manager.pollers)
	manager.mu.Unlock()
	if pollerCount != 0 || client.writeCount() != 0 {
		t.Fatalf("a rejected activation must not create a poller or ACK")
	}
}
