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
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"

	clienttypes "github.com/dayu-autostreamer/dayu-sedna/pkg/localcontroller/gmclient"
	workertypes "github.com/dayu-autostreamer/dayu-sedna/pkg/localcontroller/worker"
	activation "github.com/dayu-autostreamer/dayu-sedna/pkg/runtimeservice"
)

const (
	KindName = "runtimeservice"

	defaultEdgeMeshStatusURL = "http://127.0.0.1:10551"
	defaultPollInterval      = time.Second
	defaultActivationLease   = 2 * time.Minute
	defaultHTTPTimeout       = 2 * time.Second
)

type activationPoller struct {
	request activation.ActivationRequest
	cancel  context.CancelFunc

	mu        sync.Mutex
	expiresAt time.Time
	completed bool
}

type activationTombstone struct {
	request   activation.ActivationRequest
	expiresAt time.Time
}

type activationTombstoneKey struct {
	resource          string
	runtimeServiceUID string
}

func (p *activationPoller) renew(now time.Time, lease time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.expiresAt = now.Add(lease)
}

func (p *activationPoller) reactivate(now time.Time, lease time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.expiresAt = now.Add(lease)
	p.completed = false
}

func (p *activationPoller) state() (time.Time, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.expiresAt, p.completed
}

func (p *activationPoller) markCompleted() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.completed = true
}

func (p *activationPoller) markIncomplete() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.completed = false
}

// Manager verifies one exact EdgeMesh route through the node-local loopback API.
type Manager struct {
	Client   clienttypes.ClientI
	NodeName string

	httpClient    *http.Client
	statusBaseURL string
	pollInterval  time.Duration
	lease         time.Duration
	now           func() time.Time

	mu         sync.Mutex
	pollers    map[string]*activationPoller
	tombstones map[activationTombstoneKey]activationTombstone
}

// New creates the RuntimeService activation manager for one Sedna LC node.
func New(client clienttypes.ClientI, nodeName string) *Manager {
	return &Manager{
		Client:        client,
		NodeName:      nodeName,
		httpClient:    &http.Client{Timeout: defaultHTTPTimeout},
		statusBaseURL: defaultEdgeMeshStatusURL,
		pollInterval:  defaultPollInterval,
		lease:         defaultActivationLease,
		now:           time.Now,
		pollers:       make(map[string]*activationPoller),
		tombstones:    make(map[activationTombstoneKey]activationTombstone),
	}
}

func (m *Manager) Start() error {
	return nil
}

func activationKey(namespace, name string) string {
	return namespace + "/" + name
}

func tombstoneKey(request activation.ActivationRequest) activationTombstoneKey {
	return activationTombstoneKey{
		resource:          activationKey(request.Namespace, request.Name),
		runtimeServiceUID: string(request.RuntimeServiceUID),
	}
}

func sameActivation(left, right activation.ActivationRequest) bool {
	return left.Namespace == right.Namespace &&
		left.Name == right.Name &&
		left.RuntimeServiceUID == right.RuntimeServiceUID &&
		left.ServiceUID == right.ServiceUID &&
		left.EndpointPodUID == right.EndpointPodUID &&
		left.DeploymentRevision == right.DeploymentRevision &&
		left.RuntimeID == right.RuntimeID &&
		left.TargetNode == right.TargetNode
}

func sameRuntimeIdentity(left, right activation.ActivationRequest) bool {
	return left.Namespace == right.Namespace &&
		left.Name == right.Name &&
		left.RuntimeServiceUID == right.RuntimeServiceUID
}

func validateRequest(message *clienttypes.Message, request activation.ActivationRequest, nodeName string) error {
	if request.Namespace == "" || request.Name == "" || request.RuntimeServiceUID == "" ||
		request.ServiceUID == "" || request.EndpointPodUID == "" || request.DeploymentRevision < 1 ||
		request.RuntimeID != request.Name || request.TargetNode == "" {
		return fmt.Errorf("activation request is missing required identity")
	}
	if message.Header.Namespace != request.Namespace || message.Header.ResourceName != request.Name {
		return fmt.Errorf("activation request metadata does not match message header")
	}
	if nodeName != "" && request.TargetNode != nodeName {
		return fmt.Errorf("activation request targets node %q, local node is %q", request.TargetNode, nodeName)
	}
	return nil
}

func validateCancellation(message *clienttypes.Message, request activation.ActivationRequest, nodeName string) error {
	if request.Namespace == "" || request.Name == "" || request.RuntimeServiceUID == "" ||
		request.DeploymentRevision < 1 || request.RuntimeID != request.Name || request.TargetNode == "" {
		return fmt.Errorf("activation cancellation is missing required identity")
	}
	if message.Header.Namespace != request.Namespace || message.Header.ResourceName != request.Name {
		return fmt.Errorf("activation cancellation metadata does not match message header")
	}
	if nodeName != "" && request.TargetNode != nodeName {
		return fmt.Errorf("activation cancellation targets node %q, local node is %q", request.TargetNode, nodeName)
	}
	return nil
}

func (m *Manager) pruneTombstonesLocked(now time.Time) {
	for key, tombstone := range m.tombstones {
		if !now.Before(tombstone.expiresAt) {
			delete(m.tombstones, key)
		}
	}
}

// Insert starts, renews, or atomically replaces an activation poller.
func (m *Manager) Insert(message *clienttypes.Message) error {
	var request activation.ActivationRequest
	if err := json.Unmarshal(message.Content, &request); err != nil {
		return fmt.Errorf("decode RuntimeService activation request: %w", err)
	}
	if err := validateRequest(message, request, m.NodeName); err != nil {
		return err
	}

	key := activationKey(request.Namespace, request.Name)
	now := m.now()
	m.mu.Lock()
	m.pruneTombstonesLocked(now)
	if tombstone, found := m.tombstones[tombstoneKey(request)]; found && sameRuntimeIdentity(tombstone.request, request) {
		m.mu.Unlock()
		return nil
	}
	if current, found := m.pollers[key]; found {
		if sameActivation(current.request, request) {
			current.reactivate(now, m.lease)
			m.mu.Unlock()
			return nil
		}
		current.cancel()
		delete(m.pollers, key)
	}
	ctx, cancel := context.WithCancel(context.Background())
	poller := &activationPoller{request: request, cancel: cancel}
	poller.renew(now, m.lease)
	m.pollers[key] = poller
	m.mu.Unlock()

	go m.runPoller(ctx, key, poller)
	return nil
}

// Delete records an identity-aware tombstone before cancelling the matching
// poller. This makes concurrently dispatched websocket Add/Delete messages
// deterministic and prevents a stale Insert from resurrecting a deleted CR.
func (m *Manager) Delete(message *clienttypes.Message) error {
	var request activation.ActivationRequest
	if err := json.Unmarshal(message.Content, &request); err != nil {
		return fmt.Errorf("decode RuntimeService activation cancellation: %w", err)
	}
	if err := validateCancellation(message, request, m.NodeName); err != nil {
		return err
	}

	key := activationKey(request.Namespace, request.Name)
	identityKey := tombstoneKey(request)
	now := m.now()
	expiresAt := now.Add(m.lease)
	m.mu.Lock()
	m.pruneTombstonesLocked(now)
	m.tombstones[identityKey] = activationTombstone{request: request, expiresAt: expiresAt}
	if poller, found := m.pollers[key]; found && sameRuntimeIdentity(poller.request, request) {
		poller.cancel()
		delete(m.pollers, key)
	}
	m.mu.Unlock()
	time.AfterFunc(m.lease, func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		if tombstone, found := m.tombstones[identityKey]; found &&
			tombstone.expiresAt.Equal(expiresAt) && sameRuntimeIdentity(tombstone.request, request) {
			delete(m.tombstones, identityKey)
		}
	})
	return nil
}

func (m *Manager) runPoller(ctx context.Context, key string, poller *activationPoller) {
	ticker := time.NewTicker(m.pollInterval)
	defer ticker.Stop()
	defer func() {
		m.mu.Lock()
		if current, found := m.pollers[key]; found && current == poller {
			delete(m.pollers, key)
		}
		m.mu.Unlock()
	}()

	for {
		expiresAt, completed := poller.state()
		if !m.now().Before(expiresAt) {
			return
		}
		if !completed {
			route, ready, err := m.readRoute(ctx, poller.request)
			if err != nil {
				klog.V(4).Infof("RuntimeService route %s not activated yet: %v", key, err)
			} else if ready {
				if err := m.sendAckIfCurrent(ctx, key, poller, route); err != nil {
					klog.Warningf("failed to acknowledge RuntimeService route %s: %v", key, err)
				}
			}
		}

		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (m *Manager) sendAckIfCurrent(ctx context.Context, key string, poller *activationPoller, route activation.RouteStatus) error {
	m.mu.Lock()
	if ctx.Err() != nil || m.pollers[key] != poller {
		m.mu.Unlock()
		return nil
	}
	// Reserve this ACK while the poller is still current, but never hold the
	// lifecycle lock across the potentially blocking websocket send queue.
	// A cancellation can therefore complete immediately; a message already in
	// flight remains harmless because the GM validates every rollout identity.
	poller.markCompleted()
	m.mu.Unlock()

	if err := m.sendAck(poller.request, route); err != nil {
		m.mu.Lock()
		if ctx.Err() == nil && m.pollers[key] == poller {
			poller.markIncomplete()
		}
		m.mu.Unlock()
		return err
	}
	return nil
}

func (m *Manager) readRoute(ctx context.Context, request activation.ActivationRequest) (activation.RouteStatus, bool, error) {
	var route activation.RouteStatus
	endpoint := strings.TrimRight(m.statusBaseURL, "/") + "/v1/routes/" + url.PathEscape(string(request.ServiceUID))
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return route, false, err
	}
	response, err := m.httpClient.Do(httpRequest)
	if err != nil {
		return route, false, err
	}
	defer response.Body.Close()

	if response.StatusCode == http.StatusNotFound || response.StatusCode == http.StatusServiceUnavailable {
		return route, false, nil
	}
	if response.StatusCode != http.StatusOK {
		return route, false, fmt.Errorf("EdgeMesh status returned HTTP %d", response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(&route); err != nil {
		return route, false, err
	}

	exact := route.ServiceUID == request.ServiceUID &&
		route.RuntimeServiceUID == request.RuntimeServiceUID &&
		route.EndpointPodUID == request.EndpointPodUID &&
		route.DeploymentRevision == request.DeploymentRevision &&
		route.RuntimeID == request.RuntimeID &&
		route.TargetNode == request.TargetNode &&
		route.State == activation.RouteStateApplied &&
		route.SourceState == activation.RouteSourceSynced &&
		route.LocalSequence > 0 && !route.ObservedAt.IsZero()
	return route, exact, nil
}

func (m *Manager) sendAck(request activation.ActivationRequest, route activation.RouteStatus) error {
	ack := activation.ActivationAck{
		RuntimeServiceUID:  request.RuntimeServiceUID,
		ServiceUID:         request.ServiceUID,
		EndpointPodUID:     request.EndpointPodUID,
		DeploymentRevision: request.DeploymentRevision,
		RuntimeID:          request.RuntimeID,
		TargetNode:         request.TargetNode,
		LocalSequence:      route.LocalSequence,
		ObservedAt:         route.ObservedAt,
	}
	header := clienttypes.MessageHeader{
		Namespace:    request.Namespace,
		ResourceKind: KindName,
		ResourceName: request.Name,
		Operation:    clienttypes.StatusOperation,
	}
	return m.Client.WriteMessage(ack, header)
}

func (m *Manager) GetName() string {
	return KindName
}

// AddWorkerMessage is intentionally a no-op: activation is not worker telemetry.
func (m *Manager) AddWorkerMessage(_ workertypes.MessageContent) {}
