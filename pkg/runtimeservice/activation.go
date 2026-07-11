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

// Package runtimeservice contains the minimal wire contract shared by the
// Sedna global and local controllers. It intentionally contains no PodTemplate
// or credentials.
package runtimeservice

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const (
	KindName = "RuntimeService"

	RouteStateApplied  = "APPLIED"
	RouteStateDegraded = "DEGRADED"

	RouteSourceSynced   = "SYNCED"
	RouteSourceSyncing  = "SYNCING"
	RouteSourceDegraded = "DEGRADED"
)

// ActivationRequest asks one target-node LC to verify one exact EdgeMesh route.
type ActivationRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	RuntimeServiceUID  types.UID `json:"runtimeServiceUID"`
	ServiceUID         types.UID `json:"serviceUID"`
	EndpointPodUID     types.UID `json:"endpointPodUID"`
	DeploymentRevision int64     `json:"deploymentRevision"`
	RuntimeID          string    `json:"runtimeID"`
	TargetNode         string    `json:"targetNode"`
}

// RouteStatus is returned by the node-local EdgeMesh loopback status API.
type RouteStatus struct {
	ServiceUID         types.UID `json:"serviceUID"`
	RuntimeServiceUID  types.UID `json:"runtimeServiceUID"`
	EndpointPodUID     types.UID `json:"endpointPodUID"`
	DeploymentRevision int64     `json:"deploymentRevision"`
	RuntimeID          string    `json:"runtimeID"`
	State              string    `json:"state"`
	LocalSequence      uint64    `json:"localSequence"`
	SourceState        string    `json:"sourceState"`
	TargetNode         string    `json:"targetNode"`
	Reason             string    `json:"reason,omitempty"`
	ObservedAt         time.Time `json:"observedAt"`
}

// ActivationAck is the exact route identity acknowledged by a local controller.
type ActivationAck struct {
	RuntimeServiceUID  types.UID `json:"runtimeServiceUID"`
	ServiceUID         types.UID `json:"serviceUID"`
	EndpointPodUID     types.UID `json:"endpointPodUID"`
	DeploymentRevision int64     `json:"deploymentRevision"`
	RuntimeID          string    `json:"runtimeID"`
	TargetNode         string    `json:"targetNode"`
	LocalSequence      uint64    `json:"localSequence"`
	ObservedAt         time.Time `json:"observedAt"`
}
