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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:resource:shortName=rts
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Revision",type=integer,format=int64,JSONPath=`.spec.deploymentRevision`
// +kubebuilder:printcolumn:name="Node",type=string,JSONPath=`.spec.targetNode`
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`

// RuntimeService describes one immutable, revision-scoped runtime worker on one node.
// The controller creates exactly one single-replica Deployment and, when Endpoint is
// configured, one ClusterIP Service for the worker.
type RuntimeService struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	Spec   RuntimeServiceSpec   `json:"spec"`
	Status RuntimeServiceStatus `json:"status,omitempty"`
}

// RuntimeServiceSpec is the desired state of a single runtime worker.
type RuntimeServiceSpec struct {
	// InstallID identifies one Dayu installation sharing a Kubernetes cluster.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$`
	InstallID string `json:"installID"`

	// DeploymentRevision identifies an immutable system deployment snapshot.
	// A different pod or endpoint specification must use a new revision and CR.
	// +kubebuilder:validation:Minimum=1
	DeploymentRevision int64 `json:"deploymentRevision"`

	// Component is the Dayu runtime component, for example processor or controller.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$`
	Component string `json:"component"`

	// LogicalService is the application-level service name. It may be empty for
	// infrastructure components which do not implement a DAG service.
	// +kubebuilder:validation:MaxLength=253
	LogicalService string `json:"logicalService,omitempty"`

	// TargetNode is the exact Kubernetes node on which the worker must run.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	TargetNode string `json:"targetNode"`

	// PodTemplate is a fully rendered pod template. RuntimeService does not define a
	// second templating language for mounts, environment variables, or images.
	PodTemplate RuntimePodTemplateSpec `json:"podTemplate"`

	// Endpoint requests one mesh-visible ClusterIP Service for this worker.
	// Omit it for workers, such as generators, which accept no inbound traffic.
	// +optional
	Endpoint *RuntimeServiceEndpointSpec `json:"endpoint,omitempty"`
}

// RuntimePodTemplateSpec is the supported subset of corev1.PodTemplateSpec.
// Keeping metadata narrow makes labels and annotations explicit structural CRD
// fields instead of an empty nested ObjectMeta that the API server would prune.
// RuntimeService owns all other pod-template metadata.
type RuntimePodTemplateSpec struct {
	RuntimePodTemplateMetadata `json:"metadata,omitempty"`
	Spec                       corev1.PodSpec `json:"spec"`
}

// RuntimePodTemplateMetadata is caller metadata copied to the generated Pod.
// Controller-owned dayu.io keys are validated and overlaid during reconciliation.
type RuntimePodTemplateMetadata struct {
	// +optional
	Labels map[string]string `json:"labels,omitempty"`
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`
}

// RuntimeServiceEndpointSpec describes the single network endpoint exposed by a worker.
type RuntimeServiceEndpointSpec struct {
	// Port is the stable mesh-visible Service port.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	Port int32 `json:"port"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RuntimeServiceList is a list of RuntimeService resources.
type RuntimeServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []RuntimeService `json:"items"`
}

// RuntimeServiceConditionType identifies one independently observable readiness gate.
type RuntimeServiceConditionType string

const (
	RuntimeServiceConditionSpecAccepted        RuntimeServiceConditionType = "SpecAccepted"
	RuntimeServiceConditionResourcesReconciled RuntimeServiceConditionType = "ResourcesReconciled"
	RuntimeServiceConditionNodeReady           RuntimeServiceConditionType = "NodeReady"
	RuntimeServiceConditionWorkloadReady       RuntimeServiceConditionType = "WorkloadReady"
	RuntimeServiceConditionEndpointReady       RuntimeServiceConditionType = "EndpointReady"
	RuntimeServiceConditionActivated           RuntimeServiceConditionType = "Activated"
	RuntimeServiceConditionReady               RuntimeServiceConditionType = "Ready"
)

// RuntimeServiceStatus is the controller-owned observed state of a RuntimeService.
type RuntimeServiceStatus struct {
	// ObservedGeneration is the latest metadata.generation reflected by this status.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// ObservedRevision is the deployment revision reflected by this status.
	ObservedRevision int64 `json:"observedRevision,omitempty"`

	// ObservedSpecHash prevents a revision from silently being reused for different content.
	ObservedSpecHash string `json:"observedSpecHash,omitempty"`

	// DeploymentRef identifies the reconciled Deployment.
	// +optional
	DeploymentRef *RuntimeServiceObjectReference `json:"deploymentRef,omitempty"`

	// PodRef identifies the unique ready pod, when one exists.
	// +optional
	PodRef *RuntimeServiceObjectReference `json:"podRef,omitempty"`

	// Endpoint describes the Service and endpoint projection observed by the controller.
	// +optional
	Endpoint *RuntimeServiceEndpointStatus `json:"endpoint,omitempty"`

	// Activation records the one-time rollout barrier acknowledged through Sedna LC.
	// +optional
	Activation *RuntimeServiceActivationStatus `json:"activation,omitempty"`

	// Conditions use Kubernetes standard condition semantics and are keyed by type.
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// RuntimeServiceObjectReference is a compact immutable identity for an observed object.
type RuntimeServiceObjectReference struct {
	Name string    `json:"name"`
	UID  types.UID `json:"uid"`
}

// RuntimeServiceEndpointStatus is the observed network projection for a worker.
type RuntimeServiceEndpointStatus struct {
	ServiceRef RuntimeServiceObjectReference `json:"serviceRef"`
	DNSName    string                        `json:"dnsName"`
	Port       int32                         `json:"port"`

	ReadyAddresses    int32 `json:"readyAddresses"`
	NotReadyAddresses int32 `json:"notReadyAddresses"`
}

// RuntimeServiceActivationStatus records the exact route identity requested from
// and acknowledged by the target node's Sedna local controller.
type RuntimeServiceActivationStatus struct {
	TargetNode string `json:"targetNode"`

	RequestedRevision       int64     `json:"requestedRevision,omitempty"`
	RequestedServiceUID     types.UID `json:"requestedServiceUID,omitempty"`
	RequestedEndpointPodUID types.UID `json:"requestedEndpointPodUID,omitempty"`

	ActivatedRevision       int64     `json:"activatedRevision,omitempty"`
	ActivatedServiceUID     types.UID `json:"activatedServiceUID,omitempty"`
	ActivatedEndpointPodUID types.UID `json:"activatedEndpointPodUID,omitempty"`
	ActivatedLocalSequence  uint64    `json:"activatedLocalSequence,omitempty"`
	// +optional
	LastRequestTime *metav1.Time `json:"lastRequestTime,omitempty"`
	// +optional
	LastAckTime *metav1.Time `json:"lastAckTime,omitempty"`
}
