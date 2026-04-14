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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:resource:shortName=ji
// +kubebuilder:subresource:status

// JointInferenceService describes the data that a jointinferenceservice resource should have
type JointInferenceService struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata"`

	Spec   JointInferenceServiceSpec   `json:"spec"`
	Status JointInferenceServiceStatus `json:"status,omitempty"`
}

// JointInferenceServiceSpec is a description of a jointinferenceservice
type JointInferenceServiceSpec struct {
	EdgeWorker  EdgeWorker  `json:"edgeWorker"`
	CloudWorker CloudWorker `json:"cloudWorker"`
}

// EdgeWorker describes the data a edge worker should have
type EdgeWorker struct {
	File              ConfFile           `json:"file"`
	Config            KubeConfig         `json:"kubeConfig,omitempty"`
	LogLevel          LogLevel           `json:"logLevel"`
	Mounts            []Mount            `json:"mounts,omitempty"`
	HardExampleMining HardExampleMining  `json:"hardExampleMining"`
	Template          v1.PodTemplateSpec `json:"template"`
}

// CloudWorker describes the data a cloud worker should have
type CloudWorker struct {
	File     ConfFile           `json:"file"`
	Config   KubeConfig         `json:"kubeConfig,omitempty"`
	LogLevel LogLevel           `json:"logLevel"`
	Mounts   []Mount            `json:"mounts,omitempty"`
	Template v1.PodTemplateSpec `json:"template"`
}

type LogLevel struct {
	Level string `json:"level"`
}

// ConfFile describes the configuration file
type ConfFile struct {
	// Path keeps backward compatibility with older yaml examples.
	Path  string   `json:"path,omitempty"`
	Paths []string `json:"paths,omitempty"`
}

// KubeConfig describes a host kubeconfig directory that should be mounted into the worker container.
type KubeConfig struct {
	Path string `json:"path,omitempty"`
}

func (in ConfFile) GetPaths() []string {
	paths := make([]string, 0, len(in.Paths)+1)
	seen := make(map[string]struct{}, len(in.Paths)+1)

	appendPath := func(path string) {
		if path == "" {
			return
		}
		if _, ok := seen[path]; ok {
			return
		}
		seen[path] = struct{}{}
		paths = append(paths, path)
	}

	if in.Path != "" {
		appendPath(in.Path)
	}
	for _, path := range in.Paths {
		appendPath(path)
	}
	return paths
}

type MountSourceType string

const (
	MountSourceTypeHostPath              MountSourceType = "hostPath"
	MountSourceTypePersistentVolumeClaim MountSourceType = "persistentVolumeClaim"
	MountSourceTypeConfigMap             MountSourceType = "configMap"
	MountSourceTypeSecret                MountSourceType = "secret"
	MountSourceTypeEmptyDir              MountSourceType = "emptyDir"
)

// Mount describes a user-defined mount that should be injected into worker containers.
type Mount struct {
	Name       string      `json:"name,omitempty"`
	Source     MountSource `json:"source"`
	Target     MountTarget `json:"target,omitempty"`
	Containers []string    `json:"containers,omitempty"`
	EnvName    string      `json:"envName,omitempty"`
}

// MountSource describes where a mount comes from.
type MountSource struct {
	Type                  MountSourceType                   `json:"type,omitempty"`
	HostPath              *HostPathMountSource              `json:"hostPath,omitempty"`
	PersistentVolumeClaim *PersistentVolumeClaimMountSource `json:"persistentVolumeClaim,omitempty"`
	ConfigMap             *ConfigMapMountSource             `json:"configMap,omitempty"`
	Secret                *SecretMountSource                `json:"secret,omitempty"`
	EmptyDir              *EmptyDirMountSource              `json:"emptyDir,omitempty"`
}

type HostPathMountSource struct {
	Path   string           `json:"path"`
	Prefix string           `json:"prefix,omitempty"`
	Type   *v1.HostPathType `json:"pathType,omitempty"`
}

type PersistentVolumeClaimMountSource struct {
	ClaimName string `json:"claimName"`
	ReadOnly  bool   `json:"readOnly,omitempty"`
}

type ConfigMapMountSource struct {
	Name     string `json:"name"`
	Optional *bool  `json:"optional,omitempty"`
}

type SecretMountSource struct {
	SecretName string `json:"secretName"`
	Optional   *bool  `json:"optional,omitempty"`
}

type EmptyDirMountSource struct {
	Medium    v1.StorageMedium `json:"medium,omitempty"`
	SizeLimit string           `json:"sizeLimit,omitempty"`
}

// MountTarget describes how a volume should be mounted into the container.
type MountTarget struct {
	Path             string                   `json:"path,omitempty"`
	ReadOnly         bool                     `json:"readOnly,omitempty"`
	SubPath          string                   `json:"subPath,omitempty"`
	MountPropagation *v1.MountPropagationMode `json:"mountPropagation,omitempty"`
}

// HardExampleMining describes the hard example algorithm to be used
type HardExampleMining struct {
	Name       string     `json:"name"`
	Parameters []ParaSpec `json:"parameters,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// JointInferenceServiceList is a list of JointInferenceServices.
type JointInferenceServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []JointInferenceService `json:"items"`
}

// JointInferenceServiceStatus represents the current state of a joint inference service.
type JointInferenceServiceStatus struct {

	// The latest available observations of a joint inference service's current state.
	// +optional
	Conditions []JointInferenceServiceCondition `json:"conditions,omitempty"`

	// Represents time when the service was acknowledged by the service controller.
	// It is not guaranteed to be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// The number of actively running workers.
	// +optional
	Active int32 `json:"active"`

	// The number of workers which reached to Failed.
	// +optional
	Failed int32 `json:"failed"`

	// Metrics of the joint inference service.
	Metrics []Metric `json:"metrics,omitempty"`
}

// JointInferenceServiceConditionType defines the condition type
type JointInferenceServiceConditionType string

// These are valid conditions of a service.
const (
	// JointInferenceServiceCondPending means the service has been accepted by the system,
	// but one or more of the workers has not been started.
	JointInferenceServiceCondPending JointInferenceServiceConditionType = "Pending"
	// JointInferenceServiceCondFailed means the service has failed its execution.
	JointInferenceServiceCondFailed JointInferenceServiceConditionType = "Failed"
	// JointInferenceServiceCondRunning means the service is running.
	JointInferenceServiceCondRunning JointInferenceServiceConditionType = "Running"
)

// JointInferenceServiceCondition describes current state of a service.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type JointInferenceServiceCondition struct {
	// Type of service condition, Complete or Failed.
	Type JointInferenceServiceConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status"`
	// last time we got an update on a given condition
	// +optional
	LastHeartbeatTime metav1.Time `json:"lastHeartbeatTime,omitempty"`
	// Last time the condition transit from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`
	// (brief) reason for the condition's last transition,
	// one-word CamelCase reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	// +optional
	Message string `json:"message,omitempty"`
}
