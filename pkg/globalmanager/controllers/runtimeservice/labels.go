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
	"strconv"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

const (
	LabelMeshManaged        = "dayu.io/mesh-managed"
	LabelInstallID          = "dayu.io/install-id"
	LabelDeploymentRevision = "dayu.io/deployment-revision"
	LabelRuntimeID          = "dayu.io/runtime-id"
	LabelComponent          = "dayu.io/component"
	LabelRuntimeServiceUID  = "dayu.io/runtime-service-uid"

	AnnotationLogicalService  = "dayu.io/logical-service"
	AnnotationTargetNode      = "dayu.io/target-node"
	AnnotationRuntimeSpecHash = "dayu.io/runtime-spec-hash"
)

var identityLabelKeys = []string{
	LabelMeshManaged,
	LabelInstallID,
	LabelDeploymentRevision,
	LabelRuntimeID,
	LabelComponent,
	LabelRuntimeServiceUID,
}

func deploymentRevisionLabel(revision int64) string {
	return strconv.FormatInt(revision, 10)
}

func identityLabels(service *sednav1.RuntimeService) map[string]string {
	return map[string]string{
		LabelMeshManaged:        "true",
		LabelInstallID:          service.Spec.InstallID,
		LabelDeploymentRevision: deploymentRevisionLabel(service.Spec.DeploymentRevision),
		LabelRuntimeID:          service.Name,
		LabelComponent:          service.Spec.Component,
		LabelRuntimeServiceUID:  string(service.UID),
	}
}

func selectorLabels(service *sednav1.RuntimeService) map[string]string {
	return map[string]string{
		LabelDeploymentRevision: deploymentRevisionLabel(service.Spec.DeploymentRevision),
		LabelRuntimeID:          service.Name,
		LabelRuntimeServiceUID:  string(service.UID),
	}
}

func identityAnnotations(service *sednav1.RuntimeService) map[string]string {
	specHash, _ := runtimeSpecHash(service)
	return map[string]string{
		AnnotationLogicalService:  service.Spec.LogicalService,
		AnnotationTargetNode:      service.Spec.TargetNode,
		AnnotationRuntimeSpecHash: specHash,
	}
}

func hasIdentityLabels(actual map[string]string, expected map[string]string) bool {
	for _, key := range identityLabelKeys {
		if actual[key] != expected[key] {
			return false
		}
	}
	return true
}

func mergeStringMap(existing, required map[string]string) map[string]string {
	result := make(map[string]string, len(existing)+len(required))
	for key, value := range existing {
		result[key] = value
	}
	for key, value := range required {
		result[key] = value
	}
	return result
}
