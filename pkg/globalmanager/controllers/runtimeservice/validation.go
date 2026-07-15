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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

func runtimeSpecHash(service *sednav1.RuntimeService) (string, error) {
	data, err := json.Marshal(service.Spec)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func validateRuntimeService(service *sednav1.RuntimeService) error {
	if len(utilvalidation.IsDNS1035Label(service.Name)) != 0 {
		return fmt.Errorf("metadata.name %q must be a DNS-1035 label no longer than 63 characters", service.Name)
	}
	if service.UID == "" {
		return fmt.Errorf("metadata.uid is empty")
	}
	if service.Spec.InstallID == "" {
		return fmt.Errorf("spec.installID must not be empty")
	}
	if errors := utilvalidation.IsValidLabelValue(service.Spec.InstallID); len(errors) != 0 {
		return fmt.Errorf("spec.installID %q is not a valid label value: %s", service.Spec.InstallID, errors[0])
	}
	if service.Spec.DeploymentRevision < 1 {
		return fmt.Errorf("spec.deploymentRevision must be positive")
	}
	if service.Spec.Component == "" {
		return fmt.Errorf("spec.component must not be empty")
	}
	if errors := utilvalidation.IsValidLabelValue(service.Spec.Component); len(errors) != 0 {
		return fmt.Errorf("spec.component %q is not a valid label value: %s", service.Spec.Component, errors[0])
	}
	if len(utilvalidation.IsDNS1123Subdomain(service.Spec.TargetNode)) != 0 {
		return fmt.Errorf("spec.targetNode %q is not a valid Kubernetes node name", service.Spec.TargetNode)
	}
	if len(service.Spec.LogicalService) > 253 {
		return fmt.Errorf("spec.logicalService must not exceed 253 characters")
	}
	if len(service.Spec.PodTemplate.Spec.Containers) == 0 {
		return fmt.Errorf("spec.podTemplate.spec.containers must not be empty")
	}
	if nodeName := service.Spec.PodTemplate.Spec.NodeName; nodeName != "" && nodeName != service.Spec.TargetNode {
		return fmt.Errorf("podTemplate nodeName %q conflicts with targetNode %q", nodeName, service.Spec.TargetNode)
	}
	if restartPolicy := service.Spec.PodTemplate.Spec.RestartPolicy; restartPolicy != "" && restartPolicy != corev1.RestartPolicyAlways {
		return fmt.Errorf("podTemplate restartPolicy must be empty or Always")
	}
	if automount := service.Spec.PodTemplate.Spec.AutomountServiceAccountToken; automount != nil && *automount {
		return fmt.Errorf("podTemplate may not enable service account token automount")
	}
	for _, volume := range service.Spec.PodTemplate.Spec.Volumes {
		if volume.Projected == nil {
			continue
		}
		for _, source := range volume.Projected.Sources {
			if source.ServiceAccountToken != nil {
				return fmt.Errorf("podTemplate volume %q may not project a service account token", volume.Name)
			}
		}
	}

	for key, value := range service.Spec.PodTemplate.Labels {
		if errors := utilvalidation.IsQualifiedName(key); len(errors) != 0 {
			return fmt.Errorf("podTemplate label key %q is invalid: %s", key, errors[0])
		}
		if errors := utilvalidation.IsValidLabelValue(value); len(errors) != 0 {
			return fmt.Errorf("podTemplate label %q has invalid value %q: %s", key, value, errors[0])
		}
	}
	for key := range service.Spec.PodTemplate.Annotations {
		if errors := utilvalidation.IsQualifiedName(key); len(errors) != 0 {
			return fmt.Errorf("podTemplate annotation key %q is invalid: %s", key, errors[0])
		}
	}

	expectedLabels := identityLabels(service)
	for _, key := range identityLabelKeys {
		if value, found := service.Spec.PodTemplate.Labels[key]; found && value != expectedLabels[key] {
			return fmt.Errorf("podTemplate label %q conflicts with controller-owned value", key)
		}
	}
	expectedAnnotations := identityAnnotations(service)
	for key, expected := range expectedAnnotations {
		if value, found := service.Spec.PodTemplate.Annotations[key]; found && value != expected {
			return fmt.Errorf("podTemplate annotation %q conflicts with controller-owned value", key)
		}
	}

	if endpoint := service.Spec.Endpoint; endpoint != nil {
		if endpoint.Port < 1 || endpoint.Port > 65535 {
			return fmt.Errorf("endpoint port %d is outside 1..65535", endpoint.Port)
		}
		declaredTargetPort := false
		for _, container := range service.Spec.PodTemplate.Spec.Containers {
			for _, port := range container.Ports {
				protocol := port.Protocol
				if protocol == "" {
					protocol = corev1.ProtocolTCP
				}
				if port.ContainerPort == endpoint.Port && protocol == corev1.ProtocolTCP {
					declaredTargetPort = true
					break
				}
			}
		}
		if !declaredTargetPort {
			return fmt.Errorf("endpoint port %d is not declared as a TCP container port", endpoint.Port)
		}
	}

	return nil
}
