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

package jointmultiedge

import (
	"strings"

	v1 "k8s.io/api/core/v1"
)

const (
	workerKubeConfigVolumeName = "kubeconfig-volume"
	workerKubeConfigMountPath  = "/home/data/.kube"
	workerKubeConfigEnvName    = "KUBECONFIG"
	workerKubeConfigFilePath   = "/home/data/.kube/config"
)

func injectWorkerKubeConfig(podSpec *v1.PodSpec, kubeConfigPath string) error {
	kubeConfigPath = strings.TrimSpace(kubeConfigPath)
	if kubeConfigPath == "" {
		return nil
	}

	if err := mergeVolumesAndMounts(
		podSpec,
		[]v1.Volume{{
			Name: workerKubeConfigVolumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: kubeConfigPath,
				},
			},
		}},
		[]v1.VolumeMount{{
			Name:      workerKubeConfigVolumeName,
			MountPath: workerKubeConfigMountPath,
		}},
	); err != nil {
		return err
	}
	if err := mergeEnvVar(podSpec, v1.EnvVar{
		Name:  workerKubeConfigEnvName,
		Value: workerKubeConfigFilePath,
	}, nil); err != nil {
		return err
	}

	return nil
}
