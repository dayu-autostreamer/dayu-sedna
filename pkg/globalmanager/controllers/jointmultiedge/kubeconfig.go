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
	"fmt"
	"path/filepath"
	"strings"

	v1 "k8s.io/api/core/v1"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

const (
	defaultKubeConfigVolumeName = "dayu-kubeconfig"
	defaultKubeConfigHostPath   = "/root/.kube"
	defaultKubeConfigMountPath  = "/root/.kube"
	defaultKubeConfigSecretKey  = "config"
	defaultKubeConfigEnvName    = "KUBECONFIG"
)

type resolvedKubeConfigSpec struct {
	enabled    bool
	mountPath  string
	readOnly   bool
	volume     v1.Volume
	configPath string
}

func injectKubeConfig(podSpec *v1.PodSpec, kubeConfig *sednav1.KubeConfigSpec) error {
	resolved, err := resolveKubeConfigSpec(kubeConfig)
	if err != nil {
		return err
	}
	if !resolved.enabled {
		return nil
	}

	if !podSpecHasMountPath(podSpec, resolved.mountPath) {
		if err := mergeVolume(podSpec, resolved.volume); err != nil {
			return err
		}
		if err := mergeVolumeMount(podSpec, v1.VolumeMount{
			Name:      resolved.volume.Name,
			MountPath: resolved.mountPath,
			ReadOnly:  resolved.readOnly,
		}, nil); err != nil {
			return err
		}
	}

	for idx := range podSpec.Containers {
		container := &podSpec.Containers[idx]
		if containerHasEnvVar(container, defaultKubeConfigEnvName) {
			continue
		}
		container.Env = append(container.Env, v1.EnvVar{
			Name:  defaultKubeConfigEnvName,
			Value: resolved.configPath,
		})
	}

	return nil
}

func resolveKubeConfigSpec(kubeConfig *sednav1.KubeConfigSpec) (resolvedKubeConfigSpec, error) {
	resolved := resolvedKubeConfigSpec{
		enabled:    true,
		mountPath:  defaultKubeConfigMountPath,
		readOnly:   true,
		configPath: filepath.Join(defaultKubeConfigMountPath, defaultKubeConfigSecretKey),
	}

	if kubeConfig != nil && kubeConfig.Enabled != nil {
		resolved.enabled = *kubeConfig.Enabled
	}
	if !resolved.enabled {
		return resolved, nil
	}

	if kubeConfig != nil && strings.TrimSpace(kubeConfig.MountPath) != "" {
		resolved.mountPath = filepath.Clean(kubeConfig.MountPath)
	}
	if strings.HasPrefix(resolved.mountPath, "~") || !filepath.IsAbs(resolved.mountPath) {
		return resolvedKubeConfigSpec{}, fmt.Errorf("kubeConfig.mountPath must be an absolute path")
	}
	resolved.configPath = filepath.Join(resolved.mountPath, defaultKubeConfigSecretKey)

	if kubeConfig != nil && strings.TrimSpace(kubeConfig.SecretName) != "" {
		if strings.TrimSpace(kubeConfig.HostPath) != "" {
			return resolvedKubeConfigSpec{}, fmt.Errorf("kubeConfig.hostPath and kubeConfig.secretName cannot be set together")
		}

		secretKey := defaultKubeConfigSecretKey
		if strings.TrimSpace(kubeConfig.SecretKey) != "" {
			secretKey = strings.TrimSpace(kubeConfig.SecretKey)
		}

		resolved.volume = v1.Volume{
			Name: defaultKubeConfigVolumeName,
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{
					SecretName: kubeConfig.SecretName,
					Items: []v1.KeyToPath{
						{
							Key:  secretKey,
							Path: defaultKubeConfigSecretKey,
						},
					},
				},
			},
		}
		return resolved, nil
	}

	hostPath := defaultKubeConfigHostPath
	if kubeConfig != nil && strings.TrimSpace(kubeConfig.HostPath) != "" {
		hostPath = filepath.Clean(kubeConfig.HostPath)
	}
	if strings.HasPrefix(hostPath, "~") {
		return resolvedKubeConfigSpec{}, fmt.Errorf("kubeConfig.hostPath does not support \"~\"; please use an absolute path")
	}
	if !filepath.IsAbs(hostPath) {
		return resolvedKubeConfigSpec{}, fmt.Errorf("kubeConfig.hostPath must be an absolute path")
	}

	resolved.volume = v1.Volume{
		Name: defaultKubeConfigVolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: hostPath,
			},
		},
	}

	return resolved, nil
}

func podSpecHasMountPath(podSpec *v1.PodSpec, mountPath string) bool {
	for _, container := range podSpec.Containers {
		for _, volumeMount := range container.VolumeMounts {
			if volumeMount.MountPath == mountPath {
				return true
			}
		}
	}
	return false
}

func containerHasEnvVar(container *v1.Container, envName string) bool {
	for _, envVar := range container.Env {
		if envVar.Name == envName {
			return true
		}
	}
	return false
}
