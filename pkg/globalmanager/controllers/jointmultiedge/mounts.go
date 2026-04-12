package jointmultiedge

import (
	"fmt"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
	"github.com/dayu-autostreamer/dayu-sedna/pkg/globalmanager/runtime"
)

const dataPathPrefix = "/home/data"

type renderedMount struct {
	volume        v1.Volume
	volumeMount   v1.VolumeMount
	containerName []string
	envVar        *v1.EnvVar
}

func buildLegacyFileMounts(file sednav1.ConfFile) (map[string]string, []v1.Volume, []v1.VolumeMount) {
	envs := map[string]string{
		"VOLUME_NUM": "0",
	}
	volumes := make([]v1.Volume, 0)
	volumeMounts := make([]v1.VolumeMount, 0)
	mountedPaths := make(map[string]struct{})
	volumeCounter := 0

	for _, path := range file.GetPaths() {
		dirPath := filepath.Dir(path)
		if _, exists := mountedPaths[dirPath]; exists {
			continue
		}

		mountedPaths[dirPath] = struct{}{}
		volumeName := fmt.Sprintf("volume%d", volumeCounter)
		envs[fmt.Sprintf("VOLUME_%d", volumeCounter)] = dirPath

		volumes = append(volumes, v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: dirPath,
				},
			},
		})

		volumeMounts = append(volumeMounts, v1.VolumeMount{
			Name:      volumeName,
			MountPath: filepath.Join(dataPathPrefix, volumeName),
		})

		volumeCounter++
	}

	envs["VOLUME_NUM"] = strconv.Itoa(volumeCounter)
	return envs, volumes, volumeMounts
}

func injectExplicitMounts(podSpec *v1.PodSpec, mounts []sednav1.Mount) error {
	for idx, mount := range mounts {
		rendered, err := renderMount(mount, idx)
		if err != nil {
			return err
		}

		if err := mergeVolume(podSpec, rendered.volume); err != nil {
			return err
		}
		if err := mergeVolumeMount(podSpec, rendered.volumeMount, rendered.containerName); err != nil {
			return err
		}
		if rendered.envVar != nil {
			if err := mergeEnvVar(podSpec, *rendered.envVar, rendered.containerName); err != nil {
				return err
			}
		}
	}

	return nil
}

func mergeVolumesAndMounts(podSpec *v1.PodSpec, volumes []v1.Volume, volumeMounts []v1.VolumeMount) error {
	for _, volume := range volumes {
		if err := mergeVolume(podSpec, volume); err != nil {
			return err
		}
	}
	for _, volumeMount := range volumeMounts {
		if err := mergeVolumeMount(podSpec, volumeMount, nil); err != nil {
			return err
		}
	}
	return nil
}

func renderMount(mount sednav1.Mount, idx int) (renderedMount, error) {
	volumeName := resolveVolumeName(mount, idx)
	volume, targetLeaf, readOnly, err := buildVolume(volumeName, mount)
	if err != nil {
		return renderedMount{}, err
	}

	targetPath, err := resolveTargetPath(mount, targetLeaf)
	if err != nil {
		return renderedMount{}, err
	}

	volumeMount := v1.VolumeMount{
		Name:      volumeName,
		MountPath: targetPath,
		ReadOnly:  readOnly || mount.Target.ReadOnly,
	}
	if mount.Target.SubPath != "" {
		volumeMount.SubPath = mount.Target.SubPath
	}
	if mount.Target.MountPropagation != nil {
		volumeMount.MountPropagation = mount.Target.MountPropagation
	}

	var envVar *v1.EnvVar
	if mount.EnvName != "" {
		envVar = &v1.EnvVar{
			Name:  mount.EnvName,
			Value: targetPath,
		}
	}

	return renderedMount{
		volume:        volume,
		volumeMount:   volumeMount,
		containerName: mount.Containers,
		envVar:        envVar,
	}, nil
}

func resolveVolumeName(mount sednav1.Mount, idx int) string {
	name := mount.Name
	if name == "" {
		name = fmt.Sprintf("mount-%d", idx)
	}

	name = runtime.ConvertK8SValidName(strings.ToLower(name))
	if name == "" {
		return fmt.Sprintf("mount-%d", idx)
	}
	return name
}

func buildVolume(volumeName string, mount sednav1.Mount) (v1.Volume, string, bool, error) {
	sourceType, err := resolveSourceType(mount.Source)
	if err != nil {
		return v1.Volume{}, "", false, err
	}

	switch sourceType {
	case sednav1.MountSourceTypeHostPath:
		hostPath := mount.Source.HostPath
		if hostPath == nil || strings.TrimSpace(hostPath.Path) == "" {
			return v1.Volume{}, "", false, fmt.Errorf("mount %q requires source.hostPath.path", mount.Name)
		}

		cleanPath := filepath.Clean(hostPath.Path)
		return v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: cleanPath,
					Type: hostPath.Type,
				},
			},
		}, filepath.Base(cleanPath), false, nil

	case sednav1.MountSourceTypePersistentVolumeClaim:
		pvc := mount.Source.PersistentVolumeClaim
		if pvc == nil || strings.TrimSpace(pvc.ClaimName) == "" {
			return v1.Volume{}, "", false, fmt.Errorf("mount %q requires source.persistentVolumeClaim.claimName", mount.Name)
		}
		return v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvc.ClaimName,
					ReadOnly:  pvc.ReadOnly,
				},
			},
		}, pvc.ClaimName, pvc.ReadOnly, nil

	case sednav1.MountSourceTypeConfigMap:
		configMap := mount.Source.ConfigMap
		if configMap == nil || strings.TrimSpace(configMap.Name) == "" {
			return v1.Volume{}, "", false, fmt.Errorf("mount %q requires source.configMap.name", mount.Name)
		}
		return v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: configMap.Name,
					},
					Optional: configMap.Optional,
				},
			},
		}, configMap.Name, true, nil

	case sednav1.MountSourceTypeSecret:
		secret := mount.Source.Secret
		if secret == nil || strings.TrimSpace(secret.SecretName) == "" {
			return v1.Volume{}, "", false, fmt.Errorf("mount %q requires source.secret.secretName", mount.Name)
		}
		return v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{
					SecretName: secret.SecretName,
					Optional:   secret.Optional,
				},
			},
		}, secret.SecretName, true, nil

	case sednav1.MountSourceTypeEmptyDir:
		emptyDir := mount.Source.EmptyDir
		volumeSource := &v1.EmptyDirVolumeSource{}
		targetLeaf := volumeName
		if mount.Name != "" {
			targetLeaf = mount.Name
		}
		if emptyDir != nil {
			volumeSource.Medium = emptyDir.Medium
			if emptyDir.SizeLimit != "" {
				quantity, err := resource.ParseQuantity(emptyDir.SizeLimit)
				if err != nil {
					return v1.Volume{}, "", false, fmt.Errorf("mount %q has invalid emptyDir.sizeLimit: %w", mount.Name, err)
				}
				volumeSource.SizeLimit = &quantity
			}
		}

		return v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				EmptyDir: volumeSource,
			},
		}, targetLeaf, false, nil
	default:
		return v1.Volume{}, "", false, fmt.Errorf("mount %q has unsupported source type %q", mount.Name, sourceType)
	}
}

func resolveSourceType(source sednav1.MountSource) (sednav1.MountSourceType, error) {
	if source.Type != "" {
		return source.Type, nil
	}

	candidates := make([]sednav1.MountSourceType, 0, 5)
	if source.HostPath != nil {
		candidates = append(candidates, sednav1.MountSourceTypeHostPath)
	}
	if source.PersistentVolumeClaim != nil {
		candidates = append(candidates, sednav1.MountSourceTypePersistentVolumeClaim)
	}
	if source.ConfigMap != nil {
		candidates = append(candidates, sednav1.MountSourceTypeConfigMap)
	}
	if source.Secret != nil {
		candidates = append(candidates, sednav1.MountSourceTypeSecret)
	}
	if source.EmptyDir != nil {
		candidates = append(candidates, sednav1.MountSourceTypeEmptyDir)
	}

	if len(candidates) == 1 {
		return candidates[0], nil
	}
	if len(candidates) == 0 {
		return "", fmt.Errorf("mount source type is empty")
	}
	return "", fmt.Errorf("mount source type is ambiguous")
}

func resolveTargetPath(mount sednav1.Mount, targetLeaf string) (string, error) {
	if mount.Target.Path != "" {
		return mount.Target.Path, nil
	}

	targetLeaf = filepath.Base(filepath.Clean(targetLeaf))
	if targetLeaf == "." || targetLeaf == string(filepath.Separator) || targetLeaf == "" {
		return "", fmt.Errorf("mount %q requires target.path when source path has no basename", mount.Name)
	}

	return filepath.Join(dataPathPrefix, targetLeaf), nil
}

func mergeVolume(podSpec *v1.PodSpec, volume v1.Volume) error {
	for _, existing := range podSpec.Volumes {
		if existing.Name != volume.Name {
			continue
		}
		if reflect.DeepEqual(existing, volume) {
			return nil
		}
		return fmt.Errorf("volume %q already exists with a different definition", volume.Name)
	}

	podSpec.Volumes = append(podSpec.Volumes, volume)
	return nil
}

func mergeVolumeMount(podSpec *v1.PodSpec, volumeMount v1.VolumeMount, containerNames []string) error {
	indexes, err := getContainerIndexes(podSpec.Containers, containerNames)
	if err != nil {
		return err
	}

	for _, idx := range indexes {
		container := &podSpec.Containers[idx]
		for _, existing := range container.VolumeMounts {
			if existing.MountPath != volumeMount.MountPath {
				continue
			}
			if reflect.DeepEqual(existing, volumeMount) {
				goto NEXT_CONTAINER
			}
			return fmt.Errorf("container %q already mounts %q", container.Name, volumeMount.MountPath)
		}

		container.VolumeMounts = append(container.VolumeMounts, volumeMount)
	NEXT_CONTAINER:
	}

	return nil
}

func mergeEnvVar(podSpec *v1.PodSpec, envVar v1.EnvVar, containerNames []string) error {
	indexes, err := getContainerIndexes(podSpec.Containers, containerNames)
	if err != nil {
		return err
	}

	for _, idx := range indexes {
		container := &podSpec.Containers[idx]
		for _, existing := range container.Env {
			if existing.Name != envVar.Name {
				continue
			}
			if reflect.DeepEqual(existing, envVar) {
				goto NEXT_CONTAINER
			}
			return fmt.Errorf("container %q already defines env %q", container.Name, envVar.Name)
		}

		container.Env = append(container.Env, envVar)
	NEXT_CONTAINER:
	}

	return nil
}

func getContainerIndexes(containers []v1.Container, containerNames []string) ([]int, error) {
	if len(containerNames) == 0 {
		indexes := make([]int, 0, len(containers))
		for idx := range containers {
			indexes = append(indexes, idx)
		}
		return indexes, nil
	}

	nameToIndex := make(map[string]int, len(containers))
	for idx, container := range containers {
		nameToIndex[container.Name] = idx
	}

	indexes := make([]int, 0, len(containerNames))
	for _, name := range containerNames {
		idx, ok := nameToIndex[name]
		if !ok {
			return nil, fmt.Errorf("container %q not found", name)
		}
		indexes = append(indexes, idx)
	}
	return indexes, nil
}
