package jointmultiedge

import (
	"testing"

	v1 "k8s.io/api/core/v1"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

func TestBuildLegacyFileMountsKeepsOriginalContract(t *testing.T) {
	envs, volumes, volumeMounts := buildLegacyFileMounts(sednav1.ConfFile{
		Path: "/tmp/models/model.bin",
	})

	if got, want := envs["VOLUME_NUM"], "1"; got != want {
		t.Fatalf("unexpected VOLUME_NUM: got %q want %q", got, want)
	}
	if got, want := envs["VOLUME_0"], "/tmp/models"; got != want {
		t.Fatalf("unexpected VOLUME_0: got %q want %q", got, want)
	}
	if len(volumes) != 1 {
		t.Fatalf("unexpected volume count: got %d want 1", len(volumes))
	}
	if got, want := volumes[0].HostPath.Path, "/tmp/models"; got != want {
		t.Fatalf("unexpected host path: got %q want %q", got, want)
	}
	if got, want := volumeMounts[0].MountPath, "/home/data/volume0"; got != want {
		t.Fatalf("unexpected mount path: got %q want %q", got, want)
	}
}

func TestInjectExplicitMountsDefaultsIntoDataPathPrefix(t *testing.T) {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{Name: "main"}},
	}

	err := injectExplicitMounts(&podSpec, []sednav1.Mount{
		{
			Name: "config-file",
			Source: sednav1.MountSource{
				HostPath: &sednav1.HostPathMountSource{
					Path: "/opt/app/config.yaml",
				},
			},
			EnvName: "APP_CONFIG",
		},
	})
	if err != nil {
		t.Fatalf("inject mounts failed: %v", err)
	}

	if len(podSpec.Volumes) != 1 {
		t.Fatalf("unexpected volume count: got %d want 1", len(podSpec.Volumes))
	}
	if got, want := podSpec.Containers[0].VolumeMounts[0].MountPath, "/home/data/config.yaml"; got != want {
		t.Fatalf("unexpected mount path: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].Env[0].Value, "/home/data/config.yaml"; got != want {
		t.Fatalf("unexpected env value: got %q want %q", got, want)
	}
}

func TestInjectExplicitMountsSupportsDeviceAndContainerSelection(t *testing.T) {
	charDevice := v1.HostPathCharDev
	podSpec := v1.PodSpec{
		Containers: []v1.Container{
			{Name: "detector"},
			{Name: "sidecar"},
		},
	}

	err := injectExplicitMounts(&podSpec, []sednav1.Mount{
		{
			Name: "camera0",
			Source: sednav1.MountSource{
				Type: sednav1.MountSourceTypeHostPath,
				HostPath: &sednav1.HostPathMountSource{
					Path: "/dev/video0",
					Type: &charDevice,
				},
			},
			Target: sednav1.MountTarget{
				Path: "/dev/video0",
			},
			Containers: []string{"detector"},
			EnvName:    "CAMERA_DEVICE",
		},
	})
	if err != nil {
		t.Fatalf("inject mounts failed: %v", err)
	}

	if got, want := len(podSpec.Containers[0].VolumeMounts), 1; got != want {
		t.Fatalf("unexpected detector mount count: got %d want %d", got, want)
	}
	if got, want := len(podSpec.Containers[1].VolumeMounts), 0; got != want {
		t.Fatalf("unexpected sidecar mount count: got %d want %d", got, want)
	}
	if got, want := podSpec.Containers[0].VolumeMounts[0].MountPath, "/dev/video0"; got != want {
		t.Fatalf("unexpected detector mount path: got %q want %q", got, want)
	}
	if got, want := *podSpec.Volumes[0].HostPath.Type, v1.HostPathCharDev; got != want {
		t.Fatalf("unexpected hostPath type: got %q want %q", got, want)
	}
}
