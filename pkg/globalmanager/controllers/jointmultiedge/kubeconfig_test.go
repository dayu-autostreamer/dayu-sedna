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
	"testing"

	v1 "k8s.io/api/core/v1"

	sednav1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
)

func TestInjectKubeConfigDefaultsToRootKubeDir(t *testing.T) {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{Name: "main"}},
	}

	if err := injectKubeConfig(&podSpec, nil); err != nil {
		t.Fatalf("inject kubeconfig failed: %v", err)
	}

	if got, want := len(podSpec.Volumes), 1; got != want {
		t.Fatalf("unexpected volume count: got %d want %d", got, want)
	}
	if got, want := podSpec.Volumes[0].HostPath.Path, "/root/.kube"; got != want {
		t.Fatalf("unexpected host path: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].VolumeMounts[0].MountPath, "/root/.kube"; got != want {
		t.Fatalf("unexpected mount path: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].Env[0].Name, "KUBECONFIG"; got != want {
		t.Fatalf("unexpected env name: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].Env[0].Value, "/root/.kube/config"; got != want {
		t.Fatalf("unexpected env value: got %q want %q", got, want)
	}
}

func TestInjectKubeConfigSupportsSecretSource(t *testing.T) {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{Name: "main"}},
	}

	err := injectKubeConfig(&podSpec, &sednav1.KubeConfigSpec{
		SecretName: "edge-kubeconfig",
		SecretKey:  "admin.conf",
		MountPath:  "/var/run/dayu/kube",
	})
	if err != nil {
		t.Fatalf("inject kubeconfig failed: %v", err)
	}

	if podSpec.Volumes[0].Secret == nil {
		t.Fatalf("expected secret volume source")
	}
	if got, want := podSpec.Volumes[0].Secret.SecretName, "edge-kubeconfig"; got != want {
		t.Fatalf("unexpected secret name: got %q want %q", got, want)
	}
	if got, want := podSpec.Volumes[0].Secret.Items[0].Key, "admin.conf"; got != want {
		t.Fatalf("unexpected secret key: got %q want %q", got, want)
	}
	if got, want := podSpec.Volumes[0].Secret.Items[0].Path, "config"; got != want {
		t.Fatalf("unexpected secret mount file: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].VolumeMounts[0].MountPath, "/var/run/dayu/kube"; got != want {
		t.Fatalf("unexpected mount path: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].Env[0].Value, "/var/run/dayu/kube/config"; got != want {
		t.Fatalf("unexpected env value: got %q want %q", got, want)
	}
}

func TestInjectKubeConfigRejectsTildeHostPath(t *testing.T) {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{Name: "main"}},
	}

	err := injectKubeConfig(&podSpec, &sednav1.KubeConfigSpec{
		HostPath: "~/.kube",
	})
	if err == nil {
		t.Fatalf("expected hostPath validation error")
	}
	if !strings.Contains(err.Error(), "does not support") {
		t.Fatalf("unexpected error: %v", err)
	}
}
