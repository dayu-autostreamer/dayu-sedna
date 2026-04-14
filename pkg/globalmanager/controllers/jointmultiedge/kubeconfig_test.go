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
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestInjectWorkerKubeConfigMatchesUpstreamMountContract(t *testing.T) {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{Name: "main"}},
	}

	if err := injectWorkerKubeConfig(&podSpec, "/home/nvidia/.kube"); err != nil {
		t.Fatalf("inject worker kubeconfig failed: %v", err)
	}

	if got, want := len(podSpec.Volumes), 1; got != want {
		t.Fatalf("unexpected volume count: got %d want %d", got, want)
	}
	if got, want := podSpec.Volumes[0].HostPath.Path, "/home/nvidia/.kube"; got != want {
		t.Fatalf("unexpected host path: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].VolumeMounts[0].MountPath, "/home/data/.kube"; got != want {
		t.Fatalf("unexpected mount path: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].Env[0].Name, "KUBECONFIG"; got != want {
		t.Fatalf("unexpected env name: got %q want %q", got, want)
	}
	if got, want := podSpec.Containers[0].Env[0].Value, "/home/data/.kube/config"; got != want {
		t.Fatalf("unexpected env value: got %q want %q", got, want)
	}
}

func TestInjectWorkerKubeConfigSkipsEmptyPath(t *testing.T) {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{Name: "main"}},
	}

	if err := injectWorkerKubeConfig(&podSpec, ""); err != nil {
		t.Fatalf("inject worker kubeconfig failed: %v", err)
	}

	if got, want := len(podSpec.Volumes), 0; got != want {
		t.Fatalf("unexpected volume count: got %d want %d", got, want)
	}
	if got, want := len(podSpec.Containers[0].VolumeMounts), 0; got != want {
		t.Fatalf("unexpected mount count: got %d want %d", got, want)
	}
	if got, want := len(podSpec.Containers[0].Env), 0; got != want {
		t.Fatalf("unexpected env count: got %d want %d", got, want)
	}
}
