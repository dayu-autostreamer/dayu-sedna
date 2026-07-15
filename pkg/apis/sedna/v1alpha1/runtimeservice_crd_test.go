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
	"encoding/json"
	"io/ioutil"
	"path/filepath"
	"runtime"
	"testing"

	"sigs.k8s.io/yaml"
)

type openAPISchema struct {
	Type                  string                   `json:"type,omitempty"`
	Properties            map[string]openAPISchema `json:"properties,omitempty"`
	AdditionalProperties  *openAPISchema           `json:"additionalProperties,omitempty"`
	PreserveUnknownFields bool                     `json:"x-kubernetes-preserve-unknown-fields,omitempty"`
	Required              []string                 `json:"required,omitempty"`
}

type crdManifest struct {
	Spec struct {
		Versions []struct {
			Name   string `json:"name"`
			Schema struct {
				OpenAPIV3Schema openAPISchema `json:"openAPIV3Schema"`
			} `json:"schema"`
		} `json:"versions"`
	} `json:"spec"`
}

func repositoryRoot(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate RuntimeService CRD test source")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..", ".."))
}

func runtimeServiceSchema(t *testing.T, path string) openAPISchema {
	t.Helper()
	data, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatalf("read RuntimeService CRD %s: %v", path, err)
	}
	var manifest crdManifest
	if err := yaml.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("parse RuntimeService CRD %s: %v", path, err)
	}
	for _, version := range manifest.Spec.Versions {
		if version.Name == SchemeGroupVersion.Version {
			return version.Schema.OpenAPIV3Schema
		}
	}
	t.Fatalf("RuntimeService CRD %s does not serve %s", path, SchemeGroupVersion.Version)
	return openAPISchema{}
}

func requireStringMap(t *testing.T, parent openAPISchema, field, path string) {
	t.Helper()
	value, found := parent.Properties[field]
	if !found || value.Type != "object" || value.AdditionalProperties == nil ||
		value.AdditionalProperties.Type != "string" {
		t.Fatalf("%s.%s must be a typed string map, got %#v", path, field, value)
	}
}

func TestPublishedRuntimeServiceCRDsPreservePodTemplateMetadata(t *testing.T) {
	root := repositoryRoot(t)
	paths := map[string]string{
		"raw":  filepath.Join(root, "build", "crds", "sedna.io_runtimeservices.yaml"),
		"helm": filepath.Join(root, "build", "helm", "sedna", "crds", "sedna.io_runtimeservices.yaml"),
	}
	for name, path := range paths {
		t.Run(name, func(t *testing.T) {
			schema := runtimeServiceSchema(t, path)
			spec := schema.Properties["spec"]
			podTemplate := spec.Properties["podTemplate"]
			metadata := podTemplate.Properties["metadata"]
			if metadata.PreserveUnknownFields {
				t.Fatal("podTemplate.metadata must use a structural schema, not preserve arbitrary unknown fields")
			}
			requireStringMap(t, metadata, "labels", "spec.podTemplate.metadata")
			requireStringMap(t, metadata, "annotations", "spec.podTemplate.metadata")
			if len(metadata.Properties) != 2 {
				t.Fatalf("podTemplate.metadata must expose only labels and annotations, got %v", metadata.Properties)
			}
			if len(podTemplate.Required) != 1 || podTemplate.Required[0] != "spec" {
				t.Fatalf("podTemplate.spec must be required, got %v", podTemplate.Required)
			}
		})
	}
}

func TestRuntimePodTemplateJSONWireCompatibility(t *testing.T) {
	input := []byte(`{
        "metadata": {
            "labels": {"app.kubernetes.io/managed-by": "dayu-backend"},
            "annotations": {"dayu.io/note": "kept"}
        },
        "spec": {"containers": [{"name": "runtime", "image": "example/runtime:v1"}]}
    }`)
	var template RuntimePodTemplateSpec
	if err := json.Unmarshal(input, &template); err != nil {
		t.Fatalf("decode existing Dayu podTemplate wire shape: %v", err)
	}
	if template.Labels["app.kubernetes.io/managed-by"] != "dayu-backend" ||
		template.Annotations["dayu.io/note"] != "kept" ||
		len(template.Spec.Containers) != 1 {
		t.Fatalf("podTemplate wire fields were not decoded: %#v", template)
	}

	copy := template.DeepCopy()
	copy.Labels["app.kubernetes.io/managed-by"] = "changed"
	copy.Spec.Containers[0].Image = "example/runtime:v2"
	if template.Labels["app.kubernetes.io/managed-by"] != "dayu-backend" ||
		template.Spec.Containers[0].Image != "example/runtime:v1" {
		t.Fatal("RuntimePodTemplateSpec deepcopy aliases caller metadata or pod spec")
	}

	output, err := json.Marshal(&template)
	if err != nil {
		t.Fatalf("encode RuntimePodTemplateSpec: %v", err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(output, &wire); err != nil {
		t.Fatalf("decode RuntimePodTemplateSpec output: %v", err)
	}
	if _, found := wire["metadata"]; !found {
		t.Fatalf("encoded podTemplate lost metadata object: %s", output)
	}
	if _, found := wire["spec"]; !found || len(wire) != 2 {
		t.Fatalf("encoded podTemplate changed its public wire shape: %s", output)
	}
}
