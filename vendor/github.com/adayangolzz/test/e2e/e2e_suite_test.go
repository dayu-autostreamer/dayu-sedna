/*
Copyright 2021 The KubeEdge Authors.
Copyright 2015 The Kubernetes Authors.

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

package e2e

import (
	"flag"
	"os"
	"testing"

	"github.com/adayangolzz/sedna-modified/test/e2e/framework"

	// test sources
	_ "github.com/adayangolzz/sedna-modified/test/e2e/apps"
)

func TestMain(m *testing.M) {
	framework.RegisterFlags(flag.CommandLine)
	flag.Parse()
	os.Exit(m.Run())
}

func TestE2E(t *testing.T) {
	RunE2ETests(t)
}
