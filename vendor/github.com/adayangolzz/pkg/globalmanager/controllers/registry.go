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

package controllers

import (
	"github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/dataset"
	fe "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/featureextraction"
	fl "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/federatedlearning"
	il "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/incrementallearning"
	ji "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/jointinference"
	// DONE: mulji路径
	// mulji 
	mulji "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/jointmultiedge"
	ll "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/lifelonglearning"
	objs "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/objectsearch"
	reid "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/reid"
	va "github.com/adayangolzz/sedna-modified/pkg/globalmanager/controllers/videoanalytics"
	"github.com/adayangolzz/sedna-modified/pkg/globalmanager/runtime"
)

type FeatureFactory = func(*runtime.ControllerContext) (runtime.FeatureControllerI, error)

type Registry map[string]FeatureFactory

func NewRegistry() Registry {
	return Registry{
		ji.Name:      ji.New,
		// DONE:
		mulji.Name:	  mulji.New,
		fe.Name:      fe.New,
		fl.Name:      fl.New,
		il.Name:      il.New,
		ll.Name:      ll.New,
		reid.Name:    reid.New,
		va.Name:      va.New,
		dataset.Name: dataset.New,
		objs.Name:    objs.New,
	}
}
