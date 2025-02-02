/*
Copyright The KubeEdge Authors.

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

// Code generated by lister-gen. DO NOT EDIT.

package v1alpha1

import (
	v1alpha1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/tools/cache"
)

// ReidJobLister helps list ReidJobs.
// All objects returned here must be treated as read-only.
type ReidJobLister interface {
	// List lists all ReidJobs in the indexer.
	// Objects returned here must be treated as read-only.
	List(selector labels.Selector) (ret []*v1alpha1.ReidJob, err error)
	// ReidJobs returns an object that can list and get ReidJobs.
	ReidJobs(namespace string) ReidJobNamespaceLister
	ReidJobListerExpansion
}

// reidJobLister implements the ReidJobLister interface.
type reidJobLister struct {
	indexer cache.Indexer
}

// NewReidJobLister returns a new ReidJobLister.
func NewReidJobLister(indexer cache.Indexer) ReidJobLister {
	return &reidJobLister{indexer: indexer}
}

// List lists all ReidJobs in the indexer.
func (s *reidJobLister) List(selector labels.Selector) (ret []*v1alpha1.ReidJob, err error) {
	err = cache.ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*v1alpha1.ReidJob))
	})
	return ret, err
}

// ReidJobs returns an object that can list and get ReidJobs.
func (s *reidJobLister) ReidJobs(namespace string) ReidJobNamespaceLister {
	return reidJobNamespaceLister{indexer: s.indexer, namespace: namespace}
}

// ReidJobNamespaceLister helps list and get ReidJobs.
// All objects returned here must be treated as read-only.
type ReidJobNamespaceLister interface {
	// List lists all ReidJobs in the indexer for a given namespace.
	// Objects returned here must be treated as read-only.
	List(selector labels.Selector) (ret []*v1alpha1.ReidJob, err error)
	// Get retrieves the ReidJob from the indexer for a given namespace and name.
	// Objects returned here must be treated as read-only.
	Get(name string) (*v1alpha1.ReidJob, error)
	ReidJobNamespaceListerExpansion
}

// reidJobNamespaceLister implements the ReidJobNamespaceLister
// interface.
type reidJobNamespaceLister struct {
	indexer   cache.Indexer
	namespace string
}

// List lists all ReidJobs in the indexer for a given namespace.
func (s reidJobNamespaceLister) List(selector labels.Selector) (ret []*v1alpha1.ReidJob, err error) {
	err = cache.ListAllByNamespace(s.indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(*v1alpha1.ReidJob))
	})
	return ret, err
}

// Get retrieves the ReidJob from the indexer for a given namespace and name.
func (s reidJobNamespaceLister) Get(name string) (*v1alpha1.ReidJob, error) {
	obj, exists, err := s.indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(v1alpha1.Resource("reidjob"), name)
	}
	return obj.(*v1alpha1.ReidJob), nil
}
