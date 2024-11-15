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

// IncrementalLearningJobLister helps list IncrementalLearningJobs.
// All objects returned here must be treated as read-only.
type IncrementalLearningJobLister interface {
	// List lists all IncrementalLearningJobs in the indexer.
	// Objects returned here must be treated as read-only.
	List(selector labels.Selector) (ret []*v1alpha1.IncrementalLearningJob, err error)
	// IncrementalLearningJobs returns an object that can list and get IncrementalLearningJobs.
	IncrementalLearningJobs(namespace string) IncrementalLearningJobNamespaceLister
	IncrementalLearningJobListerExpansion
}

// incrementalLearningJobLister implements the IncrementalLearningJobLister interface.
type incrementalLearningJobLister struct {
	indexer cache.Indexer
}

// NewIncrementalLearningJobLister returns a new IncrementalLearningJobLister.
func NewIncrementalLearningJobLister(indexer cache.Indexer) IncrementalLearningJobLister {
	return &incrementalLearningJobLister{indexer: indexer}
}

// List lists all IncrementalLearningJobs in the indexer.
func (s *incrementalLearningJobLister) List(selector labels.Selector) (ret []*v1alpha1.IncrementalLearningJob, err error) {
	err = cache.ListAll(s.indexer, selector, func(m interface{}) {
		ret = append(ret, m.(*v1alpha1.IncrementalLearningJob))
	})
	return ret, err
}

// IncrementalLearningJobs returns an object that can list and get IncrementalLearningJobs.
func (s *incrementalLearningJobLister) IncrementalLearningJobs(namespace string) IncrementalLearningJobNamespaceLister {
	return incrementalLearningJobNamespaceLister{indexer: s.indexer, namespace: namespace}
}

// IncrementalLearningJobNamespaceLister helps list and get IncrementalLearningJobs.
// All objects returned here must be treated as read-only.
type IncrementalLearningJobNamespaceLister interface {
	// List lists all IncrementalLearningJobs in the indexer for a given namespace.
	// Objects returned here must be treated as read-only.
	List(selector labels.Selector) (ret []*v1alpha1.IncrementalLearningJob, err error)
	// Get retrieves the IncrementalLearningJob from the indexer for a given namespace and name.
	// Objects returned here must be treated as read-only.
	Get(name string) (*v1alpha1.IncrementalLearningJob, error)
	IncrementalLearningJobNamespaceListerExpansion
}

// incrementalLearningJobNamespaceLister implements the IncrementalLearningJobNamespaceLister
// interface.
type incrementalLearningJobNamespaceLister struct {
	indexer   cache.Indexer
	namespace string
}

// List lists all IncrementalLearningJobs in the indexer for a given namespace.
func (s incrementalLearningJobNamespaceLister) List(selector labels.Selector) (ret []*v1alpha1.IncrementalLearningJob, err error) {
	err = cache.ListAllByNamespace(s.indexer, s.namespace, selector, func(m interface{}) {
		ret = append(ret, m.(*v1alpha1.IncrementalLearningJob))
	})
	return ret, err
}

// Get retrieves the IncrementalLearningJob from the indexer for a given namespace and name.
func (s incrementalLearningJobNamespaceLister) Get(name string) (*v1alpha1.IncrementalLearningJob, error) {
	obj, exists, err := s.indexer.GetByKey(s.namespace + "/" + name)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, errors.NewNotFound(v1alpha1.Resource("incrementallearningjob"), name)
	}
	return obj.(*v1alpha1.IncrementalLearningJob), nil
}
