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

// Code generated by informer-gen. DO NOT EDIT.

package v1alpha1

import (
	"context"
	time "time"

	// sednav1alpha1 "github.com/adayangolzz/sedna-modified/pkg/apis/sedna/v1alpha1"
	// versioned "github.com/adayangolzz/sedna-modified/pkg/client/clientset/versioned"
	// internalinterfaces "github.com/adayangolzz/sedna-modified/pkg/client/informers/externalversions/internalinterfaces"
	// v1alpha1 "github.com/adayangolzz/sedna-modified/pkg/client/listers/sedna/v1alpha1"
	sednav1alpha1 "github.com/adayangolzz/sedna-modified/pkg/apis/sedna/v1alpha1"
	versioned "github.com/adayangolzz/sedna-modified/pkg/client/clientset/versioned"
	internalinterfaces "github.com/adayangolzz/sedna-modified/pkg/client/informers/externalversions/internalinterfaces"
	v1alpha1 "github.com/adayangolzz/sedna-modified/pkg/client/listers/sedna/v1alpha1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
	watch "k8s.io/apimachinery/pkg/watch"
	cache "k8s.io/client-go/tools/cache"
)

// DatasetInformer provides access to a shared informer and lister for
// Datasets.
type DatasetInformer interface {
	Informer() cache.SharedIndexInformer
	Lister() v1alpha1.DatasetLister
}

type datasetInformer struct {
	factory          internalinterfaces.SharedInformerFactory
	tweakListOptions internalinterfaces.TweakListOptionsFunc
	namespace        string
}

// NewDatasetInformer constructs a new informer for Dataset type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewDatasetInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers) cache.SharedIndexInformer {
	return NewFilteredDatasetInformer(client, namespace, resyncPeriod, indexers, nil)
}

// NewFilteredDatasetInformer constructs a new informer for Dataset type.
// Always prefer using an informer factory to get a shared informer instead of getting an independent
// one. This reduces memory footprint and number of connections to the server.
func NewFilteredDatasetInformer(client versioned.Interface, namespace string, resyncPeriod time.Duration, indexers cache.Indexers, tweakListOptions internalinterfaces.TweakListOptionsFunc) cache.SharedIndexInformer {
	return cache.NewSharedIndexInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.SednaV1alpha1().Datasets(namespace).List(context.TODO(), options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				if tweakListOptions != nil {
					tweakListOptions(&options)
				}
				return client.SednaV1alpha1().Datasets(namespace).Watch(context.TODO(), options)
			},
		},
		&sednav1alpha1.Dataset{},
		resyncPeriod,
		indexers,
	)
}

func (f *datasetInformer) defaultInformer(client versioned.Interface, resyncPeriod time.Duration) cache.SharedIndexInformer {
	return NewFilteredDatasetInformer(client, f.namespace, resyncPeriod, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, f.tweakListOptions)
}

func (f *datasetInformer) Informer() cache.SharedIndexInformer {
	return f.factory.InformerFor(&sednav1alpha1.Dataset{}, f.defaultInformer)
}

func (f *datasetInformer) Lister() v1alpha1.DatasetLister {
	return v1alpha1.NewDatasetLister(f.Informer().GetIndexer())
}