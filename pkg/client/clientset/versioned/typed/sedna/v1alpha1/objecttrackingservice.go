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

// Code generated by client-gen. DO NOT EDIT.

package v1alpha1

import (
	"context"
	"time"

	v1alpha1 "github.com/dayu-autostreamer/dayu-sedna/pkg/apis/sedna/v1alpha1"
	scheme "github.com/dayu-autostreamer/dayu-sedna/pkg/client/clientset/versioned/scheme"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// ObjectTrackingServicesGetter has a method to return a ObjectTrackingServiceInterface.
// A group's client should implement this interface.
type ObjectTrackingServicesGetter interface {
	ObjectTrackingServices(namespace string) ObjectTrackingServiceInterface
}

// ObjectTrackingServiceInterface has methods to work with ObjectTrackingService resources.
type ObjectTrackingServiceInterface interface {
	Create(ctx context.Context, objectTrackingService *v1alpha1.ObjectTrackingService, opts v1.CreateOptions) (*v1alpha1.ObjectTrackingService, error)
	Update(ctx context.Context, objectTrackingService *v1alpha1.ObjectTrackingService, opts v1.UpdateOptions) (*v1alpha1.ObjectTrackingService, error)
	UpdateStatus(ctx context.Context, objectTrackingService *v1alpha1.ObjectTrackingService, opts v1.UpdateOptions) (*v1alpha1.ObjectTrackingService, error)
	Delete(ctx context.Context, name string, opts v1.DeleteOptions) error
	DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error
	Get(ctx context.Context, name string, opts v1.GetOptions) (*v1alpha1.ObjectTrackingService, error)
	List(ctx context.Context, opts v1.ListOptions) (*v1alpha1.ObjectTrackingServiceList, error)
	Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error)
	Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v1alpha1.ObjectTrackingService, err error)
	ObjectTrackingServiceExpansion
}

// objectTrackingServices implements ObjectTrackingServiceInterface
type objectTrackingServices struct {
	client rest.Interface
	ns     string
}

// newObjectTrackingServices returns a ObjectTrackingServices
func newObjectTrackingServices(c *SednaV1alpha1Client, namespace string) *objectTrackingServices {
	return &objectTrackingServices{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the objectTrackingService, and returns the corresponding objectTrackingService object, and an error if there is any.
func (c *objectTrackingServices) Get(ctx context.Context, name string, options v1.GetOptions) (result *v1alpha1.ObjectTrackingService, err error) {
	result = &v1alpha1.ObjectTrackingService{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do(ctx).
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of ObjectTrackingServices that match those selectors.
func (c *objectTrackingServices) List(ctx context.Context, opts v1.ListOptions) (result *v1alpha1.ObjectTrackingServiceList, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &v1alpha1.ObjectTrackingServiceList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Do(ctx).
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested objectTrackingServices.
func (c *objectTrackingServices) Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Watch(ctx)
}

// Create takes the representation of a objectTrackingService and creates it.  Returns the server's representation of the objectTrackingService, and an error, if there is any.
func (c *objectTrackingServices) Create(ctx context.Context, objectTrackingService *v1alpha1.ObjectTrackingService, opts v1.CreateOptions) (result *v1alpha1.ObjectTrackingService, err error) {
	result = &v1alpha1.ObjectTrackingService{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(objectTrackingService).
		Do(ctx).
		Into(result)
	return
}

// Update takes the representation of a objectTrackingService and updates it. Returns the server's representation of the objectTrackingService, and an error, if there is any.
func (c *objectTrackingServices) Update(ctx context.Context, objectTrackingService *v1alpha1.ObjectTrackingService, opts v1.UpdateOptions) (result *v1alpha1.ObjectTrackingService, err error) {
	result = &v1alpha1.ObjectTrackingService{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		Name(objectTrackingService.Name).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(objectTrackingService).
		Do(ctx).
		Into(result)
	return
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *objectTrackingServices) UpdateStatus(ctx context.Context, objectTrackingService *v1alpha1.ObjectTrackingService, opts v1.UpdateOptions) (result *v1alpha1.ObjectTrackingService, err error) {
	result = &v1alpha1.ObjectTrackingService{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		Name(objectTrackingService.Name).
		SubResource("status").
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(objectTrackingService).
		Do(ctx).
		Into(result)
	return
}

// Delete takes name of the objectTrackingService and deletes it. Returns an error if one occurs.
func (c *objectTrackingServices) Delete(ctx context.Context, name string, opts v1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		Name(name).
		Body(&opts).
		Do(ctx).
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *objectTrackingServices) DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error {
	var timeout time.Duration
	if listOpts.TimeoutSeconds != nil {
		timeout = time.Duration(*listOpts.TimeoutSeconds) * time.Second
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("objecttrackingservices").
		VersionedParams(&listOpts, scheme.ParameterCodec).
		Timeout(timeout).
		Body(&opts).
		Do(ctx).
		Error()
}

// Patch applies the patch and returns the patched objectTrackingService.
func (c *objectTrackingServices) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v1alpha1.ObjectTrackingService, err error) {
	result = &v1alpha1.ObjectTrackingService{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("objecttrackingservices").
		Name(name).
		SubResource(subresources...).
		VersionedParams(&opts, scheme.ParameterCodec).
		Body(data).
		Do(ctx).
		Into(result)
	return
}
