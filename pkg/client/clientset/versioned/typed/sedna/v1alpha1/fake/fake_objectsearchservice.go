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

package fake

import (
	"context"

	// v1alpha1 "github.com/adayangolzz/sedna-modified/pkg/apis/sedna/v1alpha1"
	v1alpha1 "github.com/adayangolzz/sedna-modified/pkg/apis/sedna/v1alpha1"
	
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	labels "k8s.io/apimachinery/pkg/labels"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	testing "k8s.io/client-go/testing"
)

// FakeObjectSearchServices implements ObjectSearchServiceInterface
type FakeObjectSearchServices struct {
	Fake *FakeSednaV1alpha1
	ns   string
}

var objectsearchservicesResource = schema.GroupVersionResource{Group: "sedna.io", Version: "v1alpha1", Resource: "objectsearchservices"}

var objectsearchservicesKind = schema.GroupVersionKind{Group: "sedna.io", Version: "v1alpha1", Kind: "ObjectSearchService"}

// Get takes name of the objectSearchService, and returns the corresponding objectSearchService object, and an error if there is any.
func (c *FakeObjectSearchServices) Get(ctx context.Context, name string, options v1.GetOptions) (result *v1alpha1.ObjectSearchService, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewGetAction(objectsearchservicesResource, c.ns, name), &v1alpha1.ObjectSearchService{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ObjectSearchService), err
}

// List takes label and field selectors, and returns the list of ObjectSearchServices that match those selectors.
func (c *FakeObjectSearchServices) List(ctx context.Context, opts v1.ListOptions) (result *v1alpha1.ObjectSearchServiceList, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewListAction(objectsearchservicesResource, objectsearchservicesKind, c.ns, opts), &v1alpha1.ObjectSearchServiceList{})

	if obj == nil {
		return nil, err
	}

	label, _, _ := testing.ExtractFromListOptions(opts)
	if label == nil {
		label = labels.Everything()
	}
	list := &v1alpha1.ObjectSearchServiceList{ListMeta: obj.(*v1alpha1.ObjectSearchServiceList).ListMeta}
	for _, item := range obj.(*v1alpha1.ObjectSearchServiceList).Items {
		if label.Matches(labels.Set(item.Labels)) {
			list.Items = append(list.Items, item)
		}
	}
	return list, err
}

// Watch returns a watch.Interface that watches the requested objectSearchServices.
func (c *FakeObjectSearchServices) Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error) {
	return c.Fake.
		InvokesWatch(testing.NewWatchAction(objectsearchservicesResource, c.ns, opts))

}

// Create takes the representation of a objectSearchService and creates it.  Returns the server's representation of the objectSearchService, and an error, if there is any.
func (c *FakeObjectSearchServices) Create(ctx context.Context, objectSearchService *v1alpha1.ObjectSearchService, opts v1.CreateOptions) (result *v1alpha1.ObjectSearchService, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewCreateAction(objectsearchservicesResource, c.ns, objectSearchService), &v1alpha1.ObjectSearchService{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ObjectSearchService), err
}

// Update takes the representation of a objectSearchService and updates it. Returns the server's representation of the objectSearchService, and an error, if there is any.
func (c *FakeObjectSearchServices) Update(ctx context.Context, objectSearchService *v1alpha1.ObjectSearchService, opts v1.UpdateOptions) (result *v1alpha1.ObjectSearchService, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateAction(objectsearchservicesResource, c.ns, objectSearchService), &v1alpha1.ObjectSearchService{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ObjectSearchService), err
}

// UpdateStatus was generated because the type contains a Status member.
// Add a +genclient:noStatus comment above the type to avoid generating UpdateStatus().
func (c *FakeObjectSearchServices) UpdateStatus(ctx context.Context, objectSearchService *v1alpha1.ObjectSearchService, opts v1.UpdateOptions) (*v1alpha1.ObjectSearchService, error) {
	obj, err := c.Fake.
		Invokes(testing.NewUpdateSubresourceAction(objectsearchservicesResource, "status", c.ns, objectSearchService), &v1alpha1.ObjectSearchService{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ObjectSearchService), err
}

// Delete takes name of the objectSearchService and deletes it. Returns an error if one occurs.
func (c *FakeObjectSearchServices) Delete(ctx context.Context, name string, opts v1.DeleteOptions) error {
	_, err := c.Fake.
		Invokes(testing.NewDeleteAction(objectsearchservicesResource, c.ns, name), &v1alpha1.ObjectSearchService{})

	return err
}

// DeleteCollection deletes a collection of objects.
func (c *FakeObjectSearchServices) DeleteCollection(ctx context.Context, opts v1.DeleteOptions, listOpts v1.ListOptions) error {
	action := testing.NewDeleteCollectionAction(objectsearchservicesResource, c.ns, listOpts)

	_, err := c.Fake.Invokes(action, &v1alpha1.ObjectSearchServiceList{})
	return err
}

// Patch applies the patch and returns the patched objectSearchService.
func (c *FakeObjectSearchServices) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts v1.PatchOptions, subresources ...string) (result *v1alpha1.ObjectSearchService, err error) {
	obj, err := c.Fake.
		Invokes(testing.NewPatchSubresourceAction(objectsearchservicesResource, c.ns, name, pt, data, subresources...), &v1alpha1.ObjectSearchService{})

	if obj == nil {
		return nil, err
	}
	return obj.(*v1alpha1.ObjectSearchService), err
}