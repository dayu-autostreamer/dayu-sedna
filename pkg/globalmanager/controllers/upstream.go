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
	"fmt"
	"strings"

	"k8s.io/klog/v2"

	"github.com/dayu-autostreamer/dayu-sedna/pkg/globalmanager/messagelayer"
	"github.com/dayu-autostreamer/dayu-sedna/pkg/globalmanager/runtime"
)

// UpstreamController subscribes the updates from edge and syncs to k8s api server
type UpstreamController struct {
	messageLayer        messagelayer.MessageLayer
	updateHandlers      map[string]runtime.UpstreamHandler
	sourceAwareHandlers map[string]runtime.SourceAwareUpstreamHandler
}

func (uc *UpstreamController) checkOperation(operation string) error {
	// current only support the 'status' operation
	if operation != "status" {
		return fmt.Errorf("unknown operation '%s'", operation)
	}
	return nil
}

// syncEdgeUpdate receives the updates from edge and syncs these to k8s.
func (uc *UpstreamController) syncEdgeUpdate() {
	for {
		select {
		case <-uc.messageLayer.Done():
			klog.Info("Stop sedna upstream loop")
			return
		default:
		}

		update, err := uc.messageLayer.ReceiveResourceUpdate()
		if err == nil {
			err = uc.checkOperation(update.Operation)
		}
		if err != nil {
			klog.Warningf("Ignore update since this err: %+v", err)
			continue
		}

		kind := update.Kind
		namespace := update.Namespace
		name := update.Name
		operation := update.Operation

		if handler, ok := uc.sourceAwareHandlers[kind]; ok {
			err := handler(update.SourceNode, name, namespace, operation, update.Content)
			if err != nil {
				klog.Errorf("Error to handle %s %s/%s operation(%s) from node %s: %+v", kind, namespace, name, operation, update.SourceNode, err)
			}
		} else if handler, ok := uc.updateHandlers[kind]; ok {
			err := handler(name, namespace, operation, update.Content)
			if err != nil {
				klog.Errorf("Error to handle %s %s/%s operation(%s): %+v", kind, namespace, name, operation, err)
			}
		} else {
			klog.Warningf("No handler for resource kind %s", kind)
		}
	}
}

// Run starts the upstream controller
func (uc *UpstreamController) Run(stopCh <-chan struct{}) {
	klog.Info("Start the sedna upstream controller")

	uc.syncEdgeUpdate()
	<-stopCh
}

func (uc *UpstreamController) Add(kind string, handler runtime.UpstreamHandler) error {
	kind = strings.ToLower(kind)
	if _, ok := uc.updateHandlers[kind]; ok || uc.sourceAwareHandlers[kind] != nil {
		return fmt.Errorf("a upstream handler for kind %s already exists", kind)
	}
	uc.updateHandlers[kind] = handler

	return nil
}

// AddSourceAware registers a handler which receives the node identity declared
// when the LC websocket connection was established, alongside the resource
// header and payload.
func (uc *UpstreamController) AddSourceAware(kind string, handler runtime.SourceAwareUpstreamHandler) error {
	kind = strings.ToLower(kind)
	if _, ok := uc.sourceAwareHandlers[kind]; ok || uc.updateHandlers[kind] != nil {
		return fmt.Errorf("an upstream handler for kind %s already exists", kind)
	}
	uc.sourceAwareHandlers[kind] = handler
	return nil
}

// NewUpstreamController creates a new Upstream controller from config
func NewUpstreamController(cc *runtime.ControllerContext) (*UpstreamController, error) {
	uc := &UpstreamController{
		messageLayer:        messagelayer.NewContextMessageLayer(),
		updateHandlers:      make(map[string]runtime.UpstreamHandler),
		sourceAwareHandlers: make(map[string]runtime.SourceAwareUpstreamHandler),
	}

	return uc, nil
}
