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

package ws

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/dayu-autostreamer/dayu-sedna/pkg/globalmanager/messagelayer/model"
)

func TestSameKeyReplacementSurvivesBlockedWrite(t *testing.T) {
	nodeName := fmt.Sprintf("runtime-message-order-test-%d", time.Now().UnixNano())
	readStop := make(chan struct{})
	closeCh := make(chan struct{}, 2)
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	writes := make(chan model.Message, 2)
	var mu sync.Mutex
	writeCount := 0

	AddNode(nodeName,
		func() (model.Message, error) {
			<-readStop
			return model.Message{}, errors.New("test reader stopped")
		},
		func(message model.Message) error {
			mu.Lock()
			writeCount++
			current := writeCount
			mu.Unlock()
			if current == 1 {
				close(firstStarted)
				<-releaseFirst
			}
			writes <- message
			return nil
		}, closeCh)

	header := model.MessageHeader{Namespace: "dayu", ResourceKind: "runtimeservice", ResourceName: "runtime-a"}
	first := &model.Message{MessageHeader: header, Content: []byte("first")}
	first.Operation = "insert"
	second := &model.Message{MessageHeader: header, Content: []byte("second")}
	second.Operation = "delete"
	if err := SendToEdge(nodeName, first); err != nil {
		t.Fatal(err)
	}
	select {
	case <-firstStarted:
	case <-time.After(time.Second):
		t.Fatal("first write did not start")
	}
	if err := SendToEdge(nodeName, second); err != nil {
		t.Fatal(err)
	}
	close(releaseFirst)

	for expected, content := range []string{"first", "second"} {
		select {
		case message := <-writes:
			if string(message.Content) != content {
				t.Fatalf("write %d: got %q, want %q", expected, message.Content, content)
			}
		case <-time.After(time.Second):
			t.Fatalf("timed out waiting for write %d", expected)
		}
	}

	getNodeQueue(nodeName).ShutDown()
	close(readStop)
	for i := 0; i < 2; i++ {
		select {
		case <-closeCh:
		case <-time.After(time.Second):
			t.Fatal("message-layer test goroutine did not stop")
		}
	}
}
