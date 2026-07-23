package gmclient

import (
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestSendMessageStopsWithoutWaitingForQueueTraffic(t *testing.T) {
	client := &wsClient{
		WSConnection:       &WSConnection{WSConn: &websocket.Conn{}},
		SendMessageChannel: make(chan Message),
	}
	done := make(chan struct{})
	stop := make(chan struct{}, 1)
	close(done)
	go client.sendMessage(client.WSConnection.WSConn, done, stop)
	select {
	case <-stop:
	case <-time.After(time.Second):
		t.Fatal("connection-local sender leaked after reconnect teardown")
	}
}
