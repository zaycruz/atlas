import { useEffect, useRef, useState, useCallback } from 'react';

const RECONNECT_DELAY_MS = 750;

export const useWebSocket = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  const [messageQueue, setMessageQueue] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shouldReconnect = useRef(true);
  const messageCounter = useRef(0);
  const connectionId = useRef(0);

  useEffect(() => {
    shouldReconnect.current = true;

    const connect = () => {
      if (!shouldReconnect.current) return;

      try {
        const ws = new WebSocket(url);
        const currentConnectionId = ++connectionId.current;
        messageCounter.current = 0;
        wsRef.current = ws;

        ws.onopen = () => {
          console.info('[ws] connection opened', url);
          setIsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.info('[ws] RAW message received:', event.data);
            console.info('[ws] PARSED message:', data);
            console.info('[ws] message type:', data.type);
            const newMessage = {
              ...data,
              _timestamp: Date.now(),
              _sequence: ++messageCounter.current,
              _connectionId: currentConnectionId
            };
            console.info('[ws] Setting lastMessage to:', newMessage);
            setMessageQueue((prev) => [...prev, newMessage]);
          } catch (error) {
            console.error('Failed to parse WebSocket message', error);
          }
        };

        ws.onerror = (error) => {
          console.error('[ws] error', error);
        };

        ws.onclose = () => {
          console.warn('[ws] connection closed', url);
          setIsConnected(false);
          if (shouldReconnect.current) {
            reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
          }
        };
      } catch (error) {
        console.error('Failed to create WebSocket', error);
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    connect();

    return () => {
      shouldReconnect.current = false;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]);

  const sendMessage = useCallback((message: unknown) => {
    console.log('[useWebSocket] sendMessage called');
    console.log('[useWebSocket] wsRef.current:', wsRef.current);
    console.log('[useWebSocket] readyState:', wsRef.current?.readyState);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const jsonMessage = JSON.stringify(message);
      console.log('[useWebSocket] Sending:', jsonMessage);
      wsRef.current.send(jsonMessage);
      console.log('[useWebSocket] Message sent');
    } else {
      console.error('[useWebSocket] Cannot send - not connected. State:', wsRef.current?.readyState);
    }
  }, []);

  const clearQueue = useCallback((connectionId: number, sequence: number) => {
    setMessageQueue((prev) =>
      prev.filter((message) => {
        if (typeof message._connectionId !== 'number' || message._connectionId !== connectionId) {
          return true;
        }
        if (typeof message._sequence !== 'number') {
          return true;
        }
        return message._sequence > sequence;
      })
    );
  }, []);

  return { isConnected, messages: messageQueue, clearMessages: clearQueue, sendMessage };
};
