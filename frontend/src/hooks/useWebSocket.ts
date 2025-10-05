import { useEffect, useRef, useState, useCallback } from 'react';

const RECONNECT_DELAY_MS = 750;

export const useWebSocket = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shouldReconnect = useRef(true);

  useEffect(() => {
    shouldReconnect.current = true;

    const connect = () => {
      if (!shouldReconnect.current) return;

      try {
        const ws = new WebSocket(url);
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
           // Create a new object with timestamp to ensure state update triggers
           const newMessage = { ...data, _timestamp: Date.now() };
           console.info('[ws] Setting lastMessage to:', newMessage);
           setLastMessage(newMessage);
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
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  return { isConnected, lastMessage, sendMessage };
};
