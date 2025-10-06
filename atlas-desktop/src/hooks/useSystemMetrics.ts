import { useEffect, useState } from 'react';
import type { SystemMetrics } from '../types';

const DEFAULT_METRICS: SystemMetrics = {
  cpu: 12,
  memory: 42,
  network: 28,
  disk: 51
};

export const useSystemMetrics = () => {
  const [metrics, setMetrics] = useState<SystemMetrics>(DEFAULT_METRICS);

  useEffect(() => {
    let cancel: (() => void) | undefined;
    let interval: NodeJS.Timer | undefined;

    const subscribeToTauri = async () => {
      const tauriEvent = (window as unknown as { __TAURI__?: { event: { listen: Function } } }).__TAURI__?.event;
      if (!tauriEvent) {
        return false;
      }

      try {
        const unlisten = await tauriEvent.listen('system-metrics', (event: { payload: SystemMetrics }) => {
          setMetrics(event.payload);
        });
        cancel = () => {
          unlisten();
        };
        return true;
      } catch (error) {
        console.error('Failed to subscribe to Tauri system metrics', error);
        return false;
      }
    };

    const startFallback = () => {
      interval = setInterval(() => {
        setMetrics((prev) => ({
          cpu: Math.min(100, Math.max(0, prev.cpu + (Math.random() * 10 - 5))),
          memory: Math.min(100, Math.max(0, prev.memory + (Math.random() * 8 - 4))),
          network: Math.min(100, Math.max(0, prev.network + (Math.random() * 6 - 3))),
          disk: Math.min(100, Math.max(0, prev.disk + (Math.random() * 4 - 2)))
        }));
      }, 3000);
    };

    subscribeToTauri().then((success) => {
      if (!success) {
        startFallback();
      }
    });

    return () => {
      if (cancel) {
        cancel();
      }
      if (interval) {
        clearInterval(interval);
      }
    };
  }, []);

  return metrics;
};
