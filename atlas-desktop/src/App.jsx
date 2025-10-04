import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  Clock,
  Globe,
  Network,
  Terminal,
} from 'lucide-react';
import { checkHealth, fetchMetrics, resetSession, sendMessage } from './lib/api.js';

const INITIAL_HISTORY = [
  { type: 'system', text: 'ATLAS DESKTOP LINKED. Core bridge initialised.' },
  { type: 'info', text: 'Voice recognition: OFFLINE (desktop bridge handles text only).' },
  { type: 'info', text: 'Layered memory: ACTIVE. Live statistics available in Analytics.' },
];

const MODULES = [
  { id: 'terminal', icon: Terminal, label: 'Terminal' },
  { id: 'analytics', icon: Activity, label: 'Analytics' },
  { id: 'network', icon: Network, label: 'Memory Graph' },
  { id: 'global', icon: Globe, label: 'Status' },
];

function classNames(...values) {
  return values.filter(Boolean).join(' ');
}

const formatText = (text) => text || '(no response)';

const AtlasInterface = () => {
  const [time, setTime] = useState(new Date());
  const [activeModule, setActiveModule] = useState('terminal');
  const [terminalHistory, setTerminalHistory] = useState(INITIAL_HISTORY);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [metrics, setMetrics] = useState({ cpu: 32, memory: 48, network: 72, tasks: 9 });
  const [quickStats, setQuickStats] = useState({
    sessions_today: 1,
    commands_executed: 0,
    data_processed_gb: 0,
    avg_response_time: 0,
  });
  const [timeline, setTimeline] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [connectionIssue, setConnectionIssue] = useState(null);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const data = await fetchMetrics();
        if (!mounted) return;
        setMetrics(data.metrics);
        setQuickStats(data.quick_stats);
        setTimeline(data.timeline || []);
        setNotifications(data.notifications || []);
        setConnectionIssue(null);
      } catch (err) {
        if (!mounted) return;
        setConnectionIssue(err.message || 'Unable to reach desktop bridge');
      }
    };
    poll();
    const interval = setInterval(poll, 6000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const heartbeat = async () => {
      try {
        await checkHealth();
        if (mounted) {
          setConnectionIssue(null);
        }
      } catch (err) {
        if (mounted) {
          setConnectionIssue(err.message || 'Desktop bridge offline');
        }
      }
    };
    heartbeat();
    const interval = setInterval(heartbeat, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleLocalCommand = async (command) => {
    const parts = command.slice(1).trim().split(/\s+/);
    const primary = parts[0]?.toLowerCase();

    switch (primary) {
      case 'clear':
        setTerminalHistory([{ type: 'system', text: 'Terminal cleared.' }]);
        return true;
      case 'help':
        setTerminalHistory((history) => [
          ...history,
          {
            type: 'info',
            text: 'Commands: /help, /clear, /metrics, /reset. Plain text is sent to Atlas core.',
          },
        ]);
        return true;
      case 'metrics': {
        try {
          const data = await fetchMetrics();
          setTerminalHistory((history) => [
            ...history,
            {
              type: 'data',
              text: `CPU ${data.metrics.cpu}% | RAM ${data.metrics.memory}% | NET ${data.metrics.network}% | Active ${data.metrics.tasks}`,
            },
          ]);
          setMetrics(data.metrics);
          setQuickStats(data.quick_stats);
          setTimeline(data.timeline || []);
          setNotifications(data.notifications || []);
        } catch (err) {
          setTerminalHistory((history) => [
            ...history,
            { type: 'error', text: `Metrics unavailable: ${err.message}` },
          ]);
        }
        return true;
      }
      case 'reset': {
        try {
          await resetSession();
          setTerminalHistory((history) => [
            ...history,
            { type: 'system', text: 'Desktop session reset. Memories refreshed.' },
          ]);
        } catch (err) {
          setTerminalHistory((history) => [
            ...history,
            { type: 'error', text: `Reset failed: ${err.message}` },
          ]);
        }
        return true;
      }
      default:
        setTerminalHistory((history) => [
          ...history,
          { type: 'warn', text: `Unknown command: ${command}` },
        ]);
        return true;
    }
  };

  const handleSubmit = async () => {
    const trimmed = input.trim();
    if (!trimmed || isSending) return;

    setTerminalHistory((history) => [
      ...history,
      { type: 'command', text: `> ${trimmed}` },
    ]);
    setInput('');

    if (trimmed.startsWith('/')) {
      await handleLocalCommand(trimmed);
      return;
    }

    setIsSending(true);
    try {
      const result = await sendMessage(trimmed);
      setTerminalHistory((history) => {
        const entries = [...history];
        entries.push({ type: 'assistant', text: formatText(result.response) });
        if (result.objective) {
          entries.push({ type: 'meta', text: `Objective: ${result.objective}` });
        }
        if (result.tags?.length) {
          entries.push({ type: 'meta', text: `Tags: ${result.tags.join(', ')}` });
        }
        if (result.error) {
          entries.push({ type: 'error', text: `Warning from core: ${result.error}` });
        }
        return entries;
      });
      setConnectionIssue(null);
    } catch (err) {
      setTerminalHistory((history) => [
        ...history,
        { type: 'error', text: `Bridge error: ${err.message}` },
      ]);
      setConnectionIssue(err.message || 'Desktop bridge offline');
    } finally {
      setIsSending(false);
    }
  };

  let moduleContent;

  if (activeModule === 'analytics') {
    moduleContent = (
      <div className="panel">
        <header className="panel-header">
          <Activity size={16} />
          <span>System Analytics</span>
        </header>
          <div className="analytics-grid">
            <div className="analytics-card">
              <span className="label">Performance Index</span>
              <span className="value">{Math.min(100, Math.round((metrics.cpu + metrics.memory) / 2))}%</span>
              <span className="caption">Adaptive average from recent turns</span>
            </div>
            <div className="analytics-card">
              <span className="label">Active Processes</span>
              <span className="value">{metrics.tasks}</span>
              <span className="caption">Atlas tools engaged this session</span>
            </div>
            <div className="analytics-card">
              <span className="label">Network Load</span>
              <span className="value">{metrics.network}%</span>
              <span className="caption">Ollama request utilisation</span>
            </div>
            <div className="analytics-card">
              <span className="label">Avg Response Time</span>
              <span className="value">{quickStats.avg_response_time}s</span>
              <span className="caption">Rolling mean across recent turns</span>
            </div>
          </div>
          <div className="timeline">
            <span className="label">Activity Timeline</span>
            <ul>
              {(timeline.length ? timeline : [{ time: '--:--:--', event: 'Awaiting activity', status: 'info' }]).map((item, index) => (
                <li key={`${item.time}-${index}`} className={classNames('timeline-item', item.status)}>
                  <span className="time">{item.time}</span>
                  <span className="event">{item.event}</span>
                </li>
              ))}
            </ul>
        </div>
      </div>
    );
  } else if (activeModule === 'network') {
    moduleContent = (
      <div className="panel">
        <header className="panel-header">
          <Network size={16} />
          <span>Memory Topology</span>
        </header>
          <div className="network-graph">
            <svg viewBox="0 0 200 200">
              <circle cx="100" cy="100" r="60" className="ring" />
              <circle cx="100" cy="100" r="85" className="ring dashed" />
              <circle cx="100" cy="100" r="110" className="ring dotted" />
              <circle cx="100" cy="100" r="12" className="core" />
              {[0, 60, 120, 180, 240, 300].map((angle) => {
                const radians = (angle * Math.PI) / 180;
                const x = 100 + Math.cos(radians) * 80;
                const y = 100 + Math.sin(radians) * 80;
                return (
                  <g key={angle}>
                    <line x1="100" y1="100" x2={x} y2={y} className="edge" />
                    <circle cx={x} cy={y} r="8" className="node" />
                  </g>
                );
              })}
            </svg>
            <div className="legend">
              <div><span className="dot core" />Core System</div>
              <div><span className="dot node" />Memory Nodes</div>
            </div>
        </div>
      </div>
    );
  } else if (activeModule === 'global') {
    moduleContent = (
      <div className="panel">
        <header className="panel-header">
          <Globe size={16} />
          <span>Bridge Status</span>
        </header>
          <div className="status-block">
          <p>Desktop bridge {connectionIssue ? 'offline' : 'synchronised'}.</p>
          {connectionIssue ? (
            <p className="warn">{connectionIssue}</p>
          ) : (
            <p>Tracking {metrics.tasks + 4} live subsystems.</p>
          )}
        </div>
      </div>
    );
  } else {
    moduleContent = (
      <div className="panel terminal-panel">
        <header className="panel-header">
          <Terminal size={16} />
          <span>Command Terminal</span>
          {connectionIssue && <span className="warn chip">Bridge offline</span>}
        </header>
        <div className="terminal-scroll">
          {terminalHistory.map((entry, index) => (
            <div key={`${entry.type}-${index}`} className={classNames('terminal-line', entry.type)}>
              {entry.text}
            </div>
          ))}
          {isSending && <div className="terminal-line info">(processing...)</div>}
        </div>
        <div className="terminal-input">
          <span className="prompt">$</span>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Type a command or prompt"
            disabled={isSending}
          />
          <button type="button" onClick={handleSubmit} disabled={isSending}>
            Send
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div>
          <h1>A.T.L.A.S.</h1>
          <p>Advanced Tactical Logistics and Analysis System</p>
        </div>
        <div className="clock">
          <span className="time">{time.toLocaleTimeString()}</span>
          <span className="date">{time.toLocaleDateString()}</span>
        </div>
      </header>

      <main className="app-main">
        <nav className="sidebar">
          <div className="sidebar-section">
            <span className="section-title">Modules</span>
            {MODULES.map((module) => {
              const Icon = module.icon;
              const active = activeModule === module.id;
              return (
                <button
                  key={module.id}
                  type="button"
                  onClick={() => setActiveModule(module.id)}
                  className={classNames('sidebar-button', active && 'active')}
                >
                  <Icon size={16} />
                  <span>{module.label}</span>
                </button>
              );
            })}
          </div>

          <div className="sidebar-section">
            <span className="section-title">Sys Metrics</span>
            <div className="metric">
              <div className="metric-header">
                <span>CPU</span>
                <span>{metrics.cpu}%</span>
              </div>
              <div className="bar"><div style={{ width: `${metrics.cpu}%` }} /></div>
            </div>
            <div className="metric">
              <div className="metric-header">
                <span>RAM</span>
                <span>{metrics.memory}%</span>
              </div>
              <div className="bar"><div style={{ width: `${metrics.memory}%` }} /></div>
            </div>
            <div className="metric">
              <div className="metric-header">
                <span>NET</span>
                <span>{metrics.network}%</span>
              </div>
              <div className="bar"><div style={{ width: `${metrics.network}%` }} /></div>
            </div>
          </div>
        </nav>

        <section className="content">{moduleContent}</section>

        <aside className="sidebar right">
          <div className="sidebar-section">
            <div className="section-title with-icon">
              <AlertCircle size={16} />
              <span>Notifications</span>
            </div>
            <div className="notification-list">
              {(notifications.length ? notifications : [{ type: 'info', msg: 'No notifications yet', time: '—' }]).map(
                (notif, index) => (
                  <div key={`${notif.msg}-${index}`} className={classNames('notification', notif.type)}>
                    <p>{notif.msg}</p>
                    <span>{notif.time}</span>
                  </div>
                ),
              )}
            </div>
          </div>

          <div className="sidebar-section">
            <div className="section-title with-icon">
              <Clock size={16} />
              <span>Quick Stats</span>
            </div>
            <ul className="stats">
              <li>
                <span>Sessions Today</span>
                <span>{quickStats.sessions_today}</span>
              </li>
              <li>
                <span>Commands Executed</span>
                <span>{quickStats.commands_executed}</span>
              </li>
              <li>
                <span>Data Processed</span>
                <span>{quickStats.data_processed_gb} GB</span>
              </li>
              <li>
                <span>Avg Response</span>
                <span>{quickStats.avg_response_time}s</span>
              </li>
            </ul>
          </div>

          <div className="sidebar-section">
            <span className="section-title">System Status</span>
            <ul className="status-list">
              <li>
                <span>Core Engine</span>
                <span className="ok">●</span>
              </li>
              <li>
                <span>Neural Interface</span>
                <span className="ok">●</span>
              </li>
              <li>
                <span>Voice Bridge</span>
                <span className={connectionIssue ? 'warn-dot' : 'ok'}>●</span>
              </li>
              <li>
                <span>Security Layer</span>
                <span className="ok">●</span>
              </li>
            </ul>
          </div>
        </aside>
      </main>

      <footer className="app-footer">
        <span>ATLAS v2.5.8 | Desktop Bridge</span>
        <span>
          Bridge Status: <strong className={connectionIssue ? 'warn' : 'ok'}>{connectionIssue ? 'DEGRADED' : 'OPERATIONAL'}</strong>
        </span>
      </footer>
    </div>
  );
};

export default AtlasInterface;
