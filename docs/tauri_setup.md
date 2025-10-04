# ATLAS Desktop App - Tauri Setup Guide

## Prerequisites

Before starting, install:
- **Node.js** (v18+): https://nodejs.org/
- **Rust**: https://www.rust-lang.org/tools/install
  ```bash
  # On macOS/Linux:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

  # On Windows: Download from rust-lang.org
  ```

> **Already in this repo**
>
> The `atlas-desktop/` directory checked into this repository is a fully wired Tauri project that mirrors the walkthrough below. It speaks to the Python core through the FastAPI bridge exposed in `atlas_main/desktop_server.py`. If you just want to run the desktop build that ships with Atlas:
>
> ```bash
> # 1. Start the desktop bridge so the UI can talk to Atlas
> python -m atlas_main.desktop_server --port 5175
>
> # 2. In a new terminal run the Tauri shell (installs deps on first run)
> cd atlas-desktop
> npm install
> npm run tauri dev
> ```
>
> The React layer will poll `http://127.0.0.1:5175` by default. Set `VITE_ATLAS_API` in a `.env` file if you expose the bridge on a different host/port. The rest of this guide documents how that project was assembled from scratch.

## Step 1: Create New Project

```bash
# Create a new React + Tauri project
npm create tauri-app@latest

# Follow prompts:
# Project name: atlas-desktop
# Choose a package manager: npm
# Choose your UI template: React
# Choose your UI flavor: JavaScript
```

## Step 2: Project Structure

Your project will look like this:
```
atlas-desktop/
├── src/               # React frontend
│   ├── App.jsx
│   ├── main.jsx
│   └── styles.css
├── src-tauri/         # Rust backend
│   ├── src/
│   │   └── main.rs
│   ├── Cargo.toml
│   └── tauri.conf.json
├── package.json
└── vite.config.js
```

## Step 3: Replace App.jsx with ATLAS

Replace the contents of `src/App.jsx` with the ATLAS interface:

```jsx
import React, { useState, useEffect } from 'react';
import { Terminal, Globe, Network, Activity, Cpu, Clock, TrendingUp, AlertCircle } from 'lucide-react';

const AtlasInterface = () => {
  const [time, setTime] = useState(new Date());
  const [activeModule, setActiveModule] = useState('terminal');
  const [terminalHistory, setTerminalHistory] = useState([
    { type: 'system', text: 'ATLAS INITIALIZED - ALL SYSTEMS OPERATIONAL' },
    { type: 'info', text: 'Voice recognition: ONLINE' },
    { type: 'info', text: 'Natural language processing: ACTIVE' },
  ]);
  const [input, setInput] = useState('');
  const [metrics, setMetrics] = useState({
    cpu: 42,
    memory: 67,
    network: 89,
    tasks: 12
  });

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const metricsInterval = setInterval(() => {
      setMetrics({
        cpu: Math.floor(Math.random() * 40) + 30,
        memory: Math.floor(Math.random() * 30) + 50,
        network: Math.floor(Math.random() * 20) + 75,
        tasks: Math.floor(Math.random() * 10) + 8
      });
    }, 3000);
    return () => clearInterval(metricsInterval);
  }, []);

  const handleCommand = (e) => {
    if (!input.trim()) return;

    const newHistory = [...terminalHistory, { type: 'command', text: `> ${input}` }];
    
    if (input.toLowerCase().includes('status')) {
      newHistory.push({ type: 'success', text: 'All systems operational. No anomalies detected.' });
    } else if (input.toLowerCase().includes('time')) {
      newHistory.push({ type: 'info', text: `Current time: ${time.toLocaleTimeString()}` });
    } else if (input.toLowerCase().includes('help')) {
      newHistory.push({ type: 'info', text: 'Available commands: status, time, metrics, clear, analyze' });
    } else if (input.toLowerCase().includes('clear')) {
      setTerminalHistory([{ type: 'system', text: 'Terminal cleared.' }]);
      setInput('');
      return;
    } else if (input.toLowerCase().includes('metrics')) {
      newHistory.push({ type: 'data', text: `CPU: ${metrics.cpu}% | RAM: ${metrics.memory}% | NET: ${metrics.network}%` });
    } else {
      newHistory.push({ type: 'success', text: `Processing: "${input}"... Command acknowledged.` });
    }

    setTerminalHistory(newHistory);
    setInput('');
  };

  const modules = [
    { id: 'terminal', icon: Terminal, label: 'TERMINAL' },
    { id: 'analytics', icon: Activity, label: 'ANALYTICS' },
    { id: 'network', icon: Network, label: 'NETWORK' },
    { id: 'global', icon: Globe, label: 'GLOBAL' },
  ];

  return (
    <div className="min-h-screen bg-black text-green-400 font-mono p-4">
      {/* Header */}
      <div className="border-b border-green-900 pb-4 mb-4">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-yellow-500 tracking-wider">A.T.L.A.S.</h1>
            <p className="text-xs text-green-600 mt-1">Advanced Tactical Logistics and Analysis System</p>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-yellow-500">{time.toLocaleTimeString()}</div>
            <div className="text-xs text-green-600">{time.toLocaleDateString()}</div>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="grid grid-cols-12 gap-4">
        {/* Left Sidebar - Module Navigation */}
        <div className="col-span-2 space-y-2">
          <div className="border border-green-900 bg-green-950 bg-opacity-20 p-2">
            <div className="text-xs text-yellow-500 mb-2">MODULES</div>
            {modules.map(mod => {
              const Icon = mod.icon;
              return (
                <button
                  key={mod.id}
                  onClick={() => setActiveModule(mod.id)}
                  className={`w-full flex items-center gap-2 p-2 mb-1 text-xs transition-colors ${
                    activeModule === mod.id 
                      ? 'bg-yellow-900 bg-opacity-30 text-yellow-400 border-l-2 border-yellow-500' 
                      : 'text-green-500 hover:bg-green-900 hover:bg-opacity-20'
                  }`}
                >
                  <Icon size={14} />
                  {mod.label}
                </button>
              );
            })}
          </div>

          {/* System Metrics */}
          <div className="border border-green-900 bg-green-950 bg-opacity-20 p-2">
            <div className="text-xs text-yellow-500 mb-2">SYS METRICS</div>
            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <span>CPU</span>
                  <span className="text-yellow-400">{metrics.cpu}%</span>
                </div>
                <div className="w-full bg-green-950 h-1">
                  <div className="bg-green-500 h-1" style={{width: `${metrics.cpu}%`}}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span>RAM</span>
                  <span className="text-yellow-400">{metrics.memory}%</span>
                </div>
                <div className="w-full bg-green-950 h-1">
                  <div className="bg-green-500 h-1" style={{width: `${metrics.memory}%`}}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span>NET</span>
                  <span className="text-yellow-400">{metrics.network}%</span>
                </div>
                <div className="w-full bg-green-950 h-1">
                  <div className="bg-green-500 h-1" style={{width: `${metrics.network}%`}}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="col-span-7">
          {activeModule === 'terminal' && (
            <div className="border border-green-900 bg-black h-[600px] flex flex-col">
              <div className="bg-green-900 bg-opacity-20 px-4 py-2 border-b border-green-900 flex items-center gap-2">
                <Terminal size={16} className="text-yellow-500" />
                <span className="text-xs text-yellow-500">COMMAND TERMINAL</span>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-1 text-sm">
                {terminalHistory.map((entry, i) => (
                  <div key={i} className={`${
                    entry.type === 'system' ? 'text-yellow-500' :
                    entry.type === 'command' ? 'text-green-300' :
                    entry.type === 'success' ? 'text-green-400' :
                    entry.type === 'error' ? 'text-red-400' :
                    'text-green-500'
                  }`}>
                    {entry.text}
                  </div>
                ))}
              </div>
              <div className="border-t border-green-900 p-4">
                <div className="flex items-center gap-2">
                  <span className="text-yellow-500">$</span>
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleCommand(e);
                      }
                    }}
                    className="flex-1 bg-transparent outline-none text-green-400"
                    placeholder="Enter command..."
                  />
                </div>
              </div>
            </div>
          )}

          {activeModule === 'analytics' && (
            <div className="border border-green-900 bg-black p-4 h-[600px]">
              <div className="flex items-center gap-2 mb-4 pb-2 border-b border-green-900">
                <Activity size={16} className="text-yellow-500" />
                <span className="text-xs text-yellow-500">SYSTEM ANALYTICS</span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="border border-green-900 bg-green-950 bg-opacity-10 p-4">
                  <div className="text-xs text-green-600 mb-2">PERFORMANCE INDEX</div>
                  <div className="text-3xl text-yellow-500 font-bold">94.7%</div>
                  <div className="text-xs text-green-500 mt-1">↑ 2.3% from baseline</div>
                </div>
                <div className="border border-green-900 bg-green-950 bg-opacity-10 p-4">
                  <div className="text-xs text-green-600 mb-2">ACTIVE PROCESSES</div>
                  <div className="text-3xl text-yellow-500 font-bold">{metrics.tasks}</div>
                  <div className="text-xs text-green-500 mt-1">All nominal</div>
                </div>
                <div className="border border-green-900 bg-green-950 bg-opacity-10 p-4">
                  <div className="text-xs text-green-600 mb-2">NETWORK LATENCY</div>
                  <div className="text-3xl text-yellow-500 font-bold">12ms</div>
                  <div className="text-xs text-green-500 mt-1">Optimal range</div>
                </div>
                <div className="border border-green-900 bg-green-950 bg-opacity-10 p-4">
                  <div className="text-xs text-green-600 mb-2">UPTIME</div>
                  <div className="text-3xl text-yellow-500 font-bold">99.9%</div>
                  <div className="text-xs text-green-500 mt-1">47 days continuous</div>
                </div>
              </div>
              <div className="mt-4 border border-green-900 bg-green-950 bg-opacity-10 p-4">
                <div className="text-xs text-green-600 mb-3">ACTIVITY TIMELINE</div>
                <div className="space-y-2 text-xs">
                  {[
                    { time: '14:32:15', event: 'System scan completed', status: 'success' },
                    { time: '14:28:42', event: 'Background optimization running', status: 'info' },
                    { time: '14:15:03', event: 'Network connection verified', status: 'success' },
                    { time: '14:01:22', event: 'Cache cleared successfully', status: 'success' },
                  ].map((log, i) => (
                    <div key={i} className="flex items-center gap-3 border-l-2 border-green-700 pl-3 py-1">
                      <span className="text-green-600">{log.time}</span>
                      <span className="flex-1">{log.event}</span>
                      <span className={log.status === 'success' ? 'text-green-400' : 'text-yellow-400'}>
                        {log.status === 'success' ? '✓' : '◉'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeModule === 'network' && (
            <div className="border border-green-900 bg-black p-4 h-[600px]">
              <div className="flex items-center gap-2 mb-4 pb-2 border-b border-green-900">
                <Network size={16} className="text-yellow-500" />
                <span className="text-xs text-yellow-500">NETWORK TOPOLOGY</span>
              </div>
              <div className="relative h-[500px] flex items-center justify-center">
                <svg className="w-full h-full">
                  <circle cx="50%" cy="50%" r="80" fill="none" stroke="currentColor" strokeWidth="1" className="text-green-800" />
                  <circle cx="50%" cy="50%" r="120" fill="none" stroke="currentColor" strokeWidth="1" className="text-green-800" strokeDasharray="4,4" />
                  <circle cx="50%" cy="50%" r="160" fill="none" stroke="currentColor" strokeWidth="1" className="text-green-800" strokeDasharray="8,8" />
                  
                  {/* Central node */}
                  <circle cx="50%" cy="50%" r="10" fill="currentColor" className="text-yellow-500" />
                  
                  {/* Peripheral nodes */}
                  {[0, 60, 120, 180, 240, 300].map((angle, i) => {
                    const rad = (angle * Math.PI) / 180;
                    const x = 50 + Math.cos(rad) * 25;
                    const y = 50 + Math.sin(rad) * 25;
                    return (
                      <g key={i}>
                        <line 
                          x1="50%" 
                          y1="50%" 
                          x2={`${x}%`} 
                          y2={`${y}%`} 
                          stroke="currentColor" 
                          strokeWidth="1" 
                          className="text-green-700"
                        />
                        <circle cx={`${x}%`} cy={`${y}%`} r="6" fill="currentColor" className="text-green-500" />
                      </g>
                    );
                  })}
                </svg>
                <div className="absolute bottom-4 left-4 text-xs space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <span>Core System</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span>Connected Nodes</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeModule === 'global' && (
            <div className="border border-green-900 bg-black p-4 h-[600px]">
              <div className="flex items-center gap-2 mb-4 pb-2 border-b border-green-900">
                <Globe size={16} className="text-yellow-500" />
                <span className="text-xs text-yellow-500">GLOBAL MONITORING</span>
              </div>
              <div className="text-center text-green-600 text-sm mt-20">
                Global monitoring systems active
                <div className="mt-4 text-xs">Tracking {Math.floor(Math.random() * 20) + 10} endpoints worldwide</div>
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar - Status & Notifications */}
        <div className="col-span-3 space-y-4">
          <div className="border border-green-900 bg-green-950 bg-opacity-20 p-3">
            <div className="text-xs text-yellow-500 mb-3 flex items-center gap-2">
              <AlertCircle size={14} />
              NOTIFICATIONS
            </div>
            <div className="space-y-2 text-xs">
              {[
                { type: 'info', msg: 'System update available', time: '2m ago' },
                { type: 'success', msg: 'Backup completed', time: '15m ago' },
                { type: 'warn', msg: 'High CPU usage detected', time: '1h ago' },
              ].map((notif, i) => (
                <div key={i} className="border-l-2 border-green-700 pl-2 py-1">
                  <div className={
                    notif.type === 'warn' ? 'text-yellow-400' :
                    notif.type === 'success' ? 'text-green-400' : 'text-green-500'
                  }>{notif.msg}</div>
                  <div className="text-green-700 text-xs mt-1">{notif.time}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="border border-green-900 bg-green-950 bg-opacity-20 p-3">
            <div className="text-xs text-yellow-500 mb-3 flex items-center gap-2">
              <Clock size={14} />
              QUICK STATS
            </div>
            <div className="space-y-3 text-xs">
              <div className="flex justify-between">
                <span className="text-green-600">Sessions Today</span>
                <span className="text-green-400 font-bold">27</span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-600">Commands Executed</span>
                <span className="text-green-400 font-bold">1,432</span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-600">Data Processed</span>
                <span className="text-green-400 font-bold">4.7 GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-green-600">Avg Response Time</span>
                <span className="text-green-400 font-bold">0.3s</span>
              </div>
            </div>
          </div>

          <div className="border border-green-900 bg-green-950 bg-opacity-20 p-3">
            <div className="text-xs text-yellow-500 mb-3">SYSTEM STATUS</div>
            <div className="space-y-2 text-xs">
              {['Core Engine', 'Neural Network', 'Voice Recognition', 'Security Layer'].map((system, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-green-500">{system}</span>
                  <span className="text-green-400">●</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer Status Bar */}
      <div className="mt-4 pt-4 border-t border-green-900 text-xs flex justify-between text-green-600">
        <span>ATLAS v2.5.8 | Desktop Edition</span>
        <span>All Systems: <span className="text-green-400">OPERATIONAL</span></span>
      </div>
    </div>
  );
};

export default AtlasInterface;
```

## Step 4: Install Dependencies

```bash
# Navigate to project directory
cd atlas-desktop

# Install lucide-react for icons
npm install lucide-react
```

## Step 5: Configure Tauri

Edit `src-tauri/tauri.conf.json`:

```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:5173",
    "distDir": "../dist"
  },
  "package": {
    "productName": "ATLAS",
    "version": "1.0.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true
      },
      "fs": {
        "all": true,
        "scope": ["$APPDATA/*", "$HOME/*"]
      },
      "notification": {
        "all": true
      },
      "globalShortcut": {
        "all": true
      }
    },
    "bundle": {
      "active": true,
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "identifier": "com.atlas.app",
      "targets": "all"
    },
    "security": {
      "csp": null
    },
    "windows": [
      {
        "fullscreen": false,
        "resizable": true,
        "title": "ATLAS",
        "width": 1400,
        "height": 900,
        "minWidth": 1200,
        "minHeight": 800,
        "decorations": true,
        "transparent": false
      }
    ]
  }
}
```

## Step 6: Run Development Server

```bash
# Start the development server
npm run tauri dev
```

This will:
1. Start the React dev server
2. Launch the Tauri desktop window
3. Enable hot-reload for instant updates

## Step 7: Build Production App

```bash
# Create production build
npm run tauri build
```

Your app will be built to:
- **macOS**: `src-tauri/target/release/bundle/dmg/`
- **Windows**: `src-tauri/target/release/bundle/msi/`
- **Linux**: `src-tauri/target/release/bundle/deb/` or `appimage/`

## Optional Enhancements

### Add System Tray Icon

Edit `src-tauri/src/main.rs`:

```rust
use tauri::{CustomMenuItem, SystemTray, SystemTrayMenu, SystemTrayEvent};
use tauri::Manager;

fn main() {
    let quit = CustomMenuItem::new("quit".to_string(), "Quit");
    let show = CustomMenuItem::new("show".to_string(), "Show");
    let tray_menu = SystemTrayMenu::new()
        .add_item(show)
        .add_item(quit);
    
    let system_tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .system_tray(system_tray)
        .on_system_tray_event(|app, event| match event {
            SystemTrayEvent::LeftClick { .. } => {
                let window = app.get_window("main").unwrap();
                window.show().unwrap();
            }
            SystemTrayEvent::MenuItemClick { id, .. } => {
                match id.as_str() {
                    "quit" => {
                        std::process::exit(0);
                    }
                    "show" => {
                        let window = app.get_window("main").unwrap();
                        window.show().unwrap();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Add Global Keyboard Shortcut

In `src-tauri/src/main.rs`, add:

```rust
use tauri::GlobalShortcutManager;

// In your main function, after Builder:
.setup(|app| {
    let mut shortcut = app.global_shortcut_manager();
    shortcut.register("CommandOrControl+Shift+A", move || {
        // Toggle window visibility
        println!("ATLAS activated via hotkey!");
    })?;
    Ok(())
})
```

## Troubleshooting

### Rust not found?
```bash
# Restart terminal after installing Rust, or run:
source $HOME/.cargo/env
```

### Build errors?
```bash
# Clear cache and rebuild
rm -rf node_modules
npm install
npm run tauri build
```

### Icons missing?
Generate icons from a 1024x1024 PNG:
```bash
npm install -g @tauri-apps/cli
cargo install tauri-cli
npm run tauri icon path/to/icon.png
```

## Next Steps

Now you can add:
- **Real system metrics** using Tauri's system info APIs
- **File system access** for managing data
- **Native notifications** for alerts
- **Auto-launch** on system startup
- **Global shortcuts** for quick access
- **Database** (SQLite) for persistent storage
- **API integrations** for weather, news, etc.

Your ATLAS desktop app is ready to deploy! 🚀
