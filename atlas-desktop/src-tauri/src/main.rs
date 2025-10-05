#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod system_metrics;

use std::time::Duration;
use system_metrics::SystemMetrics;
use tauri::{AppHandle, Manager};

#[tauri::command]
fn get_metrics() -> SystemMetrics {
    system_metrics::get_system_metrics()
}

fn start_metrics_stream(app_handle: AppHandle) {
    let window = match app_handle.get_window("main") {
        Some(window) => window,
        None => return,
    };

    tauri::async_runtime::spawn(async move {
        loop {
            let metrics = system_metrics::get_system_metrics();
            if let Err(err) = window.emit("system-metrics", metrics) {
                eprintln!("Failed to emit system metrics: {err:?}");
            }
            tokio::time::sleep(Duration::from_secs(3)).await;
        }
    });
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle();
            start_metrics_stream(handle);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_metrics])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
