use serde::{Deserialize, Serialize};
use sysinfo::{DiskExt, NetworkExt, ProcessorExt, System, SystemExt};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemMetrics {
    pub cpu: f32,
    pub memory: f32,
    pub network: f32,
    pub disk: f32,
}

pub fn get_system_metrics() -> SystemMetrics {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu = if sys.processors().is_empty() {
        0.0
    } else {
        sys.processors()
            .iter()
            .map(|p| p.cpu_usage())
            .sum::<f32>()
            / sys.processors().len() as f32
    };

    let memory = if sys.total_memory() == 0 {
        0.0
    } else {
        (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0
    };

    let network = sys
        .networks()
        .iter()
        .map(|(_, data)| (data.received() + data.transmitted()) as f32)
        .sum::<f32>()
        / 1024.0;

    let disk = sys
        .disks()
        .first()
        .map(|disk| {
            let used = disk.total_space().saturating_sub(disk.available_space()) as f32;
            if disk.total_space() == 0 {
                0.0
            } else {
                (used / disk.total_space() as f32) * 100.0
            }
        })
        .unwrap_or(0.0);

    SystemMetrics {
        cpu: cpu.min(100.0),
        memory: memory.min(100.0),
        network: (network / 10.0).min(100.0),
        disk: disk.min(100.0),
    }
}
