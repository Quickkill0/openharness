//! Persistence for the planner: SSD measurements (one per drive) and
//! per-model speed calibration, in `%LOCALAPPDATA%\OpenHarness\planner.json`.
//!
//! A single-writer actor owns the file; readers get clones. The data is
//! small, rebuildable and machine-specific, so it lives in a JSON file rather
//! than the database (no migration).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use super::hardware::SsdMeasurement;
use super::speed::{Calibration, TierMix};
use super::ModelCalibration;

const FILE_VERSION: u32 = 1;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
struct PlannerFile {
    version: u32,
    #[serde(default)]
    ssd: Vec<SsdMeasurement>,
    /// model_id → tier mix → calibration.
    #[serde(default)]
    calibration: BTreeMap<String, ModelCalibration>,
}

/// One measured rate, folded into a model's calibration.
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    pub model_id: String,
    pub mix: TierMix,
    pub raw_decode: f64,
    pub actual_decode: Option<f64>,
    pub raw_prompt: f64,
    pub actual_prompt: Option<f64>,
}

enum Cmd {
    Ssd(oneshot::Sender<Vec<SsdMeasurement>>),
    SetSsd(SsdMeasurement),
    Calibration(String, oneshot::Sender<ModelCalibration>),
    Record(CalibrationSample),
}

#[derive(Clone)]
pub struct PlannerHandle {
    tx: mpsc::Sender<Cmd>,
}

impl PlannerHandle {
    pub async fn ssd_measurements(&self) -> Vec<SsdMeasurement> {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Cmd::Ssd(tx)).await.is_err() {
            return Vec::new();
        }
        rx.await.unwrap_or_default()
    }

    pub async fn set_ssd_measurement(&self, m: SsdMeasurement) {
        let _ = self.tx.send(Cmd::SetSsd(m)).await;
    }

    pub async fn calibration(&self, model_id: &str) -> ModelCalibration {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(Cmd::Calibration(model_id.to_string(), tx)).await.is_err() {
            return ModelCalibration::new();
        }
        rx.await.unwrap_or_default()
    }

    pub async fn record(&self, s: CalibrationSample) {
        let _ = self.tx.send(Cmd::Record(s)).await;
    }
}

pub fn store_path() -> PathBuf {
    crate::config::app_data_dir().join("planner.json")
}

fn load(path: &PathBuf) -> PlannerFile {
    match std::fs::read(path) {
        Ok(bytes) => match serde_json::from_slice::<PlannerFile>(&bytes) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("planner.json unreadable ({e}); starting fresh");
                PlannerFile { version: FILE_VERSION, ..Default::default() }
            }
        },
        Err(_) => PlannerFile { version: FILE_VERSION, ..Default::default() },
    }
}

fn save(path: &PathBuf, f: &PlannerFile) {
    let tmp = path.with_extension("json.tmp");
    let res = serde_json::to_vec_pretty(f)
        .map_err(std::io::Error::other)
        .and_then(|bytes| std::fs::write(&tmp, bytes))
        .and_then(|_| std::fs::rename(&tmp, path));
    if let Err(e) = res {
        tracing::warn!("failed to write planner.json: {e}");
    }
}

fn apply(file: &mut PlannerFile, cmd: Cmd) -> bool {
    match cmd {
        Cmd::Ssd(reply) => {
            let _ = reply.send(file.ssd.clone());
            false
        }
        Cmd::SetSsd(m) => {
            file.ssd.retain(|s| !s.mount.eq_ignore_ascii_case(&m.mount));
            file.ssd.push(m);
            true
        }
        Cmd::Calibration(id, reply) => {
            let _ = reply.send(file.calibration.get(&id).cloned().unwrap_or_default());
            false
        }
        Cmd::Record(s) => {
            let entry: &mut Calibration =
                file.calibration.entry(s.model_id).or_default().entry(s.mix).or_default();
            entry.record(s.raw_decode, s.actual_decode, s.raw_prompt, s.actual_prompt);
            true
        }
    }
}

/// Start the store actor on `path` (normally [`store_path`]).
pub fn start(path: PathBuf) -> PlannerHandle {
    let (tx, mut rx) = mpsc::channel::<Cmd>(64);
    tauri::async_runtime::spawn(async move {
        let mut file = load(&path);
        file.version = FILE_VERSION;
        while let Some(cmd) = rx.recv().await {
            if apply(&mut file, cmd) {
                save(&path, &file);
            }
        }
    });
    PlannerHandle { tx }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str, actual: f64) -> CalibrationSample {
        CalibrationSample {
            model_id: id.into(),
            mix: TierMix::Ssd,
            raw_decode: 10.0,
            actual_decode: Some(actual),
            raw_prompt: 100.0,
            actual_prompt: None,
        }
    }

    #[test]
    fn records_and_round_trips() {
        let mut f = PlannerFile { version: FILE_VERSION, ..Default::default() };
        assert!(apply(&mut f, Cmd::Record(sample("m1", 5.0))));
        assert!(apply(&mut f, Cmd::Record(sample("m1", 5.0))));
        let m = SsdMeasurement {
            mount: "E:\\".into(),
            gbps: 3.4,
            bytes_read: 1 << 30,
            seconds: 0.3,
            measured_at: 1,
            file: "x.gguf".into(),
        };
        apply(&mut f, Cmd::SetSsd(m.clone()));
        apply(&mut f, Cmd::SetSsd(SsdMeasurement { gbps: 3.5, mount: "e:\\".into(), ..m }));
        assert_eq!(f.ssd.len(), 1, "one measurement per drive");
        assert_eq!(f.ssd[0].gbps, 3.5);
        let c = &f.calibration["m1"][&TierMix::Ssd];
        assert_eq!(c.decode_samples, 2);
        assert!((c.decode_ratio - 0.5).abs() < 1e-9);

        let json = serde_json::to_vec(&f).unwrap();
        let back: PlannerFile = serde_json::from_slice(&json).unwrap();
        assert_eq!(back, f);
    }

    #[tokio::test]
    async fn actor_persists_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("planner.json");
        let h = start(path.clone());
        h.record(sample("m2", 20.0)).await;
        let cal = h.calibration("m2").await; // also flushes the queue
        assert_eq!(cal[&TierMix::Ssd].decode_samples, 1);
        let on_disk = load(&path);
        assert_eq!(on_disk.calibration["m2"][&TierMix::Ssd].decode_samples, 1);
        assert!(h.calibration("other").await.is_empty());
    }
}
