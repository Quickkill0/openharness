//! Job leases: the orchestrator's own record of generation work in flight.
//!
//! A submit takes its lease in the same actor turn that grants the
//! reservation, before the job row exists or the worker, sidecar or clone CLI
//! has started. A user unload therefore can't slip into the gap between
//! "reserved" and "row inserted". A lease ends on the job's terminal event
//! (`job_done`/`job_error`, which worker exit also reports), on an abort when
//! the submit fails before the job starts (which publishes a `job_error` of
//! its own), or when its job row is found closed, so a lost event can't pin a
//! model forever.

use std::collections::HashMap;

use serde::Serialize;
use sqlx::SqlitePool;

use super::{ReserveOutcome, TTS_RESIDENT_ID};
use crate::db::generation_jobs;
use crate::error::AppError;
use crate::worker::protocol::WorkerEvent;

/// The clone CLI's transient reservation task (see `tts::cli`).
const CLONE_RESIDENT_ID: &str = "pipeline:qwen_tts_clone";

/// A job the frontend should treat as running: a running job row, or a lease
/// whose row isn't written yet. Served to a (re)loaded UI so its locks match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ActiveJob {
    pub id: String,
    pub task: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobLease {
    pub job_id: String,
    /// The resident the job runs on.
    pub resident_id: String,
    /// The task its job row records.
    pub task: String,
}

impl JobLease {
    /// A worker pipeline job (`pipeline:{task}`).
    pub fn worker(job_id: &str, task: &str) -> Self {
        Self { job_id: job_id.into(), resident_id: format!("pipeline:{task}"), task: task.into() }
    }

    /// A Qwen3-TTS sidecar synthesis on the `"tts"` resident.
    pub fn sidecar(job_id: &str) -> Self {
        Self { job_id: job_id.into(), resident_id: TTS_RESIDENT_ID.into(), task: "text_to_speech".into() }
    }

    /// A one-shot Qwen3-TTS voice clone through the CLI.
    pub fn clone_cli(job_id: &str) -> Self {
        Self { job_id: job_id.into(), resident_id: CLONE_RESIDENT_ID.into(), task: "text_to_speech".into() }
    }
}

#[derive(Debug, Default)]
pub struct JobLeases(HashMap<String, JobLease>);

impl JobLeases {
    pub fn grant(&mut self, lease: JobLease) {
        self.0.insert(lease.job_id.clone(), lease);
    }

    /// Called in the actor turn that decided the reservation, before its
    /// reply leaves: no unload can run between "reserved" and the job
    /// starting. Only a granted reservation takes the lease.
    pub fn grant_if_reserved(&mut self, result: &Result<ReserveOutcome, AppError>, lease: Option<JobLease>) {
        if let (Ok(ReserveOutcome::Reserved), Some(lease)) = (result, lease) {
            self.grant(lease);
        }
    }

    pub fn release(&mut self, job_id: &str) {
        self.0.remove(job_id);
    }

    /// End a lease whose job will never report its own end: its submit
    /// failed after the lease was granted (row insert, sidecar not ready,
    /// worker submit). A reloaded UI may already have adopted the job from
    /// `active_jobs`, so the caller publishes the returned `job_error` on the
    /// job bus and DB sync, waiters and the UI all clear the normal way.
    /// None when the lease is already gone (its terminal event came first).
    pub fn abort(&mut self, job_id: &str, error: &str) -> Option<WorkerEvent> {
        self.0
            .remove(job_id)
            .map(|_| WorkerEvent::JobError { job_id: job_id.into(), error: error.into() })
    }

    /// Release on a terminal job event.
    pub fn on_event(&mut self, ev: &WorkerEvent) {
        if let WorkerEvent::JobDone { job_id, .. } | WorkerEvent::JobError { job_id, .. } = ev {
            self.release(job_id);
        }
    }

    /// Whether a live lease holds `resident_id`. A lease is live while its
    /// job row is running or not yet written; one whose row is already closed
    /// (its terminal event was lost) is dropped here.
    pub async fn holds(&mut self, db: &SqlitePool, resident_id: &str) -> bool {
        let ids: Vec<String> = self
            .0
            .values()
            .filter(|l| l.resident_id == resident_id)
            .map(|l| l.job_id.clone())
            .collect();
        let mut held = false;
        for id in ids {
            if lease_live(db, &id).await {
                held = true;
            } else {
                self.release(&id);
            }
        }
        held
    }

    /// Running job rows, with no history cutoff, plus leases whose rows
    /// aren't written yet.
    pub async fn active_jobs(&mut self, db: &SqlitePool) -> Vec<ActiveJob> {
        let mut jobs: Vec<ActiveJob> = generation_jobs::list_running(db)
            .await
            .unwrap_or_else(|e| {
                tracing::warn!(target: "unload", "active-job read failed: {e}");
                Vec::new()
            })
            .into_iter()
            .map(|(id, task)| ActiveJob { id, task })
            .collect();
        let leased: Vec<JobLease> = self.0.values().cloned().collect();
        for lease in leased {
            if jobs.iter().any(|j| j.id == lease.job_id) {
                continue;
            }
            if lease_live(db, &lease.job_id).await {
                jobs.push(ActiveJob { id: lease.job_id, task: lease.task });
            } else {
                self.release(&lease.job_id);
            }
        }
        jobs
    }
}

/// A lease stays live until its row is closed. A missing row means the
/// submit is between its reservation and the row insert; a failed read
/// keeps the lease (fail safe; the terminal event still releases it).
async fn lease_live(db: &SqlitePool, job_id: &str) -> bool {
    match generation_jobs::get(db, job_id).await {
        Ok(Some(job)) => job.status == "running",
        Ok(None) => true,
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn job_pool() -> SqlitePool {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    fn done(job_id: &str) -> WorkerEvent {
        WorkerEvent::JobDone {
            job_id: job_id.into(),
            outputs: vec![],
            meta: serde_json::json!({}),
            task: None,
        }
    }

    /// The race the review found: the reservation is granted and the job is
    /// about to start, but no row exists yet. The lease alone must hold the
    /// resident, for each kind of job.
    #[tokio::test]
    async fn lease_holds_before_the_job_row_exists() {
        let pool = job_pool().await;
        let mut leases = JobLeases::default();
        leases.grant(JobLease::sidecar("q1"));
        leases.grant(JobLease::clone_cli("c1"));
        leases.grant(JobLease::worker("w1", "text_to_image"));

        assert!(leases.holds(&pool, TTS_RESIDENT_ID).await);
        assert!(leases.holds(&pool, CLONE_RESIDENT_ID).await);
        assert!(leases.holds(&pool, "pipeline:text_to_image").await);
        assert!(!leases.holds(&pool, "pipeline:transcribe").await);

        // A reloaded UI sees the not-yet-written jobs too.
        let mut active = leases.active_jobs(&pool).await;
        active.sort_by(|a, b| a.id.cmp(&b.id));
        assert_eq!(
            active,
            vec![
                ActiveJob { id: "c1".into(), task: "text_to_speech".into() },
                ActiveJob { id: "q1".into(), task: "text_to_speech".into() },
                ActiveJob { id: "w1".into(), task: "text_to_image".into() },
            ]
        );
    }

    /// Only a granted reservation takes the lease; a needs-eviction answer or
    /// an error leaves nothing to release.
    #[tokio::test]
    async fn lease_is_taken_only_with_a_granted_reservation() {
        let pool = job_pool().await;
        let mut leases = JobLeases::default();
        let refused: Result<ReserveOutcome, AppError> = Ok(ReserveOutcome::NeedsEviction {
            required_mb: 9000,
            free_mb: 1000,
            candidates: vec![],
        });
        leases.grant_if_reserved(&refused, Some(JobLease::sidecar("q0")));
        leases.grant_if_reserved(&Err(AppError::Other("x".into())), Some(JobLease::sidecar("q1")));
        assert!(!leases.holds(&pool, TTS_RESIDENT_ID).await);

        leases.grant_if_reserved(&Ok(ReserveOutcome::Reserved), Some(JobLease::sidecar("q2")));
        assert!(leases.holds(&pool, TTS_RESIDENT_ID).await);
        // A submit that fails before its job starts gives the lease back.
        leases.release("q2");
        assert!(!leases.holds(&pool, TTS_RESIDENT_ID).await);
    }

    /// Terminal events release; worker exit arrives as job_error.
    #[tokio::test]
    async fn terminal_events_release_the_lease() {
        let pool = job_pool().await;
        let mut leases = JobLeases::default();
        leases.grant(JobLease::sidecar("q1"));
        leases.grant(JobLease::worker("w1", "transcribe"));

        leases.on_event(&done("q1"));
        leases.on_event(&WorkerEvent::JobError {
            job_id: "w1".into(),
            error: "worker process exited".into(),
        });
        assert!(!leases.holds(&pool, TTS_RESIDENT_ID).await);
        assert!(!leases.holds(&pool, "pipeline:transcribe").await);
        assert!(leases.active_jobs(&pool).await.is_empty());
    }

    /// A reload adopted a lease whose row was never written, then the row
    /// insert failed. Aborting the lease yields the `job_error` that clears
    /// the adopted entry, and only once.
    #[tokio::test]
    async fn aborting_an_adopted_lease_reports_its_end() {
        let pool = job_pool().await;
        let mut leases = JobLeases::default();
        leases.grant(JobLease::sidecar("q1"));
        assert_eq!(
            leases.active_jobs(&pool).await,
            vec![ActiveJob { id: "q1".into(), task: "text_to_speech".into() }]
        );

        match leases.abort("q1", "database is locked") {
            Some(WorkerEvent::JobError { job_id, error }) => {
                assert_eq!(job_id, "q1");
                assert_eq!(error, "database is locked");
            }
            other => panic!("expected a job_error, got {other:?}"),
        }
        assert!(leases.active_jobs(&pool).await.is_empty());
        assert!(!leases.holds(&pool, TTS_RESIDENT_ID).await);
        // Its terminal event already came (or it was aborted): nothing to report.
        assert!(leases.abort("q1", "again").is_none());
    }

    /// A lost terminal event can't pin a model: once the row is closed the
    /// lease is dropped. A running row keeps it.
    #[tokio::test]
    async fn closed_row_drops_a_lease_whose_event_was_lost() {
        let pool = job_pool().await;
        let mut leases = JobLeases::default();
        leases.grant(JobLease::sidecar("q1"));
        generation_jobs::insert(&pool, "q1", "text_to_speech", &serde_json::json!({}))
            .await
            .unwrap();
        assert!(leases.holds(&pool, TTS_RESIDENT_ID).await);

        generation_jobs::mark_error(&pool, "q1", "boom").await.unwrap();
        assert!(!leases.holds(&pool, TTS_RESIDENT_ID).await);
        assert!(leases.0.is_empty());
    }
}
