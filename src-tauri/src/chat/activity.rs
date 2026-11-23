//! Read-only native-chat activity for the API idle-unload policy.
//! The chat actor alone owns Writer; readers never queue behind a generation.
use std::{future::Future, sync::Arc, time::Duration};
use tokio::{
    sync::{watch, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock},
    time::Instant,
};

#[derive(Clone, Default)]
struct Snapshot {
    model: String,
    finished: Option<Instant>,
}
#[derive(Clone)]
pub struct Activity(watch::Receiver<Snapshot>, Arc<RwLock<()>>);
pub struct Writer(watch::Sender<Snapshot>, Arc<RwLock<()>>);

pub fn channel() -> (Writer, Activity) {
    let (tx, rx) = watch::channel(Snapshot::default());
    let gate = Arc::new(RwLock::new(()));
    (Writer(tx, gate.clone()), Activity(rx, gate))
}
impl Activity {
    /// Held through the actual conditional unload. Native setup cannot obtain
    /// an endpoint in the check-to-unload gap, even if the engine is busy.
    pub fn idle_reservation(
        &self,
        model: &str,
        idle: Duration,
    ) -> Option<OwnedRwLockWriteGuard<()>> {
        let reservation = self.1.clone().try_write_owned().ok()?;
        if self
            .last_used(model)
            .is_some_and(|used| used.elapsed() < idle)
        {
            return None;
        }
        Some(reservation)
    }
    /// Held through a user-requested text unload. None while a native chat
    /// turn holds the engine (setup through its last token), so the unload
    /// refuses instead of stopping a reply, and no turn can start mid-unload.
    pub fn exclusive(&self) -> Option<OwnedRwLockWriteGuard<()>> {
        self.1.clone().try_write_owned().ok()
    }
    pub fn last_used(&self, model: &str) -> Option<Instant> {
        let snapshot = self.0.borrow();
        (snapshot.model == model).then(|| snapshot.finished.unwrap_or_else(Instant::now))
    }
}
impl Writer {
    /// Take before resolving Ready, and retain through transcript setup and streaming.
    pub async fn reserve(&self) -> OwnedRwLockReadGuard<()> {
        self.1.clone().read_owned().await
    }
    pub async fn during<T>(&self, model: &str, work: impl Future<Output = T>) -> T {
        self.0.send_replace(Snapshot {
            model: model.into(),
            finished: None,
        });
        struct Finish<'a>(&'a watch::Sender<Snapshot>);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0
                    .send_modify(|snapshot| snapshot.finished = Some(Instant::now()));
            }
        }
        let _finish = Finish(&self.0);
        work.await
    }
}
