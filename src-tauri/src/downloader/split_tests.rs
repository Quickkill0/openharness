use super::*;
use crate::models::registry::split_tests::fixture_files;

async fn manager() -> Manager {
    let pool = sqlx::sqlite::SqlitePoolOptions::new().max_connections(1)
        .connect("sqlite::memory:").await.unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    Manager::new(pool, reqwest::Client::new(), 1, broadcast::channel(16).0)
}

#[tokio::test]
async fn split_pause_at_fence_checkpoints_part_and_resumes_to_registration() {
    use sha2::{Digest, Sha256};
    let mut mgr = manager().await; // one DB connection also catches leaked fences
    let dir = tempfile::tempdir().unwrap();
    let files = fixture_files(dir.path(), "Q4/model");
    let gid = Uuid::new_v4(); let gid_str = gid.to_string();
    persist_attempt(&mut mgr, gid, &files, "other", JobStatus::Done, "gguf").await;
    let id = *mgr.group_member_ids(gid).iter().find(|id| mgr.jobs[id].file.contains("-00001-of-")).unwrap();
    let final_path = std::path::PathBuf::from(&files[0].path);
    let part_path = dir.path().join("verified.part");
    let bytes = std::fs::read(&final_path).unwrap();
    let expected = format!("{:x}", Sha256::digest(&bytes));
    std::fs::rename(&final_path, &part_path).unwrap();
    let job = mgr.jobs.get_mut(&id).unwrap();
    job.status = JobStatus::Running; job.expected_sha256 = Some(expected.clone());
    // Simulate a small transfer that finished before the periodic DB flush.
    job.chunks.clear(); job.downloaded = 0;
    db_dl::upsert_job(&mgr.pool, job).await.unwrap();
    let mut transferred = job.clone();
    transferred.downloaded = bytes.len() as u64;
    transferred.chunks = vec![job::ChunkState { start: 0, end: transferred.downloaded - 1, completed: transferred.downloaded }];
    let pause = CancelToken::new();
    let result = {
        let held = mgr.pool.begin().await.unwrap(); // block fence acquisition
        let finalizing = finalize_group_member(&mgr.pool, &transferred, Some(&gid_str), &part_path, &final_path, &pause);
        tokio::pin!(finalizing);
        tokio::select! {
            result = &mut finalizing => panic!("fence must wait for the held connection: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        pause.pause(); // arrives while the finalizer is waiting at the fence
        held.rollback().await.unwrap();
        finalizing.await
    };
    let status = match result {
        Err(AppError::DownloadPaused) => JobStatus::Paused,
        other => panic!("pause must not cancel a verified member: {other:?}"),
    };
    db_dl::update_status(&mgr.pool, id, status, None, 0).await.unwrap();
    handle_command(&mut mgr, Command::JobFinished { id, final_status: status, error: None }, mpsc::channel(1).0).await;
    assert_eq!(mgr.jobs[&id].status, JobStatus::Paused);
    assert!(!closed(&mgr, gid).await); // transaction dropped before checkpoint
    assert_eq!(std::fs::read(&part_path).unwrap(), bytes);
    assert!(!final_path.exists());
    mgr.resume_job(id).await;
    let resumed = mgr.jobs[&id].clone();
    assert_eq!(resumed.status, JobStatus::Queued);
    assert_eq!(resumed.downloaded, transferred.downloaded);
    assert!(resumed.chunks.iter().all(|c| c.is_done()));
    // Completed persisted chunks must make no GET, even with an unreachable URL.
    tokio::time::timeout(Duration::from_millis(250), run_chunks(mgr.http.clone(),
        "http://127.0.0.1:1/must-not-download".into(), None, part_path.clone(), resumed.chunks.clone(),
        mpsc::channel(1).0, CancelToken::new())).await.unwrap().unwrap();
    // Hash state is deliberately rebuilt from the retained bytes on every resume.
    let rolled = RollingHasher::spawn(part_path.clone(), contiguous_frontier(&resumed.chunks)).finish().await;
    assert_eq!(hasher::finish_hash_with_progress(rolled, &part_path, resumed.total_bytes, |_| {}).unwrap(), expected);
    finalize_group_member(&mgr.pool, &resumed, Some(&gid_str), &part_path, &final_path, &CancelToken::new()).await.unwrap();
    db_dl::update_status(&mgr.pool, id, JobStatus::Done, None, 0).await.unwrap();
    handle_command(&mut mgr, Command::JobFinished { id, final_status: JobStatus::Done, error: None }, mpsc::channel(1).0).await;
    mgr.register_group_at(gid, Some(dir.path().into())).await;
    let installed = registry::list(&mgr.pool).await.unwrap();
    assert_eq!(installed.len(), 1);
    assert_eq!(installed[0].primary_sha256.as_deref(), Some(expected.as_str()));
    assert!(closed(&mgr, gid).await && !part_path.exists());
}

#[tokio::test]
async fn split_cancel_and_rename_failure_release_fence_and_keep_part_retryable() {
    let mut mgr = manager().await;
    let dir = tempfile::tempdir().unwrap();
    let files = fixture_files(dir.path(), "Q4/model");
    let gid = Uuid::new_v4(); let gid_str = gid.to_string();
    persist_attempt(&mut mgr, gid, &files, "hash", JobStatus::Done, "gguf").await;
    let id = mgr.group_member_ids(gid)[0];
    let job = mgr.jobs[&id].clone();
    let final_path = dir.path().join(&job.file);
    let part_path = dir.path().join("verified.part");
    std::fs::rename(&final_path, &part_path).unwrap();
    let cancel = CancelToken::new(); cancel.cancel();
    assert!(matches!(finalize_group_member(&mgr.pool, &job, Some(&gid_str), &part_path,
        &final_path, &cancel).await, Err(AppError::DownloadCancelled)));
    assert!(!closed(&mgr, gid).await && part_path.is_file());
    let blocked = dir.path().join("directory-not-a-file"); std::fs::create_dir(&blocked).unwrap();
    let error = finalize_group_member(&mgr.pool, &job, Some(&gid_str), &part_path,
        &blocked, &CancelToken::new()).await.unwrap_err();
    assert!(!matches!(error, AppError::DownloadPaused | AppError::DownloadCancelled));
    // The task maps ordinary I/O errors to Failed, retaining their message.
    mgr.jobs.get_mut(&id).unwrap().status = JobStatus::Running;
    let message = error.to_string();
    db_dl::update_status(&mgr.pool, id, JobStatus::Failed, Some(&message), 0).await.unwrap();
    handle_command(&mut mgr, Command::JobFinished { id, final_status: JobStatus::Failed, error: Some(message) }, mpsc::channel(1).0).await;
    assert_eq!(mgr.jobs[&id].status, JobStatus::Failed);
    assert!(!closed(&mgr, gid).await && part_path.is_file()); // no leaked transaction
    mgr.resume_job(id).await;
    assert_eq!(mgr.jobs[&id].status, JobStatus::Queued);
    finalize_group_member(&mgr.pool, &job, Some(&gid_str), &part_path, &final_path, &CancelToken::new()).await.unwrap();
    assert!(final_path.is_file());
}

#[tokio::test]
async fn split_progress_carries_new_refusal_and_ordinary_errors_and_clears_on_retry() {
    let mut mgr = manager().await;
    let dir = tempfile::tempdir().unwrap();
    let files = fixture_files(dir.path(), "Q4/model");
    let old = start(&mut mgr, &files).await.unwrap();
    handle_command(&mut mgr, Command::CancelGroup(old), mpsc::channel(1).0).await;
    start(&mut mgr, &files).await.unwrap();
    let id = mgr.group_member_ids(old)[0];
    let job = mgr.jobs.get_mut(&id).unwrap();
    job.status = JobStatus::Failed; job.error = Some("old integrity failure".into());
    let mut events = mgr.progress_tx.subscribe();
    mgr.resume_job(id).await;
    let refused = events.recv().await.unwrap();
    assert_eq!(refused.status, JobStatus::Failed);
    assert!(refused.error.as_deref().unwrap().contains("superseded"));
    assert!(serde_json::to_value(&refused).unwrap()["error"].as_str().unwrap().contains("superseded"));
    let mut single = mgr.jobs[&id].clone();
    single.id = Uuid::new_v4(); single.group_id = None; single.file = "single.gguf".into(); single.status = JobStatus::Running;
    let id = single.id; db_dl::upsert_job(&mgr.pool, &single).await.unwrap(); mgr.insert_job(single);
    handle_command(&mut mgr, Command::JobFinished { id, final_status: JobStatus::Failed, error: Some("network disconnected".into()) }, mpsc::channel(1).0).await;
    assert_eq!(events.recv().await.unwrap().error.as_deref(), Some("network disconnected"));
    mgr.resume_job(id).await;
    let retry = events.recv().await.unwrap();
    assert_eq!(retry.status, JobStatus::Queued);
    assert!(serde_json::to_value(retry).unwrap()["error"].is_null());
}

fn hf(path: &str) -> HfFile {
    HfFile { path: path.into(), size: 10, sha256: None, is_lfs: true }
}

#[tokio::test]
async fn split_single_download_refused_before_job_persistence() {
    let mut mgr = manager().await;
    let (cmd_tx, _) = mpsc::channel(1);
    for path in ["q/model-00001-of-00004.gguf", "q/model-00004-of-00004.gguf"] {
        let (tx, rx) = oneshot::channel();
        handle_command(&mut mgr, Command::Start { repo_id: "a/b".into(), revision: "main".into(), file: hf(path), response: tx }, cmd_tx.clone()).await;
        assert!(rx.await.unwrap().unwrap_err().to_string().contains("complete set"));
    }
    assert!(mgr.jobs.is_empty());
    assert!(db_dl::list_all(&mgr.pool).await.unwrap().is_empty());
    assert!(check_single_file("model-00001-of-00001.gguf").is_ok());
    assert!(check_single_file("model-Q4_K_M.gguf").is_ok());
    let legacy = Job { id: Uuid::new_v4(), repo_id: "a/b".into(), revision: "main".into(),
        file: "model-00001-of-00004.gguf".into(), dest_path: Default::default(),
        total_bytes: 10, downloaded: 0, chunks: vec![], status: JobStatus::Paused,
        error: None, expected_sha256: None, group_id: None, created_at: 0, updated_at: 0 };
    let error = run_job(mgr.pool.clone(), mgr.http.clone(), legacy, CancelToken::new(), mgr.progress_tx.clone()).await.unwrap_err();
    assert!(error.to_string().contains("complete set"));
}

#[tokio::test]
async fn split_group_lifecycle_allows_other_quants_and_heals_only_existing_files() {
    let mut mgr = manager().await;
    let dir = tempfile::tempdir().unwrap();
    let (cmd_tx, _) = mpsc::channel(1);
    let mut groups = vec![];
    for stem in ["UD-Q2/model-Q2", "UD-Q4/model-Q4"] {
        let files = fixture_files(dir.path(), stem).iter().map(|f| hf(&f.rel_path)).collect();
        let (tx, rx) = oneshot::channel();
        handle_command(&mut mgr, Command::StartRepo { repo_id: "a/b".into(), revision: "main".into(), files, format: "gguf".into(), task: None, response: tx }, cmd_tx.clone()).await;
        groups.push(rx.await.unwrap().unwrap());
    }
    let (tx, rx) = oneshot::channel();
    let duplicate = mgr.jobs.values().filter(|j| j.group_id == Some(groups[0])).map(|j| hf(&j.file)).collect();
    handle_command(&mut mgr, Command::StartRepo { repo_id: "a/b".into(), revision: "main".into(), files: duplicate, format: "gguf".into(), task: None, response: tx }, cmd_tx).await;
    assert!(rx.await.unwrap().is_err());
    // Partial completion cannot publish a model.
    mgr.register_group_at(groups[0], Some(dir.path().into())).await;
    assert!(registry::list(&mgr.pool).await.unwrap().is_empty());
    for job in mgr.jobs.values_mut() { job.status = JobStatus::Done; }
    for gid in &groups {
        mgr.register_group_at(*gid, Some(dir.path().into())).await;
        mgr.register_group_at(*gid, Some(dir.path().into())).await; // startup idempotency
    }
    let models = registry::list(&mgr.pool).await.unwrap();
    assert_eq!(models.len(), 2);
    assert!(models.iter().all(|m| m.primary_file_path.ends_with("-00001-of-00004.gguf") && m.caps.workflows.contains(&crate::models::capabilities::Workflow::Chat)));
    // Done job rows remain after a user deletes a model. Startup must not revive it.
    let removed = &models[0];
    crate::db::models::delete_by_id(&mgr.pool, &removed.id).await.unwrap();
    for path in crate::models::gguf::split_files(std::path::Path::new(&removed.primary_file_path)).unwrap() { std::fs::remove_file(path).unwrap(); }
    for gid in &groups { mgr.register_group_at(*gid, Some(dir.path().into())).await; }
    assert_eq!(registry::list(&mgr.pool).await.unwrap().len(), 1);
}

async fn persist_attempt(mgr: &mut Manager, gid: Uuid, files: &[registry::RepoFile], tag: &str, status: JobStatus, format: &str) {
    if format == "gguf" {
        db_dl::supersede_gguf_groups(&mgr.pool, &files.iter().map(|f| crate::models::split_gguf::destination_key(config::repo_root_dir("a/b").join(&f.rel_path))).collect::<Vec<_>>()).await.unwrap();
    }
    db_dl::insert_group(&mgr.pool, &db_dl::GroupRow { id: gid.to_string(), repo_id: "a/b".into(),
        revision: "main".into(), format: format.into(), task: None, created_at: 0, registration_closed: false }).await.unwrap();
    for f in files {
        let job = Job { id: Uuid::new_v4(), repo_id: "a/b".into(), revision: "main".into(),
            file: f.rel_path.clone(), dest_path: Default::default(), total_bytes: f.size,
            downloaded: f.size, chunks: vec![], status, error: None, expected_sha256: Some(tag.into()),
            group_id: Some(gid), created_at: 0, updated_at: 0 };
        db_dl::upsert_job(&mgr.pool, &job).await.unwrap(); mgr.insert_job(job);
    }
}

async fn closed(mgr: &Manager, gid: Uuid) -> bool {
    db_dl::get_group(&mgr.pool, &gid.to_string()).await.unwrap().unwrap().registration_closed
}

async fn restart(mgr: &Manager, root: &std::path::Path) -> Manager {
    let mut next = Manager::new(mgr.pool.clone(), mgr.http.clone(), 1, mgr.progress_tx.clone());
    next.load_from_db_at(Some(root.into())).await;
    next
}

async fn start(mgr: &mut Manager, files: &[registry::RepoFile]) -> Result<Uuid, AppError> {
    start_repo(mgr, "a/b", files).await
}

async fn start_repo(mgr: &mut Manager, repo: &str, files: &[registry::RepoFile]) -> Result<Uuid, AppError> {
    let (tx, rx) = oneshot::channel();
    handle_command(mgr, Command::StartRepo { repo_id: repo.into(), revision: "main".into(),
        files: files.iter().map(|f| HfFile { sha256: f.sha256.clone(), size: f.size, ..hf(&f.rel_path) }).collect(),
        format: "gguf".into(), task: None, response: tx }, mpsc::channel(1).0).await;
    rx.await.unwrap()
}

#[tokio::test]
async fn split_once_only_chain_survives_history_removal_and_catalog_deletion_in_both_orders() {
    for reverse in [false, true] {
        let mut mgr = manager().await;
        let dir = tempfile::tempdir().unwrap();
        let files = fixture_files(dir.path(), "quant/model");
        let ids = [Uuid::from_u128(u128::MAX), Uuid::from_u128(2), Uuid::from_u128(1)];
        for (gid, tag) in ids.into_iter().zip(["C", "A", "B"]) {
            persist_attempt(&mut mgr, gid, &files, tag, JobStatus::Done, "gguf").await;
        }
        assert!(closed(&mgr, ids[0]).await && closed(&mgr, ids[1]).await);
        assert!(!closed(&mgr, ids[2]).await);
        for gid in if reverse { [ids[2], ids[1], ids[0]] } else { ids } {
            mgr.register_group_at(gid, Some(dir.path().into())).await;
        }
        let model = registry::list(&mgr.pool).await.unwrap().pop().unwrap();
        assert_eq!(model.primary_sha256.as_deref(), Some("B"));
        assert!(closed(&mgr, ids[2]).await);
        // History is retained, including all older generations. Removal cannot
        // promote a survivor, nor can resume/cancel clear its marker.
        assert_eq!(db_dl::list_groups(&mgr.pool).await.unwrap().len(), 3);
        handle_command(&mut mgr, Command::DeleteGroup(ids[2]), mpsc::channel(1).0).await;
        for command in [Command::ResumeGroup(ids[0]), Command::CancelGroup(ids[1])] {
            handle_command(&mut mgr, command, mpsc::channel(1).0).await;
        }
        mgr = restart(&mgr, dir.path()).await;
        assert_eq!(registry::list(&mgr.pool).await.unwrap()[0].primary_sha256.as_deref(), Some("B"));
        crate::db::models::delete_by_id(&mgr.pool, &model.id).await.unwrap();
        mgr = restart(&mgr, dir.path()).await;
        assert!(registry::list(&mgr.pool).await.unwrap().is_empty());
        assert!(files.iter().all(|f| std::path::Path::new(&f.path).is_file()));
        // Retrying after deletion creates a fresh unconsumed owner.
        let retry = start(&mut mgr, &files).await.unwrap();
        for id in mgr.group_member_ids(retry) {
            let job = mgr.jobs.get_mut(&id).unwrap(); job.status = JobStatus::Done;
            db_dl::upsert_job(&mgr.pool, job).await.unwrap();
        }
        mgr = restart(&mgr, dir.path()).await;
        assert_eq!(registry::list(&mgr.pool).await.unwrap().len(), 1);
        assert!(closed(&mgr, retry).await);
    }
}

#[tokio::test]
async fn split_new_attempt_closes_cancelled_and_partial_predecessors_but_not_other_quants() {
    for status in [JobStatus::Done, JobStatus::Cancelled] {
        let mut mgr = manager().await;
        let dir = tempfile::tempdir().unwrap();
        let files = fixture_files(dir.path(), "Q2/model");
        let other_files = fixture_files(dir.path(), "Q4/model");
        let (old, other) = (Uuid::new_v4(), Uuid::new_v4());
        persist_attempt(&mut mgr, old, &files, "old", status, "gguf").await;
        let id = mgr.group_member_ids(old)[0];
        mgr.jobs.get_mut(&id).unwrap().status = JobStatus::Done;
        persist_attempt(&mut mgr, other, &other_files, "other", JobStatus::Done, "gguf").await;
        let new = start(&mut mgr, &files).await.unwrap();
        assert!(closed(&mgr, old).await);
        assert!(!closed(&mgr, other).await && !closed(&mgr, new).await);
        assert_eq!(mgr.group_member_ids(old).len(), 4); // drawer history stays
        assert!(start(&mut mgr, &files).await.is_err()); // active overlap refused
        mgr = restart(&mgr, dir.path()).await;
        assert!(!closed(&mgr, new).await); // queued transfer not registered
        assert_eq!(registry::list(&mgr.pool).await.unwrap().len(), 1);
        handle_command(&mut mgr, Command::CancelGroup(new), mpsc::channel(1).0).await;
        let newer = start(&mut mgr, &files).await.unwrap();
        assert!(closed(&mgr, new).await && !closed(&mgr, newer).await);
    }
}

#[tokio::test]
async fn split_crash_recovery_and_failed_registration_remain_retryable() {
    let mut mgr = manager().await;
    let dir = tempfile::tempdir().unwrap();
    let files = fixture_files(dir.path(), "quant/model");
    let gid = Uuid::new_v4();
    persist_attempt(&mut mgr, gid, &files, "current", JobStatus::Done, "gguf").await;
    // Missing files leave registration retryable. Header parsing deliberately
    // remains best-effort, like single-file GGUF registration.
    let original = std::fs::read(&files[0].path).unwrap();
    std::fs::remove_file(&files[0].path).unwrap();
    mgr = restart(&mgr, dir.path()).await;
    assert!(!closed(&mgr, gid).await && registry::list(&mgr.pool).await.unwrap().is_empty());
    std::fs::write(&files[0].path, original).unwrap();
    // Failure after marker/model writes, before commit, must roll back both.
    // Duplicate relative paths deliberately fail the file-table uniqueness key.
    use crate::db::models::{NewModel, NewModelFile, insert_model_with_files_once};
    let model = NewModel { id: "rollback".into(), repo_id: "a/b".into(), revision: "main".into(),
        display_name: "rollback".into(), architecture: None, capability: "other".into(), format: "gguf".into(),
        task: None, local_path: None, params_b: None, quantization: None, context_train: None, installed_at: 0 };
    let duplicate = || NewModelFile { role: "primary".into(), rel_path: "same".into(),
        path: "same".into(), sha256: None, size: 1 };
    assert!(insert_model_with_files_once(&mgr.pool, &model, &[duplicate(), duplicate()], Some(&gid.to_string())).await.is_err());
    assert!(!closed(&mgr, gid).await && registry::list(&mgr.pool).await.unwrap().is_empty());
    // All members are already Done: a fresh manager heals without redownloading.
    mgr = restart(&mgr, dir.path()).await;
    assert!(closed(&mgr, gid).await);
    let installed = registry::list(&mgr.pool).await.unwrap().pop().unwrap();
    crate::db::models::delete_by_id(&mgr.pool, &installed.id).await.unwrap();
    // There is no committed catalog-without-marker state: both share the commit.
    mgr = restart(&mgr, dir.path()).await;
    assert!(registry::list(&mgr.pool).await.unwrap().is_empty());
    // The database guard also refuses duplicate registration independently of
    // the manager's fast-path check, including after catalog deletion.
    registry::insert_repo_model(&mgr.pool, "a/b", "main", "gguf", None,
        dir.path().to_str().unwrap(), &files, Some(&gid.to_string())).await.unwrap();
    assert!(registry::list(&mgr.pool).await.unwrap().is_empty());
    // Failure writing predecessor markers refuses a new attempt, no new jobs.
    let count = mgr.jobs.len(); mgr.pool.close().await;
    assert!(start(&mut mgr, &files).await.is_err());
    assert_eq!(mgr.jobs.len(), count);
}

#[tokio::test]
async fn split_crash_between_superseding_and_new_group_creation_is_recoverable() {
    let mut mgr = manager().await;
    let dir = tempfile::tempdir().unwrap();
    let files = fixture_files(dir.path(), "quant/model");
    let old = Uuid::new_v4();
    persist_attempt(&mut mgr, old, &files, "old", JobStatus::Done, "gguf").await;
    db_dl::supersede_gguf_groups(&mgr.pool, &files.iter().map(|f| crate::models::split_gguf::destination_key(config::repo_root_dir("a/b").join(&f.rel_path))).collect::<Vec<_>>()).await.unwrap();
    // Simulate exit before creating the replacement group. Old rights stay closed.
    mgr = restart(&mgr, dir.path()).await;
    assert!(closed(&mgr, old).await && registry::list(&mgr.pool).await.unwrap().is_empty());
    assert!(start(&mut mgr, &files).await.is_ok());
}

// Catalog-only claims use real absolute destination spellings, but never
// create/read files in the app's data directory. Fixture I/O stays in TempDir.
async fn claim_paths(mgr: &Manager, files: &[registry::RepoFile], only: Option<usize>, format: &str) {
    use crate::db::models::{NewModel, NewModelFile, insert_model_with_files};
    let model = NewModel { id: "installed".into(), repo_id: "legacy/repo".into(), revision: "old".into(),
        display_name: "installed".into(), architecture: None, capability: "other".into(), format: format.into(),
        task: None, local_path: None, params_b: None, quantization: None, context_train: None, installed_at: 0 };
    let owned: Vec<_> = files.iter().enumerate().filter(|(i, _)| only.is_none_or(|n| *i == n)).map(|(i, f)| NewModelFile {
        role: if only.is_some() || i == 0 { "primary" } else { "weight" }.into(), rel_path: f.rel_path.clone(),
        path: config::repo_root_dir("a/b").join(&f.rel_path).to_string_lossy().replace('\\', "/").to_lowercase(),
        sha256: Some("installed-hash".into()), size: 111,
    }).collect();
    insert_model_with_files(&mgr.pool, &model, &owned).await.unwrap();
}

#[tokio::test]
async fn split_installed_refusal_is_read_only_and_remove_allows_start() {
    for (only, format) in [(None, "gguf"), (Some(2), "gguf"), (Some(2), "safetensors")] {
        for keep_files in [false, true] {
            let mut mgr = manager().await;
            let dir = tempfile::tempdir().unwrap();
            let files = fixture_files(dir.path(), "Q4/É-model");
            let bytes: Vec<_> = files.iter().map(|f| std::fs::read(&f.path).unwrap()).collect();
            let old = Uuid::new_v4();
            persist_attempt(&mut mgr, old, &files, "old", JobStatus::Cancelled, "gguf").await;
            claim_paths(&mgr, &files, only, format).await;
            let jobs = mgr.jobs.len();
            let error = start_repo(&mut mgr, "A/B", &files).await.unwrap_err().to_string();
            assert!(error.contains("already installed") && error.contains("Remove it from Models"), "{error}");
            assert!(!closed(&mgr, old).await);
            assert_eq!(mgr.jobs.len(), jobs);
            assert_eq!(db_dl::list_groups(&mgr.pool).await.unwrap().len(), 1);
            assert_eq!(registry::list(&mgr.pool).await.unwrap()[0].primary_sha256.as_deref(), Some("installed-hash"));
            for (f, before) in files.iter().zip(&bytes) { assert_eq!(&std::fs::read(&f.path).unwrap(), before); }
            // Both Remove outcomes clear ownership. Simulate fixture deletion
            // separately; never delete the catalog-only absolute destinations.
            crate::db::models::delete_by_id(&mgr.pool, "installed").await.unwrap();
            if !keep_files { for f in &files { std::fs::remove_file(&f.path).unwrap(); } }
            let new = start(&mut mgr, &files).await.unwrap();
            assert!(closed(&mgr, old).await && !closed(&mgr, new).await);
            // Drive completion/self-heal with tiny produced GGUFs, no network.
            fixture_files(dir.path(), "Q4/É-model");
            for id in mgr.group_member_ids(new) {
                let job = mgr.jobs.get_mut(&id).unwrap(); job.status = JobStatus::Done;
                db_dl::upsert_job(&mgr.pool, job).await.unwrap();
            }
            mgr = restart(&mgr, dir.path()).await;
            assert_eq!(registry::list(&mgr.pool).await.unwrap().len(), 1);
            assert!(closed(&mgr, new).await);
        }
    }
}

#[tokio::test]
async fn split_aliases_guard_active_jobs_close_predecessors_and_refuse_duplicates() {
    use crate::models::split_gguf::destination_key;
    assert_eq!(destination_key(r"C:\Repo\É\ς.gguf"), destination_key("c:/repo/é/Σ.gguf"));
    assert_eq!(destination_key(config::repo_root_dir("a/b__c").join("x")),
        destination_key(config::repo_root_dir("a__b/c").join("x")));
    let mut mgr = manager().await;
    let dir = tempfile::tempdir().unwrap();
    let files = fixture_files(dir.path(), "Q4/É-model");
    let old = start(&mut mgr, &files).await.unwrap();
    let aliases: Vec<_> = files.iter().map(|f| registry::RepoFile {
        rel_path: f.rel_path.replace('/', "\\").to_lowercase(), path: f.path.clone(), size: f.size, sha256: f.sha256.clone(),
    }).collect();
    assert!(start_repo(&mut mgr, "A/B", &aliases).await.unwrap_err().to_string().contains("already have a download"));
    assert!(!closed(&mgr, old).await);
    handle_command(&mut mgr, Command::CancelGroup(old), mpsc::channel(1).0).await;
    let new = start_repo(&mut mgr, "A/B", &aliases).await.unwrap();
    assert!(closed(&mgr, old).await && !closed(&mgr, new).await);
    assert_eq!(db_dl::list_groups(&mgr.pool).await.unwrap().len(), 2);
    let duplicates: Vec<_> = files.iter().chain([&aliases[0]]).map(|f| registry::RepoFile {
        rel_path: f.rel_path.clone(), path: f.path.clone(), size: f.size, sha256: f.sha256.clone(),
    }).collect();
    assert!(start(&mut mgr, &duplicates).await.unwrap_err().to_string().contains("same file"));
    assert!(!closed(&mgr, new).await);
    // Even a worker that already passed queue checks cannot finalize after closure.
    assert!(db_dl::gguf_write_fence(&mgr.pool, &old.to_string()).await.is_err());
    db_dl::gguf_write_fence(&mgr.pool, &new.to_string()).await.unwrap().commit().await.unwrap();
    assert!(!closed(&mgr, new).await);
}

#[tokio::test]
async fn split_superseded_resume_retry_and_startup_queue_cannot_write() {
    let mut mgr = manager().await;
    let dir = tempfile::tempdir().unwrap();
    let files = fixture_files(dir.path(), "Q4/model");
    let c = start(&mut mgr, &files).await.unwrap();
    handle_command(&mut mgr, Command::CancelGroup(c), mpsc::channel(1).0).await;
    let a = start(&mut mgr, &files).await.unwrap();
    for id in mgr.group_member_ids(a) {
        let job = mgr.jobs.get_mut(&id).unwrap(); job.status = JobStatus::Done;
        db_dl::upsert_job(&mgr.pool, job).await.unwrap();
    }
    mgr.register_group_at(a, Some(dir.path().into())).await;
    claim_paths(&mgr, &files, None, "gguf").await;
    for status in [JobStatus::Paused, JobStatus::Failed] {
        for id in mgr.group_member_ids(c) { mgr.jobs.get_mut(&id).unwrap().status = status; }
        handle_command(&mut mgr, Command::ResumeGroup(c), mpsc::channel(1).0).await;
        assert!(mgr.group_member_ids(c).iter().all(|id| mgr.jobs[id].status == JobStatus::Failed
            && mgr.jobs[id].error.as_deref().unwrap().contains("superseded")));
    }
    for id in mgr.group_member_ids(c) {
        let job = mgr.jobs.get_mut(&id).unwrap(); job.status = JobStatus::Queued;
        db_dl::upsert_job(&mgr.pool, job).await.unwrap();
    }
    mgr = restart(&mgr, dir.path()).await;
    mgr.maybe_start_queued(mpsc::channel(1).0).await;
    assert!(mgr.active.is_empty());
    assert!(closed(&mgr, c).await);
    assert!(mgr.group_member_ids(c).iter().all(|id| mgr.jobs[id].status == JobStatus::Failed));
    let installed = registry::list(&mgr.pool).await.unwrap();
    assert_eq!(installed.len(), 2);
    for model in installed { crate::db::models::delete_by_id(&mgr.pool, &model.id).await.unwrap(); }
    let fresh = start(&mut mgr, &files).await.unwrap(); // closed Failed history must not block
    handle_command(&mut mgr, Command::CancelGroup(fresh), mpsc::channel(1).0).await;
    claim_paths(&mgr, &files, None, "gguf").await;
    // An unclosed manual/legacy attempt must also obey catalog ownership.
    let legacy = Uuid::new_v4();
    persist_attempt(&mut mgr, legacy, &files, "legacy", JobStatus::Paused, "gguf").await;
    handle_command(&mut mgr, Command::ResumeGroup(legacy), mpsc::channel(1).0).await;
    assert!(mgr.group_member_ids(legacy).iter().all(|id| mgr.jobs[id].error.as_deref().unwrap().contains("already installed")));
    assert!(!closed(&mgr, legacy).await);
}

#[tokio::test]
async fn split_upgrade_backfills_only_gguf_and_directory_groups_still_replay() {
    for has_gguf in [false, true] {
        let pool = sqlx::sqlite::SqlitePoolOptions::new().max_connections(1).connect("sqlite::memory:").await.unwrap();
        let mut old = sqlx::migrate!("./migrations");
        old.migrations = old.iter().filter(|m| m.version < 15).cloned().collect::<Vec<_>>().into();
        old.run(&pool).await.unwrap();
        let dir = tempfile::tempdir().unwrap();
        let files = fixture_files(dir.path(), "quant/model");
        let (gguf, directory) = (Uuid::new_v4(), Uuid::new_v4());
        // Pre-upgrade rows use the old INSERT, which has no marker column.
        for (id, format) in [(directory, "safetensors"), (gguf, "gguf")] {
            if format == "gguf" && !has_gguf { continue; }
            db_dl::insert_group(&pool, &db_dl::GroupRow { id: id.to_string(), repo_id: "a/b".into(), revision: "main".into(),
                format: format.into(), task: None, created_at: 0, registration_closed: false }).await.unwrap();
        }
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        let mut mgr = Manager::new(pool, reqwest::Client::new(), 1, broadcast::channel(16).0);
        if has_gguf { assert!(closed(&mgr, gguf).await); }
        assert!(!closed(&mgr, directory).await);
        // Seed member jobs for the existing directory group and replay twice,
        // including after catalog deletion with every file still present.
        for f in &files {
            let job = Job { id: Uuid::new_v4(), repo_id: "a/b".into(), revision: "main".into(), file: f.rel_path.clone(),
                dest_path: Default::default(), total_bytes: f.size, downloaded: f.size, chunks: vec![], status: JobStatus::Done,
                error: None, expected_sha256: None, group_id: Some(directory), created_at: 0, updated_at: 0 };
            db_dl::upsert_job(&mgr.pool, &job).await.unwrap();
        }
        mgr = restart(&mgr, dir.path()).await;
        let model = registry::list(&mgr.pool).await.unwrap().pop().unwrap();
        assert_eq!(model.id, registry::repo_model_id("a/b", "main"));
        assert!(!closed(&mgr, directory).await);
        crate::db::models::delete_by_id(&mgr.pool, &model.id).await.unwrap();
        mgr = restart(&mgr, dir.path()).await;
        assert_eq!(registry::list(&mgr.pool).await.unwrap().len(), 1);
    }
}
