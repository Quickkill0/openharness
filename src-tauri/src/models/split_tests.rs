use super::*;

async fn pool() -> SqlitePool {
    let pool = sqlx::sqlite::SqlitePoolOptions::new().max_connections(1)
        .connect("sqlite::memory:").await.unwrap();
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    pool
}

pub(crate) fn fixture_files(dir: &Path, stem: &str) -> Vec<RepoFile> {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gguf/glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf");
    super::super::gguf::split_files(&fixture).unwrap().iter().enumerate().map(|(i, src)| {
        let rel_path = format!("{stem}-{:05}-of-00004.gguf", i + 1);
        let path = dir.join(&rel_path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::copy(src, &path).unwrap();
        RepoFile { rel_path, size: std::fs::metadata(&path).unwrap().len(),
            path: path.to_string_lossy().into_owned(), sha256: None }
    }).collect()
}

#[test]
fn split_set_ids_include_quant_and_subfolder_without_changing_repo_ids() {
    assert_ne!(gguf_set_model_id("a/b", "main", "Q2/m-Q2"), gguf_set_model_id("a/b", "main", "Q4/m-Q4"));
    assert_ne!(gguf_set_model_id("a/b", "main", "one/m"), gguf_set_model_id("a/b", "main", "two/m"));
    use sha2::{Digest, Sha256};
    assert_eq!(repo_model_id("a/b", "main"), format!("{:x}", Sha256::digest(b"a/b@main")));
}

#[tokio::test]
async fn split_registration_metadata_primary_and_file_scoped_deletion() {
    let pool = pool().await;
    let dir = tempfile::tempdir().unwrap();
    let mut a = fixture_files(dir.path(), "quants/GLM-UD-Q2");
    let b = fixture_files(dir.path(), "quants/GLM-UD-Q4");
    let expected = super::super::gguf::parse_header(Path::new(&a[0].path)).unwrap();
    assert!(expected.architecture.is_some());
    assert!(expected.context_length.is_some());
    a.reverse(); // Manager members come from a HashMap; primary cannot depend on order.
    let root = dir.path().to_str().unwrap();
    let id = insert_repo_model(&pool, "unsloth/GLM", "main", "gguf", None, root, &a, None).await.unwrap();
    let other = insert_repo_model(&pool, "unsloth/GLM", "main", "gguf", None, root, &b, None).await.unwrap();
    let model = get_by_id(&pool, &id).await.unwrap().unwrap();
    assert!(model.primary_file_path.ends_with("-00001-of-00004.gguf"));
    assert_eq!(model.architecture, expected.architecture);
    assert_eq!(model.params_b, expected.params_b);
    assert_eq!(model.quantization, expected.quantization);
    assert_eq!(model.context_train, expected.context_length);
    assert_eq!(model.file_count, 4);
    assert_eq!(model.total_size, a.iter().map(|f| f.size).sum::<u64>());
    assert_eq!(model.display_name, "GLM-UD-Q2");
    assert_eq!(model.capability, Capability::Other);
    assert_eq!(model.task, None);
    assert_eq!(model.weight_size(), model.total_size);
    assert!(model.caps.workflows.contains(&super::super::capabilities::Workflow::Chat));
    assert!(model.local_path.is_none());
    assert_eq!(find_by_source_path(&pool, &model.primary_file_path).await.unwrap().unwrap().id, id);
    assert!(find_by_source_path(&pool, root).await.unwrap().is_none());
    scan_dir_in(&pool, dir.path()).await.unwrap();
    assert_eq!(list(&pool).await.unwrap().len(), 2);
    delete_in(&pool, &id, dir.path()).await.unwrap();
    assert!(a.iter().all(|f| !Path::new(&f.path).exists()));
    assert!(b.iter().all(|f| Path::new(&f.path).exists()));
    assert!(get_by_id(&pool, &other).await.unwrap().is_some());
    std::fs::remove_file(&b[1].path).unwrap();
    scan_dir_in(&pool, dir.path()).await.unwrap();
    assert!(get_by_id(&pool, &other).await.unwrap().is_none());
}

#[test]
fn split_validation_rejects_partial_duplicate_and_mixed_sets() {
    use super::super::split_gguf::validate;
    assert!(validate(["a-00001-of-00002.gguf"]).is_err());
    assert!(validate(["a-00001-of-00002.gguf", "a-00001-of-00002.gguf"]).is_err());
    assert!(validate(["a-00001-of-00002.gguf", "b-00002-of-00002.gguf"]).is_err());
    assert!(validate(["a-00001-of-00002.gguf", "a-00002-of-00002.gguf", "mmproj.gguf"]).is_err());
    assert!(validate(["a-00001-of-00002.gguf", "a-00002-of-00002.gguf_file"]).is_err());
}

#[tokio::test]
async fn split_capabilities_follow_task_or_header_and_sizes_preserve_other_formats() {
    use crate::models::capabilities::{RuntimeKind, Workflow};
    let pool = pool().await;
    let dir = tempfile::tempdir().unwrap();
    for (arch, task, workflow) in [
        ("glm5-next", None, Some(Workflow::Chat)),
        ("glm5-next", Some("text-generation"), Some(Workflow::Chat)),
        ("bert", None, Some(Workflow::Embeddings)),
        ("bert", Some("feature-extraction"), Some(Workflow::Embeddings)),
        ("qwen_image", None, Some(Workflow::Image)),
        // Keep the existing explicit-image-task refusal; never mislabel it as Chat.
        ("qwen_image", Some("text-to-image"), None),
    ] {
        let files = fixture_files(dir.path(), &format!("{arch}/{}", task.unwrap_or("no-task")));
        let mut header = Vec::new();
        fn string(out: &mut Vec<u8>, s: &str) {
            out.extend((s.len() as u64).to_le_bytes()); out.extend(s.as_bytes());
        }
        header.extend(b"GGUF"); header.extend(3u32.to_le_bytes());
        header.extend(0u64.to_le_bytes()); header.extend(1u64.to_le_bytes());
        string(&mut header, "general.architecture"); header.extend(8u32.to_le_bytes());
        string(&mut header, arch);
        std::fs::write(&files[0].path, header).unwrap();
        let id = insert_repo_model(&pool, "test/models", "main", "gguf", task,
            dir.path().to_str().unwrap(), &files, None).await.unwrap();
        let mut model = get_by_id(&pool, &id).await.unwrap().unwrap();
        assert_eq!(model.task.as_deref(), task);
        assert_eq!(model.caps.workflows, workflow.into_iter().collect::<Vec<_>>());
        if arch == "qwen_image" && task.is_none() {
            assert_eq!(model.caps.runtime, Some(RuntimeKind::Worker));
        }
        let total = 109_000_000_000;
        model.total_size = total;
        model.primary_file_size = 9 * 1024 * 1024;
        assert_eq!(model.weight_size(), total);
        // Legacy explicit-ngl gate/ledger and reclaim inputs must price all parts.
        let meta = super::super::gguf::parse_header(Path::new(&model.primary_file_path)).unwrap();
        let inputs = crate::engine::fit::FitInputs::from_meta(&meta, model.weight_size() / (1024 * 1024), 4096, 2);
        assert_eq!(inputs.file_size_mb, total / (1024 * 1024));
        model.format = "safetensors".into();
        assert_eq!(model.weight_size(), model.primary_file_size);
        model.format = "diffusers-dir".into();
        assert_eq!(model.weight_size(), model.primary_file_size);
        model.format = "gguf".into();
        model.primary_file_path = "content-hash.gguf".into();
        assert_eq!(model.weight_size(), model.primary_file_size);
    }
}
