use super::*;

#[test]
fn split_gguf_pipeline_estimate_counts_all_parts_but_single_files_and_dirs_are_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let files = crate::models::registry::split_tests::fixture_files(dir.path(), "quant/image");
    for f in &files { std::fs::OpenOptions::new().write(true).open(&f.path).unwrap().set_len(2 * 1024 * 1024).unwrap(); }
    let primary = Path::new(&files[0].path);
    assert_eq!(path_size_mb(primary), 8);
    assert_eq!(estimate_task_vram_mb("text_to_image", &serde_json::json!({"model": primary})), 8 + PIPELINE_OVERHEAD_MB);
    let single = dir.path().join("single.gguf");
    std::fs::write(&single, vec![0; 1024 * 1024]).unwrap();
    assert_eq!(path_size_mb(&single), 1);
    assert_eq!(path_size_mb(primary.parent().unwrap()), 8);
}
