-- M8 §8.5 — generation jobs: one row per Python-worker pipeline job
-- (text_to_image / text_to_speech / transcribe / text_to_video).
-- Outputs land in assets\staging\ and are recorded here as absolute paths;
-- M10's asset library promotes them into the curated library.
CREATE TABLE generation_jobs (
  id          TEXT PRIMARY KEY,             -- job_id, generated Rust-side pre-submit
  task        TEXT NOT NULL,                -- pipeline task name
  status      TEXT NOT NULL,                -- running | done | error | cancelled
  params      TEXT NOT NULL,                -- JSON request params (prompt, model, ...)
  outputs     TEXT,                         -- JSON array of absolute output paths
  meta        TEXT,                         -- JSON worker meta (seed, vram_mb, duration, ...)
  error       TEXT,
  created_at  INTEGER NOT NULL,
  finished_at INTEGER
);

CREATE INDEX idx_generation_jobs_created ON generation_jobs(created_at DESC);
