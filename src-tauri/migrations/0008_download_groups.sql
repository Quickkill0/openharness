-- M9 §9.2 — Downloader v2: multi-file repo downloads.
--
-- A download *group* is one user action ("download this repo") fanned out
-- into one job per file. Group metadata lives here; member jobs reference it
-- via downloads.group_id. Legacy single-GGUF jobs keep group_id NULL and
-- behave exactly as before.
CREATE TABLE download_groups (
  id         TEXT PRIMARY KEY,
  repo_id    TEXT NOT NULL,
  revision   TEXT NOT NULL,
  format     TEXT NOT NULL,    -- 'safetensors' | 'diffusers-dir' (gguf stays single-job)
  task       TEXT,             -- HF pipeline_tag-style task for catalog registration
  created_at INTEGER NOT NULL
);

ALTER TABLE downloads ADD COLUMN group_id TEXT REFERENCES download_groups(id);
CREATE INDEX idx_downloads_group ON downloads(group_id);
