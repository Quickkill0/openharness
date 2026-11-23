-- M10 — shared asset library: one row per generated artifact (image, video,
-- audio clip, transcript file). Rows are written when a generation job
-- completes and its outputs are promoted out of assets\staging\ into the
-- per-kind library directories. `job_id` is a soft reference — deleting the
-- job row keeps the asset.
CREATE TABLE assets (
  id          TEXT PRIMARY KEY,
  job_id      TEXT,
  kind        TEXT NOT NULL,        -- image | video | audio | text
  path        TEXT NOT NULL UNIQUE, -- absolute path inside the assets dir
  task        TEXT NOT NULL,        -- source pipeline task
  prompt      TEXT,                 -- prompt / spoken text, when the job had one
  model       TEXT,                 -- human-readable model label
  params      TEXT,                 -- JSON request params
  meta        TEXT,                 -- JSON worker meta (seed, duration, ...)
  size        INTEGER NOT NULL,     -- bytes on disk at registration time
  created_at  INTEGER NOT NULL
);

CREATE INDEX idx_assets_created ON assets(created_at DESC);
CREATE INDEX idx_assets_kind ON assets(kind, created_at DESC);

-- FTS over the searchable text. Synced app-level in db/assets.rs (same
-- strategy as messages_fts): insert and delete touch both tables.
CREATE VIRTUAL TABLE assets_fts USING fts5(
  asset_id UNINDEXED,
  prompt, model, task,
  tokenize = 'porter unicode61'
);
