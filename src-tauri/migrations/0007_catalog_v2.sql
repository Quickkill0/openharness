-- M9 §9.1 — Catalog v2: multi-modal formats + multi-file models.
--
-- `models` learns what *kind* of artifact it holds:
--   format     — 'gguf' | 'safetensors' | 'diffusers-dir'
--   task       — HF pipeline_tag-style string ('text-generation', 'text-to-image', …)
--   local_path — root directory for dir-layout models (GGUF blobs keep using
--                the primary model_files row; this stays NULL for them)
ALTER TABLE models ADD COLUMN format TEXT NOT NULL DEFAULT 'gguf';
ALTER TABLE models ADD COLUMN task TEXT;
ALTER TABLE models ADD COLUMN local_path TEXT;

-- `model_files` becomes a true multi-file table:
--   * rel_path (path inside the repo) added; uniqueness moves from
--     (model_id, role) to (model_id, rel_path) so a model can have many
--     'weight'/'config' rows.
--   * sha256 becomes nullable — non-LFS repo files (configs, tokenizers)
--     carry no LFS digest.
-- SQLite cannot alter constraints in place: rebuild the table.
CREATE TABLE model_files_new (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id  TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  role      TEXT NOT NULL,
  rel_path  TEXT NOT NULL,
  path      TEXT NOT NULL,
  sha256    TEXT,
  size      INTEGER NOT NULL,
  UNIQUE(model_id, rel_path)
);

-- Legacy rows are all single-GGUF primaries whose on-disk name is <model_id>.gguf.
INSERT INTO model_files_new (id, model_id, role, rel_path, path, sha256, size)
  SELECT id, model_id, role, model_id || '.gguf', path, sha256, size
  FROM model_files;

DROP TABLE model_files;
ALTER TABLE model_files_new RENAME TO model_files;
CREATE INDEX idx_model_files_model ON model_files(model_id);
