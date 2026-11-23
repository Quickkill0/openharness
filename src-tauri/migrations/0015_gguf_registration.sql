-- GGUF groups register once; superseded groups must never replay.
ALTER TABLE download_groups ADD COLUMN registration_closed INTEGER NOT NULL DEFAULT 0;
-- No Store GGUF groups existed before this feature. Conservatively close any
-- pre-migration rows from development builds; catalog rows/files stay intact.
UPDATE download_groups SET registration_closed = 1 WHERE format = 'gguf';
