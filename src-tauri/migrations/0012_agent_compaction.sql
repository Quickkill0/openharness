-- 0012: per-profile auto-compaction tuning (M7 runner constants made configurable).
-- NULL = use the built-in defaults (ratio 0.8, keep 6 recent entries).
ALTER TABLE agents ADD COLUMN compact_ratio REAL;
ALTER TABLE agents ADD COLUMN keep_recent INTEGER;
