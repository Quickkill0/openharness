-- M7.5 — projects: first-class named working directories.
-- A profile describes *how* an agent works; a project describes *where*.
-- Projects carry the durable identity of a directory (run history grouping,
-- future git surfacing and per-project allowlists).
CREATE TABLE projects (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  path         TEXT NOT NULL UNIQUE,
  created_at   INTEGER NOT NULL,
  last_used_at INTEGER NOT NULL
);

-- Runs belong to a project. The existing working_dir column stays as the
-- resolved-path snapshot so old runs remain readable if a project row is
-- deleted or its path later changes.
ALTER TABLE conversations ADD COLUMN project_id TEXT REFERENCES projects(id) ON DELETE SET NULL;

-- Working directories now live on projects; profiles are portable across them.
ALTER TABLE agents DROP COLUMN working_dir;
