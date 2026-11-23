-- Per-project "approve always" persistence for built-in agent tools (M7 open fold).
-- The 0001 shell table keyed rows by agent profile; the approval gate is scoped to
-- WHERE the tool runs (the project), so rows gain a project_id. The legacy agent_id
-- column stays (SQLite can't drop it cheaply) but is unused.
ALTER TABLE tool_allowlists ADD COLUMN project_id TEXT REFERENCES projects(id) ON DELETE CASCADE;

-- One decision per (project, tool). NULL project_ids (never written by the app)
-- are distinct under SQLite's unique-index semantics, which is fine.
CREATE UNIQUE INDEX idx_tool_allowlists_project_tool ON tool_allowlists(project_id, tool_name);
