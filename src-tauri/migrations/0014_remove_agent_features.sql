-- M17 direction (2026-11-07): agentic work moves to AI Shuffle.
-- Permanently remove agent runs (including descendants), profiles, MCP and projects.
-- Ordinary chat rows, their transcript paths, settings, indexes and FTS survive.
CREATE TEMP TABLE removed_agent_conversations AS
WITH RECURSIVE runs(id) AS (
    SELECT id FROM conversations WHERE agent_id IS NOT NULL OR fluid_session_id IS NOT NULL
    UNION
    SELECT c.id FROM conversations c JOIN runs r ON c.parent_conversation_id = r.id
)
SELECT id FROM runs;

-- FTS has no foreign key. messages_index and conversation_settings cascade.
DELETE FROM messages_fts
WHERE conversation_id IN (SELECT id FROM removed_agent_conversations);
DELETE FROM conversations
WHERE id IN (SELECT id FROM removed_agent_conversations);
DROP TABLE removed_agent_conversations;

-- Drop foreign keys before the referenced agent-only tables. No table rebuild:
-- rebuilding conversations with foreign_keys=ON would cascade into chat data.
DROP INDEX idx_conversations_parent;
ALTER TABLE conversations DROP COLUMN parent_conversation_id;
ALTER TABLE conversations DROP COLUMN agent_id;
ALTER TABLE conversations DROP COLUMN project_id;
ALTER TABLE conversations DROP COLUMN working_dir;
ALTER TABLE conversations DROP COLUMN mode;
ALTER TABLE conversations DROP COLUMN fluid_session_id;

DROP TABLE tool_allowlists;
DROP TABLE mcp_tool_consents;
DROP TABLE mcp_servers;
DROP TABLE projects;
DROP TABLE agents;
DELETE FROM settings WHERE key IN ('fluid_tool_style', 'fluid_provider_extra');
