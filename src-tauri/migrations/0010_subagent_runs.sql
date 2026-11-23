-- M12: sub-agent orchestrator.
-- A sub-agent run is a normal agent conversation linked to the run that
-- spawned it. Deleting the parent cascades to its children (transcript files
-- are removed by the app's delete path, which collects child transcripts).
ALTER TABLE conversations
    ADD COLUMN parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE;

CREATE INDEX idx_conversations_parent ON conversations(parent_conversation_id);
