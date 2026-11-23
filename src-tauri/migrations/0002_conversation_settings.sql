-- M6 §6.4 — per-conversation chat configuration.
-- System prompt, sampling parameters, and load-time config are stored per
-- conversation as JSON blobs. Absence of a row means "use global defaults".
CREATE TABLE conversation_settings (
  conversation_id  TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
  system_prompt    TEXT,
  sampling_json    TEXT,
  load_config_json TEXT,
  updated_at       INTEGER NOT NULL
);
