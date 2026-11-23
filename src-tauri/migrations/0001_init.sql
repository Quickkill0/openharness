-- Pre-release schema — may be edited until first public release.
-- After release, schema changes must be added as new migration files.

-- Settings (k/v)
CREATE TABLE settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Agents (shell)
CREATE TABLE agents (
  id             TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  system_prompt  TEXT NOT NULL,
  default_model  TEXT,
  default_mode   TEXT NOT NULL DEFAULT 'auto',
  tools_json     TEXT NOT NULL DEFAULT '[]',
  created_at     INTEGER NOT NULL
);

-- Models catalog (shell)
CREATE TABLE models (
  id              TEXT PRIMARY KEY,
  repo_id         TEXT NOT NULL,
  revision        TEXT NOT NULL,
  display_name    TEXT NOT NULL,
  architecture    TEXT,
  capability      TEXT NOT NULL,
  params_b        REAL,
  quantization    TEXT,
  context_train   INTEGER,
  installed_at    INTEGER NOT NULL,
  last_used_at    INTEGER,
  pinned          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_models_capability ON models(capability);

CREATE TABLE model_files (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id    TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  role        TEXT NOT NULL,
  path        TEXT NOT NULL,
  sha256      TEXT NOT NULL,
  size        INTEGER NOT NULL,
  UNIQUE(model_id, role)
);

-- Downloads (shell)
CREATE TABLE downloads (
  id               TEXT PRIMARY KEY,
  repo_id          TEXT NOT NULL,
  revision         TEXT NOT NULL,
  file             TEXT NOT NULL,
  dest_path        TEXT NOT NULL,
  total_bytes      INTEGER NOT NULL,
  downloaded       INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL,
  error            TEXT,
  chunks_json      TEXT NOT NULL DEFAULT '[]',
  expected_sha256  TEXT,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);

-- Engines (llama.cpp pinned releases) (shell)
CREATE TABLE engines (
  id           TEXT PRIMARY KEY,
  version      TEXT NOT NULL,
  variant      TEXT NOT NULL,
  path         TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  active       INTEGER NOT NULL DEFAULT 0,
  installed_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX idx_engines_active ON engines(active) WHERE active = 1;

-- Conversations (shell)
CREATE TABLE conversations (
  id            TEXT PRIMARY KEY,
  title         TEXT,
  agent_id      TEXT REFERENCES agents(id),
  model_id      TEXT,
  transcript    TEXT NOT NULL,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

CREATE TABLE messages_index (
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  seq             INTEGER NOT NULL,
  byte_offset     INTEGER NOT NULL,
  byte_len        INTEGER NOT NULL,
  role            TEXT NOT NULL,
  has_tool_use    INTEGER NOT NULL DEFAULT 0,
  ts              INTEGER NOT NULL,
  PRIMARY KEY (conversation_id, seq)
);
CREATE INDEX idx_messages_index_conv_role ON messages_index(conversation_id, role);

-- FTS5 over messages text.
-- M5 (chat module) is responsible for keeping this table in sync with messages_index,
-- either via app-level inserts on transcript writes or via SQLite triggers added in a
-- later migration. This is an explicit hand-off: the schema owns the table; M5 owns
-- the sync strategy.
CREATE VIRTUAL TABLE messages_fts USING fts5(
  conversation_id UNINDEXED,
  seq UNINDEXED,
  role UNINDEXED,
  text,
  tokenize = 'porter unicode61'
);

-- Tools / allowlists (shell)
CREATE TABLE tool_allowlists (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id    TEXT REFERENCES agents(id),
  tool_name   TEXT NOT NULL,
  arg_pattern TEXT NOT NULL,
  decision    TEXT NOT NULL,
  created_at  INTEGER NOT NULL
);

-- MCP (shell)
CREATE TABLE mcp_servers (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  transport   TEXT NOT NULL,
  command     TEXT,
  args_json   TEXT,
  url         TEXT,
  env_json    TEXT,
  enabled     INTEGER NOT NULL DEFAULT 1,
  source      TEXT NOT NULL,
  created_at  INTEGER NOT NULL
);

CREATE TABLE mcp_tool_consents (
  server_id   TEXT NOT NULL REFERENCES mcp_servers(id) ON DELETE CASCADE,
  tool_name   TEXT NOT NULL,
  decision    TEXT NOT NULL,
  PRIMARY KEY (server_id, tool_name)
);
