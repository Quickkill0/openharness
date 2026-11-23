-- M7 §7.2 — agent profiles.
-- Extend the M1 `agents` shell into a full profile: visual identity, a default
-- working directory, per-profile sampling + load config (reusing the same JSON
-- shapes as conversation_settings), a loop step budget, and a built-in flag so
-- curated starter agents can be reset to their code-defined defaults.
--
-- Existing columns (from 0001): id, name, system_prompt, default_model,
-- default_mode, tools_json, created_at. `tools_json` is the allowed-tool list;
-- an empty array (the default) OR `["*"]` means "all tools".

ALTER TABLE agents ADD COLUMN description      TEXT;
ALTER TABLE agents ADD COLUMN icon             TEXT;
ALTER TABLE agents ADD COLUMN color            TEXT;
ALTER TABLE agents ADD COLUMN working_dir      TEXT;
ALTER TABLE agents ADD COLUMN sampling_json    TEXT;
ALTER TABLE agents ADD COLUMN load_config_json TEXT;
ALTER TABLE agents ADD COLUMN max_iterations   INTEGER;
ALTER TABLE agents ADD COLUMN is_builtin       INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agents ADD COLUMN updated_at       INTEGER NOT NULL DEFAULT 0;
