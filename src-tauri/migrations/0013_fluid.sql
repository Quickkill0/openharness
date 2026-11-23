-- Fluid conversion — agent runs move onto the fluid harness sidecar.
--
-- `conversations.fluid_session_id` links a run's metadata row to its fluid
-- JsonlStore session (`%LOCALAPPDATA%\OpenHarness\fluid-sessions\<sid>.jsonl`),
-- which is the source of truth for run content. NULL for chat conversations
-- and for pre-conversion agent runs (shown as "recorded under the previous
-- harness").
--
-- `agents.code_mode` is the profile-level toggle for fluid's code-as-actions
-- execution (run_python cells instead of direct tool calls).

ALTER TABLE conversations ADD COLUMN fluid_session_id TEXT;
ALTER TABLE agents ADD COLUMN code_mode INTEGER NOT NULL DEFAULT 0;
