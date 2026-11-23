-- M7 §7.3 — agent runs as first-class conversations.
-- A run is a `conversations` row with `agent_id` set (column exists since 0001).
-- Add the run-scoped fields: the working directory the tools resolved against
-- and the execution mode the run was launched with (plan | auto | yolo).
ALTER TABLE conversations ADD COLUMN working_dir TEXT;
ALTER TABLE conversations ADD COLUMN mode TEXT;
