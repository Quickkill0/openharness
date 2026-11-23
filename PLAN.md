# OpenHarness — Plan & Roadmap

> Local-first Windows desktop AI **workspace**: agent harness, multi-modal Hugging Face
> store, generation Studio (image / speech / transcription), chat with web tools — on a
> llama.cpp + Python-worker runtime layer, all offline.
> (Formerly "Hangar AI"; see **Milestone 6 → Rebrand**.)

This is the working plan of record. The terse milestone list lives in `CLAUDE.md`;
this file is the detailed, ordered plan. Status legend: ✅ done · 🔜 next · ⬜ planned.

---

## M17 direction (2026-11-07)

OpenHarness is a local model loader/runner plus Studio (image, video, speech and
transcription). Plain local-model chat and voice stay. Agentic work belongs in
AI Shuffle, which will connect over an OpenAI-compatible API in a later PR.
This PR is step 1: remove agents, Fluid, tools, MCP, projects and chat web search.
Migration 0014 permanently removes their database data; ordinary chats and FTS stay.
The roadmap below records the earlier direction, not the current feature set.

## Product vision (historical)

OpenHarness is a **local-first AI workspace**: one Windows desktop app where a user can browse
all of Hugging Face, download a model, and *use* it — for agents, chat, speech, transcription,
and image generation — end to end, entirely offline. **Four co-equal pillars** (decided with
the user 2026-06-09, broadening the earlier same-day "agents are the product" framing):

1. **Store** — browse/search ALL of HF by task (text-gen, TTS, STT, text-to-image, …) with
   "runs in OpenHarness" badges, curated VRAM-aware picks per modality; raw browse stays.
2. **Studio** — generate AI content locally: task pages (Image / Speech / Transcribe; video
   is post-v1) + a shared, searchable asset library.
3. **Agents** — the autonomous agent harness (M7): profiles, runs-as-conversations, full
   context stack, approval gate. "Workflows" = profile-based agent runs (no node editor).
4. **Chat** — general chat with web search/fetch tools and voice mode (STT in / TTS out).

Principles that shape the roadmap:
- **Two engine families, ever.** llama.cpp for text (GGUF) + ONE OpenHarness-managed **Python
  worker** sidecar (uv-managed pinned venv, pin-and-fetch like the engine, torch CUDA/CPU wheel
  by GPU detection, ~4–6 GB one-time) covering image/TTS/STT/(video later) via
  diffusers / faster-whisper / TTS libs. Per-modality C++ sidecars (sd.cpp, whisper.cpp, ONNX
  TTS) were rejected 2026-06-09: three more upstreams to track for *less* model coverage.
- **Cross-pillar composition is the differentiator.** Generation/transcription are tools:
  agents can call generate_image / tts / transcribe mid-run; chat speaks and listens.
- **VRAM orchestration: concurrent-when-fits.** Models stay resident when the fit estimator
  says they fit; when a requested load doesn't fit, prompt the user with a choice of what to
  unload — no silent eviction.
- **Adaptive hardware target.** No fixed VRAM baseline — every recommendation and filter
  derives from the detected GPU via the fit estimator.
- **Profile-first.** Users define named agent profiles (system prompt, model, tools, mode, working
  dir, sampling, load config, icon) and launch runs from them. Ship curated, editable starters.
- **Own the full context stack.** A run is a persisted conversation capturing every tool call, tool
  result, assistant message, and thinking block — with a **context viewer** and **auto-compaction**
  so long runs don't dead-end on the model's context window.
- **Model-agnostic + local.** ReAct text protocol (no native tool-calling dependency); unrestricted
  user-selected working dir (safety via the approval gate, not a path jail); zero telemetry.
- **Smart defaults, expert escape hatch.** One-click model loading (auto-`-ngl` from VRAM fit) with
  full knobs behind "Advanced"; a curated, VRAM-aware HuggingFace store over the raw browser.

---

## Status at a glance

| # | Milestone | Status |
|---|-----------|--------|
| M1 | Scaffold (Tauri + Svelte + sqlx shell, full schema, route placeholders) | ✅ |
| M2 | HuggingFace model browser (read-only) | ✅ |
| M3 | Downloader actor (resumable, chunked, GGUF parse, catalog) | ✅ |
| M4 | Engine module (pin-and-fetch llama.cpp), sidecar, NVML, VRAM fit | ✅ |
| M5 | Basic chat (single model, streaming, JSONL transcripts, FTS5) | ✅ |
| **M6** | **Rebrand → OpenHarness + UI/UX overhaul + model/chat config** | **✅** (§6.0–6.4; folds closed in PR #11) |
| **M7** | **Agent harness** (loop + tools + approval gate + Plan/Auto/Yolo + **profiles** + full context stack + compaction + smart loading + **projects**) | **✅** (§7.1–7.7, 2026-06-09) |
| **M8** | **Runtime v2: Python worker sidecar + multi-engine VRAM orchestrator + generation primitives** | **✅** (§8.1–8.6, 2026-06-10, PR #5) |
| M9 | Multi-modal store: browse-by-task, repo downloads, runnable badges, curated picks | ✅ (§9.1–9.6, 2026-06-10, PR #6) |
| M10 | Studio: Image / **Video** / Speech / Transcribe task pages + shared asset library | ✅ (§10, 2026-06-10, PR #7) |
| **M11** | **Cross-pillar: agent generation tools, chat voice mode, web_search tool** | **✅** (§11, 2026-06-10) |
| M12 | Sub-agent orchestrator (spawn scoped sub-agents, collect results) | ✅ (§12, 2026-06-10) |
| M13 | MCP client (rmcp): stdio + HTTP, sampling → local model, registry | ✅ (§13, 2026-06-10, PR #10) |
| M14 | Updater, code-signing, signed MSI, first-run wizard, onboarding | ⬜ *(deferred — needs user-supplied signing cert)* |
| **M15** | **Model-first capability layer: per-model caps records, "Use with…" routing, capability-driven Studio forms** | **✅** (§15 phases 1–3, 2026-06-12) |
| **M16** | **UX redesign: load-first models, chat → claude.ai feel, agents → Claude Code feel, adaptive Studio, friendly store, engine-registry settings** | **✅** (§16 W1–W6, 2026-06-12) |

> **Renumbering note (multi-modal re-plan, 2026-06-09):** the four-pillar workspace vision
> (see above) inserted **M8–M11** (runtime v2 / store / Studio / cross-pillar) after M7;
> sub-agents → **M12**, MCP → **M13**, release → **M14**. Release can be pulled earlier if
> needed. Earlier shifts (M6 rebrand insert; profiles pulled into M7) still apply.

---

## Handoff — current state (for the next agent)

**As of 2026-06-10**, M1–M13 are complete and merged to `main` (M7/M7.5 via PRs #2–#4, M8 via
PR #5, M9 via PR #6, M10 via PR #7, M11 via PR #8, M12 via PR #9, M13 via PR #10). The
autonomous build-out scope (M8–M13) is **done** — what remains is the open-folds backlog
below and the deferred M14 release work.

> **2026-06-10 autonomy handoff:** the user granted full autonomous control to complete the app
> through **M8–M13**, ending at "ready for the user's final design/fix pass". Decisions locked:
> **M14 deferred** (code-signing cert is user-supplied); **video generation pulled INTO v1**
> (text-to-video pipeline in the M8 worker + a Video task page in M10 Studio, 24 GB-class models
> e.g. LTX-Video / Wan); git flow = **PR per milestone, self-merged** once tests + e2e pass;
> model/wheel downloads pre-approved at any size; **check in with the user before GPU-heavy
> inference sessions** (shared machine). Dev hardware: RTX 4090 24 GB, ~265 GB free disk.

**Build & verify:**
- Frontend: `pnpm check` (svelte-check, must be 0 errors) and `pnpm build` (also compiles Tailwind).
- Backend: `cargo test --manifest-path src-tauri/Cargo.toml --lib` (50 tests) and
  `cargo check --manifest-path src-tauri/Cargo.toml`.
- Full app (only way to exercise the live chat workspace / engine): `pnpm tauri dev`.
- Design-system / static UI without Tauri: `pnpm dev` → open `http://localhost:1420/settings/appearance`
  in a browser (the kitchen-sink renders every primitive; `invoke`-backed data fails in plain Chrome,
  which is expected — engine/chat show their gates).

**sqlx discipline (important):** `SQLX_OFFLINE=true` is set in `src-tauri/.cargo/config.toml`, so
`query!`/`query_as!` macros validate against the committed `.sqlx/` cache, not a live DB. After adding
or changing any `query!`, regenerate the cache: install `sqlx-cli` (already installed:
`cargo install sqlx-cli --no-default-features --features sqlite,rustls`), point `DATABASE_URL` at a
freshly-migrated SQLite file, then `cargo sqlx prepare` from `src-tauri/`. LEFT-JOIN columns use
force-nullable annotations (`AS "col?"`) so the cache is sqlx-version-independent.

**Open folds carried out of M6** — closed in PR #11 (2026-06-10) except where noted:
- **Chat regenerate / delete-from-here** — ✅ `truncate_from` (JSONL rewrite + index/FTS prune),
  `TruncateFrom`/`Regenerate` chat-actor commands + `truncated` event, per-message hover buttons.
- **Engine logs viewer** (`/settings/engine`) — ✅ `EngineCmd::Logs` + `engine_logs` command + viewer card.
- **Orphan engine rows** — ✅ `engines::prune_inactive` at actor startup and after installs (kept
  random ids; pruning makes the stable-id rework unnecessary).
- **Custom (user-saved) system-prompt presets** — still open (post-v1 nice-to-have).

See also `CLAUDE.md` → "Tracked debt" for cross-cutting items (dev/prod CSP split before release, SHA-NI
release flag, engine ZIP checksum pinning, etc.).

---

## What's done (M1–M5)

- **M1** — Tauri 2 + SvelteKit (SPA, adapter-static, ssr=false) shell, sqlx + SQLite (WAL) +
  FTS5, full 12-table schema, settings page, six route placeholders.
- **M2** — HF browser: search/filter/sort, file-tree, detail pane, README render (DOMPurify).
- **M3** — Downloader actor: resumable, parallel-chunked, GGUF header parse, local catalog,
  download drawer + installed-models route.
- **M4** — Engine: pin-and-fetch llama.cpp (`b9222`), llama-server sidecar lifecycle, NVML
  VRAM telemetry, VRAM fit estimator, install wizard, VRAM gauge, fit badge.
- **M5** — Chat: single-model streaming over `Channel<ChatEvent>` (SSE → llama-server OpenAI
  endpoint), JSONL transcripts with seq/byte-offset recovery, FTS5 search, conversation list,
  model load/unload, engine-installed gate. Verified end-to-end (load → stream → response).

---

## M6 — Rebrand → OpenHarness + UI/UX Overhaul  ✅

The current UI is a functional **testing placeholder**. This milestone makes OpenHarness look
and feel like a real product — adopting the **ai-shuffle** design language — and fills the
glaring functional gaps (model load config, sampling, system prompt). It also completes the
rename from "Hangar AI" to **OpenHarness**.

Decisions locked: **full rename with data migration**, **faithful adoption** of the
ai-shuffle aesthetic (clean sans for headers, not Playfair serif).

### 6.0 — Rebrand: Hangar AI → OpenHarness  ✅

Full rename, with a one-time migration so the current install (engine, models, DB) carries over.
*Landed: crate `openharness` / lib `openharness_lib`, identifier `com.openharness.app`, data dir
`%LOCALAPPDATA%\OpenHarness`, credential service `ai.openharness.app`, all visible strings
updated. First-launch migration moves the legacy `Hangar` dir + HF-token credential and shows a
one-off toast (`consume_migration_notice`). `cargo check` + `pnpm check` green.*

- **Identifiers & strings**
  - Crate: `hangar-ai` → `openharness` (`Cargo.toml`, `tauri.conf.json` `productName`,
    `mainBinaryName`, window title, `identifier` e.g. `com.openharness.app`).
  - All visible "Hangar"/"Hangar AI" strings (Sidebar footer, window title, install wizard
    copy, READMEs).
- **Filesystem & secrets** (in `config` module + `commands/settings.rs`)
  - Data dir: `%LOCALAPPDATA%\Hangar` → `%LOCALAPPDATA%\OpenHarness` (db, models,
    transcripts, engine, settings.json).
  - Credential Manager: service `ai.hangar.app` → `ai.openharness.app`.
- **One-time migration** (run on first launch of the renamed build)
  - If `%LOCALAPPDATA%\OpenHarness` is absent but `…\Hangar` exists: move the directory
    (db, models, transcripts, engine, settings.json) and migrate the HF-token credential
    entry. Log + toast on completion. Idempotent; never clobber an existing new dir.
- **Acceptance:** fresh clone builds as `openharness.exe`, window titled "OpenHarness", and an
  existing Hangar install upgrades in place with all models/engine/history intact.

### 6.1 — Design system (port ai-shuffle tokens)  ✅

*Landed: `app.css` rewritten with Tailwind-4 `@theme inline` mapping semantic utilities
(`bg-surface`, `text-foreground`, `border-border`, `bg-primary`, tool colors) to runtime
OKLCH vars under `:root`/`html.dark` + `html.light` overrides. Inter + JetBrains Mono via
`@fontsource`; 11→18px type scale (`text-2xs`…`text-xl`); radius family; neo-glass `.neo-surface`
/ `.glass` / `.shadow-*` / `.shadow-glow` helpers; 150ms motion. Restyled primitives (Button
gradient + press, Card, Input, Badge chips, Drawer/Toast glass, Toggle, Spinner) and added
Slider, Select, Tabs, Modal, Tooltip, KbdHint. Kitchen-sink at `/settings/appearance`. `pnpm
check` + `pnpm build` green. (Visual eyeball pending — browser extension not connected here.)*


Adopt the ai-shuffle visual foundation. Tailwind 4's `@theme` in `src/app.css` is the home for
tokens (we stay CSS-first; no JS theme file beyond tooling).

- **Color** — port the OKLCH palette: dark blue-gray background (`oklch(0.14 0.008 260)`),
  layered surfaces (`secondary`/`muted` ~`oklch(0.22 …)`), **teal primary**
  (`oklch(0.72 0.14 180)`), semantic success/warning/error/info, and the tool-color set
  (bash=green, read=blue, write=amber, glob=purple) for M7. Keep `html.light` overrides for a
  future light mode (dark ships first).
- **Type** — Inter (sans, body + headers) + JetBrains Mono (code/metrics) via `@fontsource`.
  Type scale 11→18px per ai-shuffle. *(Swap their Playfair serif for Inter — dev-tool fit.)*
- **Spacing / radius / shadow** — 8px grid, `--radius` 12px family, layered neo-glass shadows,
  `backdrop-filter: blur(12px)` for overlays.
- **Primitives** — restyle the existing `ui/` kit (`Button`, `Card`, `Input`, `Badge`,
  `Drawer`, `Toast`, `Toggle`, `Spinner`) to the new tokens: gradient primary button with
  `scale(0.98)` press, neo-glass cards (border + inset highlight + layered shadow), focus rings,
  150ms `cubic-bezier(0.4,0,0.2,1)` transitions. Add `Slider`, `Select/Dropdown`, `Tabs`,
  `Modal` (port `UniversalModal` pattern), `Tooltip`, `KbdHint`.
- **Acceptance:** a `/settings` "Appearance" preview (or a temporary kitchen-sink route) renders
  every primitive in the new style; no slate/blue-500 placeholders remain.

### 6.2 — App shell  ✅

Replace the static sidebar with the ai-shuffle Codex-style shell.
*Landed: resizable sidebar (drag handle with teal hover bar, 240–400px, width +
collapsed state persisted to settings.json via the UI store), collapsible to a
64px icon rail, restyled nav with active accent bar, footer icon links
(Settings/Engine/Appearance). Responsive: off-canvas overlay + backdrop-blur with
a floating menu trigger under 1024px. `pnpm check` + `pnpm build` green. (Context
list — conversations in the sidebar — deferred to §6.4 with the chat rebuild.)*

- **Resizable left sidebar** (240–400px, drag handle with teal hover bar), collapsible; main
  panel fills remaining width. Persist width to settings.
- Sidebar sections: brand/header, primary nav (Chat, Models, Browse, Agents, MCP, Settings),
  and a context list (conversations on Chat, etc.). Footer icons (docs/stats/settings).
- Responsive: overlay + backdrop blur under ~1024px (desktop-first, but don't break narrow).
- **Acceptance:** sidebar resizes and persists; nav + active states match the new design;
  layout holds from 1024px up.

### 6.3 — Restyle existing surfaces  ✅ (chat deferred to §6.4)

*Landed: swept ~330 hardcoded slate/blue/semantic utilities across 21 M2–M5 surface
files onto the new tokens (bg-surface/elevated/muted, text-foreground/muted-foreground,
border-border, bg-primary, success/warning/error chips, fit badge + VRAM gauge colors).
`chat/+page.svelte` intentionally left on old colors — it gets rebuilt in §6.4. `pnpm
check` + `pnpm build` green.*

Re-skin (not rebuild) the working M2–M5 pages onto the new system: `/models/browse`
(list + detail pane, file tree, filters), `/models` (installed table, fit badge, load/unload),
Downloads drawer + job rows, `/settings`, `/settings/engine` (VRAM gauge, install wizard).
- **Acceptance:** every shipped surface uses the new primitives; visual parity with ai-shuffle
  density and depth; no functional regressions (load/unload, download pause/resume, search).

### 6.4 — Chat surface + the missing model/chat config  ⭐ ✅

*Landed. **Backend:** migration 0002 `conversation_settings`; `chat::settings`
(SamplingParams/LoadConfig/ConversationSettings + get/set/effective/global_defaults);
chat manager prepends the system prompt and applies the full sampling set (incl.
stop); `engine_load_model` threads batch/threads/flash-attn/KV-cache/mmap/mlock to
llama-server; `get/set_conversation_settings` commands. **Frontend:** rebuilt chat
workspace (token-styled bubbles, markdown render via marked→DOMPurify with code-block
copy, streaming indicator, auto-scroll, copy action, token footer) + collapsible
right config panel (Model load-time config with live VRAM fit + Apply&reload,
Sampling sliders with reset, System prompt + presets); settings persist per
conversation. Also fixed the `engineSwitchVariant` TS return-type lie. `pnpm check`
+ `pnpm build` + `cargo test` green.
**Deferred:** message regenerate / delete-from-here need a backend transcript-truncate
command (follow-up); custom (user-saved) system-prompt presets — built-in presets ship now.*


The headline gap. Build the real chat workspace and expose the controls we don't have yet.

**Frontend**
- **Chat workspace redesign** — message bubbles → ai-shuffle assistant/user styling, markdown
  render (DOMPurify-sanitized) + code blocks with copy, streaming typewriter + indicator,
  message actions (copy, regenerate, delete-from-here), token/latency footer.
- **Right-hand config panel** (per-conversation), collapsible:
  - **Model** — which model is loaded; load/unload from here.
  - **Load-time config** (applied on (re)load of the model into llama-server):
    context length (`-c`), GPU layers (`-ngl`), batch size, threads, flash-attention toggle,
    KV-cache type, mmap/mlock. Show the VRAM fit estimate live as ctx/ngl change.
  - **Sampling** (per-request, live-editable): temperature, top_p, top_k, min_p, max_tokens,
    repeat_penalty, presence/frequency penalty, seed, stop sequences. Sliders + numeric inputs
    with sane defaults and "reset to default".
  - **System prompt** — multiline, per-conversation; with a small library of saved presets.
- **Defaults vs per-conversation** — global defaults in `/settings`, overridable per
  conversation; show when a conversation diverges from defaults.

**Backend**
- **System prompt**: chat manager currently builds the request from history only. Add a system
  message (per-conversation) prepended to the `messages` array.
- **Per-conversation settings**: persist sampling + system prompt + load config per conversation
  (new columns on `conversations`, or a `conversation_settings` table) instead of only the
  global `settings` rows that `load_sampling` reads today.
- **Extend `engine_load_model`**: it currently takes `ctx` + `ngl` only — thread through batch,
  threads, flash-attn, KV-cache type, mmap/mlock to the llama-server child args.
- **Reload semantics**: changing load-time config requires an engine reload; sampling/system
  prompt apply on the next message without reload. Make the distinction obvious in the UI.
- **Stop sequences / extra sampling** in the request body builder (`sse`/`manager`).

**Acceptance:** I can set a system prompt + temperature + context length on a conversation, send
a message, and observe the behavior change; load-time changes trigger a visible reload; settings
persist across restart and per conversation.

### M6 — tracked-debt cleanups to fold in
- **Engine logs viewer** (`/settings/engine`) — ✅ closed in PR #11 (`engine_logs` + viewer card).
- **Orphan engine rows** — ✅ closed in PR #11 (`engines::prune_inactive` on startup + install).
- **`engineSwitchVariant` type lie** — ✅ fixed in §6.4: TS now returns `void`.

---

## M7 — Agent harness  *(was "agent loop"; now the headline milestone)*

The core of the product (see **Product vision**): an agentic loop with built-in tools
(**Bash / Read / Write / Edit / Glob / Grep / WebFetch**), a **per-tool approval gate**, and
**Plan / Auto / Yolo** modes — surfaced through a **profile-first workspace**. The old M8 (agent
profiles) has been **pulled forward into M7**: in a harness, profiles *are* the primary surface, not a
follow-on. Tool-call rendering uses the semantic tool colors from the design system; events stream
over a Tauri Channel.

**Decisions locked (M7 planning, 2026-06-09):**
- **ReAct text protocol** — model-agnostic, works with any GGUF, no native function-calling /
  chat-template dependency.
- **User-selected, unrestricted working dir** — no path jail; safety comes from the approval gate.
- **Profile-first** — users define named agent profiles and launch runs from them; ship curated
  starters (adapted from ai-shuffle's built-in subagents, see Appendix).
- **A run is a conversation** with the **full context stack** persisted (every tool call, tool
  result, assistant message, thinking block — not just role+content), plus a **context viewer** and
  **auto-compaction** so runs don't dead-end on the model's ctx window.
- **Smart model loading** — one-click load with auto-`-ngl` from the VRAM fit estimate + sensible
  ctx; expert knobs behind "Advanced." Launching an agent **auto-reloads** its model if not active.

### 7.1 — Backend foundation  ✅ *(2026-06-09)*

*Landed. **Tools** (`src-tauri/src/tools/`): `ToolKind` enum-dispatch over the 7 built-ins,
each resolving relative paths against `ToolContext.cwd`; `ToolRisk` = ReadOnly | Network |
Mutating; tools never `Err` to the loop — failures return `ToolOutput{is_error:true}` so the
model can recover. `read` (cat-n + offset/limit), `write` (mkdir -p parents), `edit` (unique-or-
replace_all string swap), `glob` (glob crate, newest-first), `grep` (walkdir + globset filter +
regex, content/files/count modes), `bash` (PowerShell on Windows / `sh -c` elsewhere, cwd, timeout
+ output cap), `web_fetch` (reqwest + crude HTML→text strip). 25 unit tests.
**Agent** (`src-tauri/src/agent/`): `policy` (Plan/Auto/Yolo × risk → Auto/NeedsApproval/Forbidden
+ per-session "approve-always" allowlist; Plan is a hard read-only fence); `protocol` (parse
`<tool_call>{json}</tool_call>` blocks — tolerant of code fences / name+arguments aliases / unknown
tools / bad JSON — plus the system-prompt builder and `<tool_result>` formatter); `runner` (the
actor + loop: model call → parse → policy → approval gate via `oneshot` → execute → feed results
back, 25-iteration cap, token streaming). Model access is behind `ModelClient` (boxed-future trait)
so the loop is unit-tested with a scripted mock; `LlamaModelClient` is the production impl streaming
from llama-server (resolves live engine port/model per call, injects a `"<tool_result"` stop).
20 agent tests (incl. full approve/deny/forbidden round-trips). **Wiring:** `agent` handle on
`AppState`; commands `agent_run` / `agent_approve` / `agent_cancel` / `subscribe_agent_events`.
`cargo test` 95 green, `cargo check` clean (no new warnings), `pnpm check` 0 errors.*

### 7.2 — Agent profiles + curated starters  ✅ *(2026-06-09)*

*Landed. **Migration 0003** extends `agents` with `description`/`icon`/`color`/`working_dir`/
`sampling_json`/`load_config_json`/`max_iterations`/`is_builtin`/`updated_at` (revert-to-defaults
reads the code-defined builtins rather than storing `default_*` columns — simpler). **`agent/profile.rs`**:
`AgentProfile` (reuses `SamplingParams`/`LoadConfig`), `ProfileInput`, `allowed_tools()` (empty or
`["*"]` = all), full CRUD (`list`/`get`/`create`/`update`/`delete`/`reset_to_defaults`; built-ins
can't be deleted, only reset), and `seed_builtins` (idempotent, run on launch). **Curated starters**
(adapted from ai-shuffle): General Purpose (all tools, Auto), Explore (read/glob/grep, Plan), Plan
(read/glob/grep architect, Plan), Coder (all tools, Auto), Researcher (read/glob/grep/web_fetch,
Auto). **Runner wired to profiles:** `RunSpec` gained `allowed_tools`/`max_iterations`/`sampling`;
the loop refuses tools outside the profile's allow-list (before the mode policy), honours the
per-profile iteration budget, and `ModelClient::complete` now takes sampling per call. **Commands:**
`list_agents`/`get_agent`/`create_agent`/`update_agent`/`delete_agent`/`reset_agent_to_defaults`;
`agent_run(profile_id, prompt, working_dir?, mode?)` resolves the profile → `RunSpec` (sampling
falls back to global defaults; working dir = override → profile default → error). `cargo test` 101
green, `.sqlx` cache regenerated, `cargo check` clean.*
**Deferred to later slices:** persisting the "approve-always" allowlist to `tool_allowlists` (waits
for the approval UI, §7.4); applying a profile's load config via auto-reload (§7.6).

### 7.3 — Run-as-conversation + full context stack  ✅ *(2026-06-09)*

*Landed. Migration 0004 adds `working_dir`/`mode` to `conversations`; typed transcript entries
(`user`/`assistant`/`thinking`/`tool_call`/`tool_result`/`summary`); the runner persists every step
and reconstructs each model call's context from the stack. Commands `list_agent_runs`/
`get_agent_run`/`delete_agent_run`.*

Make agent runs first-class, persisted conversations that own the **entire** context.
- **Richer transcript:** extend the JSONL message schema + `MessageRecord` beyond `role`/`content`
  to typed entries — `user`, `assistant`, `thinking`, `tool_call` (tool + args + call_id),
  `tool_result` (call_id + output + is_error). Flip `messages_index.has_tool_use` accordingly; keep
  FTS over the human-readable text.
- **Persist the loop:** the runner writes each step to the transcript (currently events-only).
  A run = a `conversations` row with `agent_id` + working dir; reload restores the full thread.
- **Reconstruct context from the stack** for each model call (instead of the ad-hoc message list),
  so what the model sees == what's on disk == what the viewer shows.

### 7.4 — Profile-first workspace UI + approval gate  ✅ *(2026-06-09)*

*Landed. `/agents` rebuilt: profile rail + editor modal (`ProfileEditor.svelte` — identity, system
prompt, mode, model, tool allowlist, working dir, max iterations; reset for built-ins), launch view
(prompt + mode pills + working-dir input — typed inline rather than an fs dialog for now), run
timeline (`RunTimeline.svelte` + `ToolCallCard.svelte` on the semantic tool palette), approval
banner (approve / always-allow / deny / abort), run history with delete. Verified end-to-end in the
browser via the mock-IPC harness (`src/lib/ipc/mock.ts`). **Open folds closed in PR #11:**
always-allow persists per-project to `tool_allowlists` (migration 0011, seeds each run, sub-agents
inherit the parent's project); ProjectPicker gained a native Browse… via tauri-plugin-dialog.*

Rebuild `/agents` (currently the M7 placeholder) on the design system.
- **Profile list / editor** — cards (icon/color/description), create/edit/clone/delete,
  revert-to-defaults for built-ins; editor exposes system prompt, model, allowed tools, mode,
  working dir, sampling, load config, max-iterations.
- **Run view** — launch from a profile (working-dir picker via Tauri fs dialog, mode override,
  prompt). Render the event stream: model tokens (typewriter), `assistant_message` steps, and
  **tool-call cards** colored by the semantic palette (bash=green, read=blue, write=amber,
  glob=purple) with args + collapsible output. Run history in the sidebar (reuses conversations).
- **Approval gate UI** — on `approval_required`, inline approve / approve-always / deny / abort
  wired to `agent_approve`; "approve-always" persists to `tool_allowlists`.
- Subscribe to `subscribe_agent_events`; map each `AgentEvent` variant to the UI.

### 7.5 — Context viewer + auto-compaction  ⭐ ✅ *(2026-06-09)*

*Landed. `ContextViewer.svelte` panel: token gauge vs ctx window, per-entry estimates, evicted
entries dimmed/struck, amber summary cards. Runner compacts at the threshold via a model call,
writes a `summary` entry covering older seqs (`compaction_cutoff`), emits `context_compacted`.*

- **Context viewer** — an inspectable panel showing the exact context stack being sent to the model
  (system + every message/tool entry), with per-entry token estimates and a running total vs. the
  loaded model's ctx window.
- **Auto-compaction** — when the context approaches the ctx limit (e.g. ~80%), summarize the older
  turns via a model call into a compact summary entry, keep recent turns verbatim, and continue.
  Full transcript stays on disk; only the *active* context is compacted. Surface a "compacted"
  marker in the thread. (Mirrors Claude Code's compaction; ai-shuffle's chat-history-summarizer is
  a reference for the summary shape.)

### 7.6 — Smart model loading  ✅ *(2026-06-09)*

*Landed. `smart_load_args` (commands/engine.rs) derives `-ngl` + ctx from the VRAM fit estimator;
`agent_run` auto-loads/reloads the profile's `default_model` before the run starts.*

- **One-click load** — auto-derive `-ngl` from the existing VRAM fit estimator + a sensible ctx;
  full expert panel (batch, threads, flash-attn, KV-cache, mmap/mlock) behind "Advanced." Refactors
  the M6 per-conversation load panel toward sensible defaults.
- **Auto-reload on agent launch** — if the launched agent's `default_model` (+ its load config)
  isn't the active engine model, trigger a visible reload before the run starts.

### 7.7 — Projects (M7.5)  ✅ *(2026-06-09)*

*Landed. Profiles describe **how** an agent works; projects describe **where**. Replaces the
per-profile `working_dir` with a first-class `projects` table.*

- **Schema** (migration `0005_projects.sql`) — `projects` (id, name, path UNIQUE, created_at,
  last_used_at); `conversations.project_id` (FK, `ON DELETE SET NULL`); **`agents.working_dir`
  dropped** — profiles are now fully portable across projects.
- **Backend** (`src-tauri/src/projects.rs` + `commands/projects.rs`) — `open_project` is the one
  registration path: normalizes the path (`std::path::absolute`, no canonicalize → avoids `\\?\`
  UNC), validates it's a directory, dedupes by normalized path (touch + keep the original name),
  defaults the name to the folder basename. Plus `list_projects` (recency order), `rename_project`,
  `delete_project`. `agent_run` takes `project_id` instead of a raw dir: fresh runs require a
  project; resumed runs fall back to the conversation's `working_dir` snapshot if the project was
  deleted (runs never break). `conversations.working_dir` stays as the resolved-path snapshot.
- **Frontend** — `ProjectPicker.svelte` on the launch view: recent-projects popover + an
  "Open folder…" input that registers ad-hoc paths as projects (same single code path). Store
  preselects the most-recently-used project. Run header and history rows show the project name.
  Profile editor's working-dir field removed.
- **Deferred by user choice ("None yet")** — git surfacing (branch, dirty state) on projects is a
  later slice. The open **always-allow persistence** fold will land per-project in
  `tool_allowlists` when picked up.

## Post-M7 roadmap — multi-modal re-plan (2026-06-09)

> Decided with the user 2026-06-09 (same day as the M7 re-scope): OpenHarness broadens from
> agents-first to the four-pillar workspace (see **Product vision**). **M7 finishes first as
> specced** (§7.3–7.6). The milestones below are direction-level — each gets the full slice
> treatment when picked up — and ordering may shift (release can be pulled earlier).

## M8 — Runtime v2: Python worker + engine orchestrator  ✅ *(sliced + landed 2026-06-10, PR #5)*

> **Landed notes (2026-06-10):** all six slices shipped as specced, with these deltas:
> requirements are plain pinned `.txt` (not `.lock`) files; `generation_jobs` schema is
> id/task/status/params/outputs/meta/error/created_at/finished_at (progress is live-only,
> never persisted); the orchestrator's eviction signal is a typed `NeedsEviction` outcome
> (returned, not an event) emitted only when evicting would actually fix the fit; `job_done`
> carries `task` so the ledger attributes NVML actuals without job tracking. `agent_run`
> routes through the orchestrator; `engine_load_model` stays on the legacy direct path until
> the chat/models UI migrates (the ledger adopts externally-loaded models). Real-GPU e2e
> (venv install + actual generation) passed 2026-06-11 (PR #12). 143 Rust + 18 Python tests.

The second (and final) engine family, plus the VRAM brain that lets engines coexist.
**Decisions locked (2026-06-10):** worker protocol is **JSON-Lines over stdio** (request/response
+ unsolicited progress events; no port races, no HTTP server dep) — binary outputs are **written
to disk by the worker** (`assets\` staging) and returned as paths, never streamed over the pipe.
**Video generation is in scope** (pulled into v1 by user choice): text-to-video joins the
pipeline set, sized for 24 GB-class hardware (LTX-Video / Wan 2.x small).

### 8.1 — Worker bootstrap (pin-and-fetch uv + venv)  ✅
- `src-tauri/src/worker/bootstrap.rs`, modeled on `engine/installer.rs`: download a pinned
  standalone **uv** release zip → `%LOCALAPPDATA%\OpenHarness\worker\uv\`; materialize the venv
  at `worker\venv\` from **locked requirements** shipped with the app (`worker-py/requirements.
  cuda.lock` / `.cpu.lock`; torch CUDA vs CPU wheel chosen by NVML GPU detection); uv also
  provisions the pinned CPython (3.12) itself. Disk precheck (~10 GiB), staged progress
  (`downloading uv` → `installing python` → `resolving venv`), rollback on failure, idempotent
  re-verify on launch. Pins live in `worker/mod.rs` (`UV_VERSION`, `WORKER_PYTHON`, lockfile
  hash → venv stamp file for cheap up-to-date checks).
### 8.2 — Worker server (Python) + sidecar lifecycle (Rust)  ✅
- **`worker-py/`** (repo root, shipped as a Tauri resource, versioned with the app): a thin
  **JSON-Lines-over-stdio** server — `openharness_worker/__main__.py`, `protocol.py` (requests:
  `ping`/`capabilities`/`submit`/`cancel`/`unload`/`shutdown`; events: `job_progress`/
  `job_done`/`job_error`/`log`), `pipelines/` registry with **lazy imports** so startup costs
  nothing until a pipeline is first used. Each pipeline declares capability metadata (task,
  VRAM-estimate fn, model formats).
- **Rust** `worker/sidecar.rs` (spawn venv python on the resource dir, stdio pipes, stderr ring
  buffer, `ping` health, graceful `shutdown`-then-kill) + `worker/manager.rs` actor
  (`WorkerCmd` mpsc + `WorkerEvent` broadcast, `WorkerHandle` on `AppState`) mirroring
  `engine/manager.rs`. Commands + `subscribe_worker_status`; mock-IPC parity.
### 8.3 — Pipelines  ✅
- **text-to-image** — `diffusers`: SDXL-class safetensors + GGUF-quantized Flux for 24 GB.
- **STT** — `faster-whisper` (file in; segments + timestamps out).
- **TTS** — **Kokoro** first (small, fast); Qwen3-TTS / XTTS-class voice-clone next.
- **text-to-video** — LTX-Video / Wan 2.x small, sized for 24 GB; long-job progress (per-step
  callbacks → `job_progress`).
- Each pipeline: load/unload entry points (reports actual VRAM after load via worker), submit
  with params schema, deterministic output file naming into the assets staging dir.
### 8.4 — Engine orchestrator  ✅
- One actor owning **all** engines (llama-server instances + worker pipelines): VRAM ledger
  (fit estimator + NVML actuals), **concurrent-when-fits**, and when a requested load doesn't
  fit, emit a choice of what to unload (no silent eviction). Absorbs §7.6 smart-loading.
  Chat/agent/Studio all request models through the orchestrator from here on.
### 8.5 — Generation primitives  ✅
- Migration `0006_generation_jobs.sql` (`generation_jobs`: id, task, model_id, params_json,
  status, progress, output_paths_json, error, created/started/finished_at). Commands
  `generate_image` / `synthesize_speech` / `transcribe_audio` / `generate_video` +
  `list/cancel/delete_generation_job` + `subscribe_generation_progress` (Channel). Reused
  verbatim by Studio (M10), agents + chat voice (M11).
### 8.6 — Runtime settings UI  ✅
- `/settings/runtime`: worker install wizard (M4 wizard pattern), venv status/version/variant
  (CUDA/CPU), pipeline capability table, worker log panel, reinstall/repair.

## M9 — Multi-modal store  *(absorbs old "curated HF store")*  — sliced 2026-06-10

Expand the M2/M3 browser+downloader from GGUF-text-only to every modality. Slices land
backend-first, one commit each, on `m9-multimodal-store`.

### 9.1 — Catalog v2 (schema + registry generalization)  ✅
- Migration `0007_catalog_v2.sql`: `models` gains `format` (`gguf` | `safetensors` |
  `diffusers-dir`, NOT NULL DEFAULT `'gguf'`), `task` (HF `pipeline_tag`-style string, nullable),
  and `local_path` (set for dir-layout models; GGUF blobs keep the `model_files` path).
  `model_files` rebuilt with `rel_path` (path inside the repo) and `UNIQUE(model_id, rel_path)`
  replacing `UNIQUE(model_id, role)` — multi-file models are many rows (`role`:
  `primary`/`weight`/`config`/`other`).
- `db/models.rs`: `insert_model_with_files` (vec), list/get carry format/task/local_path +
  aggregate size; existing single-file GGUF path unchanged in behavior.
- `models/registry.rs`: format-dispatch on registration — GGUF keeps the parse-then-register
  path; repo downloads register with `format`/`task` from the layout classifier + HF card.
### 9.2 — Downloader v2 (multi-file repo downloads)  ✅
- Job groups: `download_jobs` gains `group_id` + per-job `rel_path`; `Command::StartRepo
  { repo_id, revision, files, task, format }` fans out one job per file under one group.
  Pause/resume/cancel/delete accept a group and apply to all member jobs. Existing
  single-GGUF `Start` is a group of one.
- Disk layout: dir-layout models land at `models\repos\<owner>__<name>\<rel_path>`
  (diffusers needs the directory shape preserved); GGUF blobs stay `<sha256>.gguf`.
- Registration fires once, when the **last** job in the group verifies; partial-failure
  leaves the group resumable, never half-registers.
- Progress: per-job events unchanged; frontend aggregates by `group_id`.
### 9.3 — Browse ALL of HF by task  ✅
- `models/hf.rs`: `HfSearchParams.task` (raw `pipeline_tag` passthrough) supersedes the
  M2 capability mapping (kept for back-compat); store tabs map to: `text-generation`,
  `text-to-image`, `text-to-video`, `text-to-speech`, `automatic-speech-recognition`,
  `feature-extraction`. Library filter generalized (`gguf`/`safetensors`/`diffusers`/
  `transformers`).
- Repo **layout classifier** (`models/layout.rs`): siblings → `gguf` | `safetensors-single` |
  `safetensors-shards` | `diffusers-dir` (`model_index.json` present) | `unknown`; result +
  the file subset to download surfaced in `HfModelDetails`.
### 9.4 — Runnable badges + curated picks  ✅
- Runnability resolver: (layout, task) → verdict: runs on llama.cpp (gguf ×
  text-generation), runs on a worker pipeline (task maps + format ∈
  `PipelineCapability.formats`), or **not runnable here** (download still allowed, badged
  honestly). Command `store_runnability` (batch, for search pages).
- **Curated picks** per modality: static manifest baked into the binary (repo id +
  recommended file/variant + min-VRAM tier + blurb), filtered by detected VRAM;
  `store_curated_picks` command.
### 9.5 — Store frontend  ✅
- `/models/browse`: task tabs (All · Text · Image · Video · Speech · Transcribe ·
  Embeddings), format filter, runnable badge on rows + detail pane verdict, **Download
  repo** button for dir-layout models (aggregate group progress in the downloads tray),
  curated-picks rail per tab.
- `/models`: format/task badges on installed rows; dir-layout models show folder size.
- Full mock-IPC parity for every new/changed command.
### 9.6 — VRAM-aware "best quant" one-click + quant guidance  ✅
- For GGUF text repos: resolve the largest quant that fits current free VRAM (fit estimator
  over the file list, headroom margin) → one-click "Best for my GPU" download button;
  plain-language quant guidance copy in the file tree. Collections/favorites **deferred**
  (post-v1 fold) — `pinned` on installed models already covers the minimal case.

## M10 — Studio  *(new)*  ✅

- **Task pages** ✅ — Image (txt2img; SDXL/SD/Flux arch auto-detect, size presets,
  steps/guidance/count/seed, multi-image grid), Speech (Kokoro voice picker + speed,
  default-model fallback, inline player), Transcribe (Whisper size fallback,
  transcribe/translate, word timestamps, transcript via asset protocol). Each page:
  task-filtered ModelPicker, streaming JobProgressCard, results inline, eviction
  dialog (unload/force/retry) against the VRAM orchestrator. img2img + mic capture
  deferred to a later fold.
- **Shared asset library** ✅ — migration 0009 `assets` table + FTS, generation outputs
  promoted to `%LOCALAPPDATA%\OpenHarness\assets\` on JobDone; gallery with kind filter
  pills + debounced FTS search, per-kind previews (asset protocol via `fileSrc`),
  reveal-in-explorer, delete with optional file removal.
- **Video task page** ✅ — text-to-video via the M8 worker pipeline (LTX-Video / Wan-class
  presets), 8n+1 frame slider with seconds estimate, long-job progress, results into the
  same asset library. *(Pulled into v1 on 2026-06-10 by user decision.)*
- Browser-verified e2e via the mock-IPC harness (all five tabs + eviction flows).
  **Real-GPU smoke (M8 venv install + actual generation) passed 2026-06-11 (PR #12).**

## M11 — Cross-pillar integration  *(new — the differentiator)*  ✅ *(2026-06-10)*

- **Agent generation tools** ✅ — `generate_image` / `text_to_speech` / `transcribe` join the
  built-in tool set (`src-tauri/src/tools/generate.rs`, `GenDeps { db, worker, orchestrator }`;
  risk-tiered + approval-gated like every other tool; results promoted into the asset library).
- **`web_search` tool** ✅ — built-in harness tool beside `web_fetch`
  (`tools/web.rs::run_search`, keyless DuckDuckGo HTML scrape, no API key required), available
  to the agent loop (`ToolKind::WebSearch`, 11 tools total) **and** chat: a per-message globe
  toggle augments the outgoing request with live results (`ChatEvent::WebSearching` +
  "Searching the web…" indicator; the stored transcript stays verbatim — only the model-bound
  copy is augmented).
- **Chat voice mode** ✅ — mic STT input (MediaRecorder → `save_voice_capture` →
  faster-whisper transcribe job → composer) and TTS read-aloud output (Kokoro
  `text_to_speech` job → audio playback; per-message Speak/Stop + auto-speak toggle in the
  chat header, voice settings persisted to localStorage). Voice jobs run headless through
  `studio.submitJobAndWait` (waiter map on the single generation-channel subscription) so they
  never disturb Studio page state; on `needs_eviction` they fail with a clear message instead
  of force-evicting.
- Browser-verified e2e via the mock-IPC harness (web-search flow, voice settings persistence,
  Speak/Stop lifecycle). 179 Rust tests. **Real-GPU voice smoke (actual whisper/Kokoro runs) passed 2026-06-11 (PR #12).**
- Open folds: paid web-search API backend (Brave/Serper as an optional settings key).
  Voice-job eviction dialog ✅ closed in PR #11 (global EvictionDialog + submitJobAndWait
  resubmit/cancel hooks).

## M12 — Sub-agent orchestrator  *(was M8)* ✅ *(2026-06-10, PR #9)*

A parent agent can **spawn scoped sub-agents** (each backed by a profile, à la ai-shuffle's
subagent model), pass them a task + working dir, collect their results, and fold them back into the
parent's context. Includes a sub-agent tool exposed to the loop, concurrency control, and nested
run rendering in the workspace.

**Shipped:**
- **`task` tool** (`src-tauri/src/tools/task.rs`, aliases `subagent`/`spawn_agent`/…): args
  `profile` (name or id, case-insensitive), `task`, optional `working_dir`. Risk tier
  **read-only** — safe because the child's effective mode is **clamped to at least the parent's
  restrictiveness** (`policy::clamp_subagent_mode`, Plan > Auto > Yolo), so a Plan parent can
  never spawn a mutating child.
- **No grandchildren:** the child's allow-list is the profile's tool set (or ALL) **minus
  `task`** — nested spawn attempts are denied by the existing allow-list gate.
- **Runner plumbing:** `RunOutcome` (Finished/Aborted/Failed) over a dedicated oneshot via
  `AgentHandle::run_watched`; `RunScope` injected into `ToolContext.run` (run/conversation ids,
  mode, sampling, ctx window, cancel token, event bus); `AgentDeps` (handle + db) in
  `ToolContext.agent`. Parent blocks on the child; cancelling the parent cancels the child.
- **Persistence:** child runs are real conversations with `parent_conversation_id` (migration
  0010, FK ON DELETE CASCADE), excluded from `list_agent_runs`, surfaced via
  `list_agent_run_children`; child transcript JSONL like any run.
- **Events:** `SubAgentStarted` on the parent's bus (child run/conversation ids, profile,
  task, clamped mode); all child events flow on the child's `run_id`.
- **Frontend:** nested sub-run panel inside the `task` ToolCallCard (profile, mode pill, live
  status/stream tail/step count, final report, **Open run →** into the persisted child);
  approval banner gains a `sub-agent · <profile>` chip when a child blocks on approval; store
  routes child events via a `subRuns` map correlated to the spawning call.
- Mock-IPC parity (scripted sub-agent demo, child-run registry) for browser e2e. 185 Rust
  tests (4 new M12 integration tests: spawn/fold-report, persist+clamp, unknown-profile soft
  error, no-grandchildren).
- Deliberately deferred: concurrency >1 (children run sequentially inside the parent's tool
  call — parallel fan-out lands with a `parallel` arg when a real need shows up).

## M13 — MCP client (rmcp)  *(was M9)* — ✅ 2026-06-10 (PR #10)

`rmcp 1.7` MCP client: **stdio + streamable-HTTP** transports, a **sampling handler** that routes
MCP sampling requests to the local loaded model, and an **MCP registry browser** to add/configure
servers (schema already present: `mcp_servers`, `mcp_tool_consents`). MCP tools become callable from
the agent loop under the same approval gate.

**Landed** (`src-tauri/src/mcp/`: `persistence` / `manager` / `sampling` / `e2e_tests`):
- **Connection manager actor** (single-writer, mirrors chat/agent/worker): handshakes established
  **off-actor** and handed back via an internal `Attach` message; tool calls clone the connection's
  `Peer` and run off-actor too. 30 s connect / 120 s call timeouts, 64 KiB output cap, status
  broadcast → `subscribe_mcp_events` channel bridge.
- **Namespacing**: tools surface as `mcp__<server-slug>__<tool>` (slugify + dedup);
  `protocol::is_mcp_name()` routes them in the agent loop before built-in `ToolKind` parsing.
- **Risk/consent model**: every MCP call is `Mutating` — Plan forbids, Auto prompts, Yolo runs.
  *Always allow* persists to `mcp_tool_consents` (+ in-memory allowlist for unpersisted runs);
  a saved *deny* blocks before prompting.
- **Sampling**: `LocalSamplingHandler` answers server-initiated sampling with the local model,
  auto-loading the default model via the orchestrator when nothing is resident.
- **HTTP transport quirk**: rmcp 1.7 implements `StreamableHttpClient` only for reqwest 0.13;
  project pins 0.12 (reqwest-eventsource) → renamed `reqwest13` dep scoped to the MCP manager.
- **12 Tauri commands** (registry CRUD, connect/disconnect, tool catalogue, consents) + channel.
- **Frontend**: `/mcp` registry page — live status cards, enable toggle, add/edit modal
  (stdio command/args/env, http URL), per-tool consent pills (Ask / Always allow / Deny),
  orphaned-consent cleanup, two-click delete; typed IPC wrappers + full mock-IPC parity.
- **e2e test**: real rmcp server over `tokio::io::duplex` through the production client path —
  manager round trip + agent-run dispatch (Yolo executes, Plan forbids). 197 Rust tests total.

## M14 — Release  *(was M11)*

Updater wiring, **code-signing pipeline**, signed **MSI** release, first-run **Backend Setup
wizard**, onboarding. Includes the **dev-vs-prod CSP split** (strip `unsafe-eval`/`unsafe-inline`
+ `ws://localhost` + `http://ipc.localhost` in the prod overlay) and the **SHA-NI release
rustflags** (`-C target-feature=+sha`) for fast GGUF verification.

## M15 — Model-first capability layer  ✅ *(phases 1–3, 2026-06-12)*

**Goal:** "choose the model, choose the workflow" — plug-and-play for newcomers, full control
underneath. Born from live feedback: the Speech form didn't react to the Qwen3-TTS variant
(engine had to be hand-flipped), and nothing answered "what can I *do* with this model?"

- **Phase 1 — capability records** (`src-tauri/src/models/capabilities.rs`): every
  `InstalledModel` carries a `caps: ModelCaps` — `runtime` (llama-server / worker / qwen-tts),
  `pipeline`, `workflows` (chat/agent/image/speech/transcribe/video/embeddings), `speech`
  (engine + voices + supports_speed/supports_instructions), `image_arch`, honest `reason` for
  non-runnable combos. **Computed at read time** in `from_row` from existing signals (GGUF arch,
  format, capability enum, HF task) — never stored, so resolver improvements self-heal existing
  installs with no migration. Resolution order: qwen3-tts arch first, then gguf-by-capability,
  then dir-layout-by-task. 12 resolver tests (228 total).
- **Phase 2 — "Use with…" routing** (`InstalledModelTable.svelte`): actions column renders one
  button per workflow — Chat/Agent open `ModelLoadDialog` (new `destination` prop) then land on
  the pillar; Image/Speech/Transcribe/Video deep-link `/studio?task=X&model=<id>`, consumed by
  `/studio` (`$app/state` query params → tab + per-task `preselect` prop). Non-runnable models
  show the `reason` as a tooltip.
- **Phase 3 — capability-driven Studio forms**: `SpeechTask` rebuilt around
  `selected?.caps.speech ?? BUILTIN_KOKORO_CAPS` (`src/lib/speech.ts`) — one picker covers
  built-in Kokoro, Kokoro repos, and Qwen3-TTS talkers; voices/speed/instructions render from
  the caps record and the engine chip is informational (never hand-flipped). ImageTask consults
  `caps.image_arch` before filename guessing. `ModelPicker` gains `capsFilter`/`onmodel` and
  drops the duplicate placeholder when a `defaultOption` exists. Full mock-IPC parity
  (`resolveMockCaps` mirrors the Rust resolver); browser-verified e2e including the
  needs-eviction → unload → retry flow.
- **Deferred phases** (user-approved follow-ups): **phase 4** — generic `transformers-auto` /
  `diffusers-auto` worker pipelines (run unlisted HF models with a `trust_remote_code` consent
  gate); **phase 5** — backend gallery manifest (folds into M14 installer debt, incl. qwentts
  prebuilt hosting).

---

## M16 — UX redesign: load-first models + pillar overhauls + engine consolidation  ✅ *(W1–W6, 2026-06-12)*

**Goal** (spec: `specs/m16-ux-redesign.md`): invert M15's "use this model with X" into
"**load** the model, then pick it inside X" — plug-and-play for the layman, full control one
disclosure away — and give each pillar its reference feel (chat → claude.ai, agents → Claude
Code, Studio → adaptive canvas). The M15 capability layer stays as the routing brain.

- **W1 — load-first model flow.** Backend: unified `load_model(model_id)` command dispatches
  on `caps.runtime` — llama-server loads go **through the orchestrator's gated text slot**
  (legacy-direct `engine_load_model` deleted; closes the M8 fold), qwen-tts through the
  sidecar, worker pipelines through a **new `preload` protocol op** (loads the model without
  running a job, reports NVML-actual VRAM via the normal `job_done` event; `preload()` in
  every pipeline module). `unload_model(model_id)` frees by catalog id. Ledger residents carry
  the originating catalog `model_id` (lazy submits correlate by model path via
  `registry::find_by_source_path`). Frontend: `ledger.svelte.ts` store (single ledger
  subscriber, loaded/loading per catalog id, eviction → global EvictionDialog via the studio
  store's generalized prompt slot); `/models` rows get **Load/Unload + Loaded chip (VRAM)**,
  workflow buttons → informational badges, `ModelLoadDialog` survives as the tuned-load
  "Options…" path; `ModelPicker` pins loaded models on top with a lazy-load hint. Mock-IPC
  parity for all three runtime kinds. **Lazy load-on-submit unchanged everywhere — preload is
  an accelerator, never a gate.**
- **W2 — chat → claude.ai feel.** Centered 48rem transcript (assistant = plain markdown on the
  page background, user = compact right-aligned blocks, hover actions kept), floating composer
  card (growing textarea + `ChatModelPicker` pill/popover with loaded-first grouping + inline
  load, web-search globe, mic, send/stop) replacing the load-gate page, collapsible recents
  rail, per-conversation config behind a gear → slide-over Drawer, centered-greeting empty
  state. New: `ChatComposer.svelte`, `ChatModelPicker.svelte`.
- **W3 — agents → Claude Code feel.** Terminal-flavored transcript (collapsed `⏺ Tool(args)`
  rows with status dots + result tails, dimmed collapsible thinking, compaction dividers),
  status line while running, follow-up composer with attached Plan/Auto/Yolo `ModePill` (mode
  passed per turn) + `ContextMeter` (toggles ContextViewer), **inline `ApprovalPrompt` anchored
  at the pending tool call** (ApprovalBanner deleted), left rail = project picker + history
  grouped by project, profiles demoted to a `ProfileManager` modal, honest per-run
  `AgentModelPicker` (loads via the ledger store; profile `default_model` shown as an auto-load
  lock since `agent_run` has no model param).
- **W4 — Studio adaptive canvas.** `TaskShell` result-first scaffold (compact controls | hero
  canvas with inline shimmer progress/error + `RecentStrip` remix history), simple-by-default
  forms (model + primary input + Generate), every numeric knob behind `AdvancedSection`
  (state per task in localStorage), capability-adaptive presets (`OptionPills`): image
  quality/size by `caps.image_arch` (turbo pins guidance 1.0, steps 4), video length/quality
  per family (LTX 24 fps / Wan 16 fps, 8n+1), speech Advanced from `caps.speech` incl. new
  VibeVoice **multi-speaker** reference audio. `JobProgressCard` deleted. No parameter
  regressions; browser e2e (Playwright + mock IPC) passed all tasks incl. eviction rounds.
- **W5 — store friendliness.** Plain-language task tabs with one-line descriptions
  (`task-meta.ts`), rows lead with "what it does" + "Runs on this PC / Needs setup / Won't run
  here", stats demoted, file list behind an "All files & formats" disclosure when a primary
  action exists, honest `caps.reason` callout in the detail pane.
- **W6 — settings IA + engine registry.** `engines.svelte.ts` adapter exposes uniform
  `EngineDescriptor[]` (status/facts/actions/detail-actions/pipelines/logs/residents) consumed
  by one generic `EngineCard`; `/settings/engine` + `/settings/runtime` merged into
  **`/settings/engines`** (cards + shared VRAM ledger) with 308 redirects from the old routes;
  hub regrouped General / Engines / Appearance; GpuStatus + sidebar links updated. Adding an
  engine = appending one descriptor builder; a Rust-side registry can slot in later.

**Verification:** 237 Rust + 23 Python tests; `pnpm check` 0 errors; headless-browser smoke of
every redesigned surface via the mock-IPC harness (models load/unload/chips, chat streaming +
drawer, agent run + inline approval, store browse, engines page + redirects, studio tabs) and
a full Studio e2e (generate/evict/cancel/remix per task).

---

## Tracked debt (carried + new)

From `CLAUDE.md`, still open (most resolved at the milestone noted):
- **CSP split** dev vs prod → **M14** (release).
- **Shared `reqwest::Client`** — `validate_hf_token` still builds a client per call; an
  `AppState.http` exists now, migrate remaining callers.
- **Engine ZIP SHA-256 pinning** — fetched over TLS, not checksum-pinned → engine-updater work.
- **CPU RAM-fit estimation** — `estimate_model_fit` is permissive with no GPU; refine with
  `sysinfo` when CPU inference matures.
- ~~GQA fallback table~~ — resolved by the loader planner (KV sized from K/V tensor widths).
- **Engine updater UX** — build pinned to `b12182`; no in-app update yet.
- **Graceful llama-server shutdown** — Windows hard-kills; try `/shutdown` with a grace window
  for long M7 sessions.
- **Port allocation race** — `pick_free_port` binds-then-drops; long-term pass an open socket
  or use a fixed range.
- **`EnginePhase` re-export** — type lives in `engine::manager`; re-export when M-work needs it.

New this cycle:
- **Orphan engine rows** on reinstall (see M6 debt cleanups).
- **`engineSwitchVariant` return type** mismatch (see M6 debt cleanups).
- *(Resolved this session: chat token-DTO shape, install-wizard freeze, engine `active` UNIQUE
  constraint, download chunk-retry/Range-200 corruption, model load/unload UI.)*

---

## Appendix — ai-shuffle design tokens to adopt

Reference: the AI Shuffle design system. The palette and component conventions are summarized below.

- **Palette (OKLCH, dark):** bg `0.14 0.008 260` · surface `0.22 0.008 260` ·
  border `0.28 0.01 260` · primary/teal `0.72 0.14 180` · accent-cyan `0.75 0.15 195` ·
  success `0.65 0.18 145` · warning `0.75 0.15 85` · error `0.55 0.22 25` · info `0.65 0.15 240`.
- **Tool colors:** bash=green `0.65 0.18 145` · read=blue `0.65 0.15 240` ·
  write=amber `0.75 0.15 85` · glob=purple `0.65 0.20 300`.
- **Type scale:** 11 / 12 / 13 / 14 / 16 / 18px. Inter + JetBrains Mono.
- **Radius:** `--radius` 12px; sm 6 / md 10 / lg 14.
- **Shadows:** layered (s/m/l) + selection ring/glow; inset top highlight `rgba(255,255,255,.03)`.
- **Motion:** 150ms `cubic-bezier(0.4,0,0.2,1)` default; 200–300ms
  `cubic-bezier(0.16,1,0.3,1)` for panels; `scale(0.98)` button press.
- **Shell:** resizable sidebar 240–400px, 6px drag handle (teal on hover), backdrop-blur
  overlays under ~1024px.
