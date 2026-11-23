# M16 — UX Redesign: load-first models, pillar UI overhauls, engine consolidation

**Status:** spec, ready for implementation (written 2026-06-12, dictated by the product owner, drafted by the live session).
**Read first:** `CLAUDE.md` (constraints — Svelte 5 runes only, Tauri Channels, single-writer state, mock-IPC parity), `PLAN.md` (M6 design system, M8 orchestrator, M9 store, M10 studio, M15 capability layer).
**Baseline:** `main` at PR #19 (VibeVoice voice cloning) — 236 Rust + 20 Python tests, svelte-check clean.

## Why

M15 gave every installed model an honest capability record (`InstalledModel.caps`) and routed users from `/models` into workflows via per-row "Use with…" buttons. Live testing showed that flow inverts the owner's mental model. The product direction, in his words:

> "I'll load a model capable of TTS, and then I can just go to Studio and select that model as the model I wanna use. Or for LLMs — I just want to load the model, and then I can chat or use agents because I have that model loaded. … User experience should be extremely plug-and-play for the layman while allowing full control over models and processes for deeper learning."

Plus: Chat should feel like **claude.ai**, Agents should feel like the **Claude Code / Claude desktop app**, Studio should be an **adaptive, beginner-friendly canvas** (not "a bunch of inputs and a giant image viewer"), and Settings → Runtime must scale to many engines without one bespoke card per engine.

**Keep the M15 capability layer.** It stays the routing brain (which models appear in which pickers, which form controls render). What changes is the *user-facing verb*: from "use this model with X" to "load this model, then pick it inside X".

## Workstreams

Order matters: **W1 first** (it changes the data flow every other surface consumes). W2/W3/W4/W5/W6 are independent after W1 and can run in parallel. One PR per workstream.

---

### W1 — Load-first model flow

**Concept.** "Loaded" = resident in the orchestrator ledger, regardless of engine family (llama-server model, worker pipeline model, qwen-tts talker). Users load models on `/models` (or on demand from a workspace picker); workspaces then select among capable models, with loaded ones instantly usable.

**Backend:**
- New command `load_model(model_id) -> Result<LoadOutcome>` that dispatches on `caps.runtime`:
  - `llama-server` → existing engine load path **through the orchestrator** (this finally migrates `engine_load_model` off the legacy-direct path — closes the M8 fold; the ledger currently only "adopts" external loads).
  - `qwen-tts` → existing tts sidecar load.
  - `worker` → **new**: explicit pipeline preload. Add a worker protocol request `preload` (task + model path) that loads the pipeline/model into memory without running a job; reports NVML-actual VRAM like jobs do. Orchestrator reserves with the existing fit estimate first; `NeedsEviction` flows through the existing typed outcome → the global EvictionDialog.
- `unload_model(model_id)` symmetric, routed by resident kind (orchestrator unload already exists per resident).
- Ledger residents need a stable `model_id` correlation so the UI can show loaded state per catalog row (today residents are keyed by slot/kind; extend the snapshot payload with the originating catalog id where known).
- Generation commands keep their lazy-load behavior — submitting a job for a non-loaded model still works (reserve→load→run). Preload is a UX accelerator, not a gate.

**`/models` page:**
- Primary action per row: **Load** (or **Unload** + a "Loaded" status chip with VRAM when resident). Remove the per-workflow "Use with…" buttons.
- Keep the workflow capabilities visible as small **informational badges** (Chat, Agent, Image, Speech, …) — they tell the user where the model will appear, they're no longer buttons. Not-runnable rows keep the honest `caps.reason` tooltip.
- Loading a model that needs eviction goes through the existing EvictionDialog.

**Workspace pickers:**
- Chat: model picker in the chat surface (see W2) lists `caps.workflows ∋ chat` models, **Loaded** group pinned on top; picking an unloaded one triggers `load_model` inline with progress.
- Agents: profile `default_model` picker + a per-run model override in the launch view, same grouping.
- Studio: `ModelPicker.svelte` gains the same Loaded-first grouping (it already filters by caps); selecting an unloaded model just means the first job pays the load cost (existing behavior) — optionally show a "will load on first generate" hint.
- Deep-link routing (`/studio?task=X&model=Y`) and `ModelLoadDialog`'s `destination` prop survive only if W2/W3 still need them; otherwise delete dead code.

**Acceptance:** load a TTS-capable model on `/models` → it shows Loaded → open Studio → Speech → it's in the picker's Loaded group → generate works with no further load wait. Same for an LLM → Chat and Agents both offer it. Unload from `/models` or `/settings` frees VRAM and the chip clears. Mock IPC simulates load/unload/ledger for all three runtime kinds.

---

### W2 — Chat UI → claude.ai feel

Current: three-pane workspace (conversation list / transcript / config panel) from M6. Target: the calm, centered claude.ai layout.

- **Centered conversation column** (~48rem max-width), generous whitespace. Assistant messages are plain rendered markdown on the page background (no bubble); user messages are compact contained blocks, right-aligned or inset like claude.ai.
- **Composer**: floating rounded card centered at the bottom — textarea grows with content, with the **model picker** (W1 grouping), web-search globe toggle, voice input mic, and send button integrated in the card's bottom row. This replaces the "Pick a model to load" gate page: with no model loaded, the picker is the call to action.
- **Conversation list**: collapsible left rail (recents + search), default-collapsed on narrow windows. New-chat button prominent.
- **Config panel** (system prompt, sampling, load-time args — the `conversation_settings` work from M6 §6.4): move behind a settings icon → slide-over drawer. Keep every control; it's tuner UI, not default chrome.
- **Streaming**: keep token streaming + the web-search indicator; claude.ai-style typing affordance. Per-message hover actions (copy, regenerate, delete-from-here, Speak) stay.
- Empty state: centered greeting + composer (claude.ai's "How can I help" pattern), recent conversations beneath.

**Acceptance:** mock-IPC browser pass shows the new layout streaming a simulated reply; all existing chat features (voice in/out, web search, regenerate/truncate, per-conversation config) reachable; svelte-check clean.

---

### W3 — Agents UI → Claude Code feel

Current: profile rail + launch view + run timeline with colored tool cards. Target: the Claude Code / Claude desktop session aesthetic.

- **Run view = terminal-flavored session transcript**: monospace accents (JetBrains Mono is already in the design system), assistant prose in proportional type, **tool calls as single collapsed rows** (`⏺ Read(src/main.rs)` style — icon + tool + one-line summary + status dot) that expand on click to args/output. Thinking entries dimmed/italic, collapsed by default.
- **Status line** while running: spinner + current activity ("Running Bash…", "Waiting for approval…"), like Claude Code's spinner line.
- **Composer at the bottom of a run** for follow-up turns, with the **mode pill** (Plan / Auto / Yolo) attached to the composer the way Claude Code shows its permission mode; mode switching mid-run if the runner allows it (it does between turns).
- **Approval banner** → inline approval block in the transcript at the point of the pending tool call (approve / always-allow / deny / abort), styled like Claude Code's permission prompt, not a global banner.
- **Left rail**: project picker + run history (grouped by project), profiles demoted to a manage screen/modal — profile editing is occasional, not every-session chrome. Launch = pick project, pick profile, type the prompt in the same composer.
- Context gauge + compaction summaries stay (ContextViewer), surfaced as a small meter near the composer like Claude Code's context indicator; sub-agent (task tool) nested panels keep their live tail.

**Acceptance:** mock-IPC scripted run renders the new transcript (every AgentEvent variant has a visual); approval flow works inline; profile CRUD still reachable; svelte-check clean.

---

### W4 — Studio adaptive redesign

Current: per-task tab pages, each a `[340px controls | results]` grid with raw parameter inputs. Target: model-adaptive, beginner-first creative canvas.

- **Simple by default, full control on demand.** Each task shows only: model picker, the primary input (prompt / text / audio file), and Generate. Everything numeric — steps, guidance/cfg, seed, resolution, frames, fps, speed — moves into an **Advanced** disclosure (collapsed by default, state persisted per task in localStorage). Where it helps, offer plain-language presets instead of numbers (e.g. quality "Fast / Balanced / Best" mapping to steps+guidance; size "Square / Portrait / Widescreen").
- **Capability-adaptive forms.** The form is a function of `selected.caps` — this generalizes what SpeechTask already does (Kokoro voices+speed vs Qwen voices+instructions vs VibeVoice reference-audio+cfg). Image: arch-appropriate presets from `caps.image_arch` (e.g. turbo models hide guidance and default steps low). Video: frame/fps rules per family. Consider a shared `TaskForm` scaffold so each task page declares fields rather than hand-rolling layout.
- **Result-first layout.** The generation canvas is the hero (large centered preview/player with subtle frame); controls live in a compact panel (left or bottom bar); a **recent-generations strip** (this task's history from the asset library) sits under the canvas for quick compare/reuse — clicking a recent item restores its prompt/params into the form ("remix").
- **Progress** inline on the canvas (skeleton/shimmer + stage text + cancel), not a separate card stack. Errors stay honest (TaskError) but styled into the canvas area. Eviction keeps the global dialog.
- Library tab: keep, but unify visual language with the new canvas (cards, hover actions).

**Acceptance:** a first-time user can generate an image typing only a prompt; the Advanced panel exposes every parameter that exists today (no capability regression — including VibeVoice cloning fields); mock-IPC e2e per task; svelte-check clean.

---

### W5 — Store (browse) friendliness

Lighter touch than W4 — the M9 store works; make it legible to non-experts.

- Task tabs get plain-language labels + one-line descriptions ("Text — models you chat with", "Speech — text to voice", …). Curated picks rail stays the front door.
- Model cards: lead with what it does and whether it runs on this PC (runnable badge + VRAM fit in plain words — "Runs great on your GPU" / "Too big for your GPU"); park download counts/likes/quant jargon in secondary text.
- File tree / quant selection: keep "Best for my GPU" as the single obvious action; the full file list collapses behind "All files & formats" for experts.
- Detail pane: keep the honest runnability reasons (M15 `caps.reason` language) front and center for non-runnable repos.

**Acceptance:** browsing "Speech" as a novice yields an obvious "this one, click here" path; expert file-level control intact; mock parity.

---

### W6 — Settings IA + engine-registry runtime page

Current `/settings/runtime`: a Python-worker card, a hand-written Qwen3-TTS card, pipeline table, ledger. This dies at N engines (VibeVoice's runtime rides the worker, but more sidecars are coming — see CLAUDE.md tracked debt on qwentts hosting).

- **Engine registry UI.** One generic `EngineCard` component rendered from a list: llama.cpp (from `/settings/engine`), Python worker, Qwen3-TTS sidecar, future engines. Uniform contract per engine: name/family, status (with phase), version/variant, install-or-update action (wizard hook), start/stop where applicable, logs viewer, and its resident models. Engine-specific extras (CUDA variant switch, pipeline capability table) render in the card's expandable detail, not as separate page sections.
  - Frontend-first is acceptable: derive the card list from the existing per-engine commands behind a small TS adapter (`engines.svelte.ts` store). A Rust-side unified `EngineDescriptor` registry is a stretch goal — don't block the UI on it, but leave the adapter shaped so the Rust registry can slot in.
- **Merge `/settings/engine` + `/settings/runtime` → `/settings/engines`** (one page, engine cards + the shared VRAM ledger/loaded-models table with per-resident unload). Keep route redirects from the old paths.
- **Settings IA**: sidebar groups become General (HF token, paths), Engines, Appearance. The loaded-models/VRAM view here is the power-user mirror of W1's `/models` loaded state.
- The sidebar GpuStatus widget links here.

**Acceptance:** adding a hypothetical fourth engine = appending one descriptor, zero new page sections; all existing controls (worker install wizard, variant switch, tts logs/unload, per-resident unload) reachable; mock parity for every engine state.

---

## Cross-cutting requirements (all workstreams)

- **Constraints from CLAUDE.md hold**: Svelte 5 runes only; design-system tokens from M6 (`@theme` OKLCH variables, existing `ui/` primitives — extend, don't fork styles); streaming via existing Channels; no new telemetry.
- **Mock-IPC parity is mandatory per PR** (`src/lib/ipc/mock.ts`): every new command, channel payload, and state must be simulated so `pnpm dev` in a plain browser exercises the full UI. This is how UI work is verified here.
- **Gates per PR**: `cargo test` (suite currently 236), `pnpm check` (0 errors), browser e2e pass via mock IPC of the changed surface. Update `PLAN.md` (add M16 section) and `CLAUDE.md` milestone index in the final PR.
- **No backend behavior regressions**: chat/agent/generation/eviction flows keep their semantics; W1 is additive backend (preload + routing), everything else is frontend.
- **Reference look**: claude.ai (chat) and Claude Code desktop (agents) are *aesthetic* references — information density, calm hierarchy, collapsed-by-default detail. Don't clone branding; use OpenHarness tokens.

## Decisions already made (don't relitigate)

1. Capability layer stays; "Use with…" buttons go. Workflow badges become informational.
2. "Loaded" is orchestrator-resident, unified across all three engine families; worker pipelines gain explicit preload.
3. Lazy load-on-submit stays everywhere — preload is an accelerator, never a prerequisite.
4. Advanced parameters are hidden, not removed. Full control remains one disclosure away.
5. Per-workstream PRs, W1 first.

## Open items for the owner (non-blocking, default if unanswered)

- W3 demotes profiles to a manage modal — default: yes, launch flow = project + profile dropdown + prompt.
- W6 merges the engine and runtime settings pages — default: yes, with redirects.
- Real-GPU smoke for VibeVoice (PR #19) is still pending owner approval (~5.4 GB download, ~7 GB VRAM) — unrelated to this spec but queued.
