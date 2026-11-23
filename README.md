# OpenHarness

A local model loader and runner for Windows, with a Hugging Face store, VRAM-aware runtime, plain chat with voice, and Studio for image, video, speech and transcription. Agentic work lives in AI Shuffle; an OpenAI-compatible API connection is planned separately.

> Formerly **Hangar AI**. An existing Hangar install is migrated in place on first launch.

## Prerequisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| Rust | 1.78 | `winget install Rustlang.Rustup` |
| Node.js | 22 | `winget install OpenJS.NodeJS.LTS` |
| pnpm | 11 | `npm install -g pnpm` |
| Tauri CLI | 2.x | `cargo install tauri-cli --version "^2"` |
| WebView2 | runtime | ships with Windows 11; or install from Microsoft |

## Development

```powershell
# Install JS dependencies
pnpm install

# Launch Tauri dev shell (starts Vite + Rust hot-reload)
pnpm tauri dev
```

The first launch creates `%LOCALAPPDATA%\OpenHarness\data.db` and runs all migrations automatically.

## Build

```powershell
# Production MSI bundle (output in src-tauri/target/release/bundle/msi/)
pnpm tauri build
```

## Project layout

```
openharness/
├── src/               # SvelteKit frontend (SPA mode)
│   ├── lib/
│   │   ├── ipc/       # Typed Tauri invoke wrappers
│   │   ├── stores/    # Svelte 5 rune-based state
│   │   ├── components/# UI + nav components
│   │   └── types/     # Shared TypeScript types
│   └── routes/        # App routes (Chat, Studio, Models, Store, Settings)
└── src-tauri/         # Rust backend
    ├── src/
    │   ├── commands/  # Tauri IPC command handlers
    │   ├── db/        # SQLite pool + query helpers
    │   ├── auth/      # Keyring (Windows Credential Manager) helpers
    │   └── config/    # App data directory paths
    └── migrations/    # SQLx migrations
```

## Data directories

| Purpose | Path |
|---------|------|
| App data | `%LOCALAPPDATA%\OpenHarness\` |
| Database | `%LOCALAPPDATA%\OpenHarness\data.db` |
| Models | `%LOCALAPPDATA%\OpenHarness\models\` |
| Transcripts | `%LOCALAPPDATA%\OpenHarness\transcripts\` |
| Engine | `%LOCALAPPDATA%\OpenHarness\engine\` |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG` | Tracing filter (e.g. `debug`, `info`, `openharness_lib=debug`) |
| `SQLX_OFFLINE` | Set to `true` to skip live DB during `cargo check` |

## SQLx offline mode

After query or schema changes, regenerate the offline query cache against a disposable database, never the live app database:

```powershell
cd src-tauri
$env:DATABASE_URL = "sqlite:C:/path/to/disposable/openharness-prepare.db"
cargo sqlx database create
cargo sqlx migrate run
cargo sqlx prepare -- --all-targets
```

Commit the resulting `src-tauri/.sqlx/` directory.

Migration 0014 permanently deletes agent runs and their message/search/settings rows, agent profiles, projects, tool approvals and MCP configuration. Ordinary chats, their FTS data and all models/Studio data are preserved. Old transcript files and Fluid runtime folders on disk are not removed by SQL migrations.
