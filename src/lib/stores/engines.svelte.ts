/**
 * Engine-registry adapter store (M16 W6).
 *
 * Presents every runtime engine family with a settings surface — llama.cpp
 * text engine, Python worker, future engines — as a uniform
 * `EngineDescriptor` consumed by the generic EngineCard on
 * /settings/engines. (The Qwen3-TTS sidecar is card-less: fully on-demand,
 * surfaced only through the shared VRAM ledger.) Frontend-first by design: each descriptor is derived
 * from the existing per-engine stores and commands, so a future Rust-side
 * unified registry can slot in behind this same interface.
 *
 * Adding an engine = appending one builder to `builders` below. The page
 * renders whatever this store returns; zero new page sections.
 */
import {
	engineLogs,
	engineSwitchVariant,
	orchestratorUnloadIf,
	workerLogs,
} from '$lib/ipc/commands';
import { unloadLock } from '$lib/stores/chipLocks.svelte';
import {
	engineStore,
	initEngineStore,
	refreshActiveEngine,
	refreshNativeEngine,
} from '$lib/stores/engine.svelte';
import { gpuStore } from '$lib/stores/gpu.svelte';
import { initLedgerStore, ledgerStore } from '$lib/stores/ledger.svelte';
import { initWorkerStore, workerStore } from '$lib/stores/worker.svelte';
import { addToast } from '$lib/stores/toasts.svelte';
import type {
	EngineStatus,
	EngineVariant,
	PipelineCapability,
	ResidentInfo,
	ResidentKind,
	WorkerVariantSlug,
} from '$lib/types';

// ---------------------------------------------------------------------------
// Public descriptor contract
// ---------------------------------------------------------------------------

export type EngineTone = 'green' | 'blue' | 'yellow' | 'red' | 'slate';

/** Live install/download progress (engines that stream install state). */
export interface EngineInstallProgress {
	stage: string;
	pct: number;
	bytes: number;
	total: number;
}

/** Uniform status view every engine family maps its own phase union onto. */
export interface EngineStatusView {
	/** Raw phase key (engine-specific union member name). */
	phase: string;
	/** Badge label. */
	label: string;
	tone: EngineTone;
	/** One-line plain-language description of the current state. */
	detail?: string;
	error?: string;
	progress?: EngineInstallProgress;
	/** True while the engine is busy transitioning (starting/loading). */
	transitioning?: boolean;
}

/** A button the EngineCard renders generically. Errors are toasted by the card. */
export interface EngineAction {
	id: string;
	label: string;
	/** Label shown while the action's promise is in flight. */
	busyLabel?: string;
	tone: 'primary' | 'secondary' | 'danger';
	disabled?: boolean;
	/** Why the action is unavailable, shown as a tooltip when disabled. */
	reason?: string;
	run: () => Promise<void> | void;
}

export interface EngineFact {
	label: string;
	value: string;
	mono?: boolean;
}

export interface EngineLogsView {
	available: boolean;
	/** Shown when logs aren't available (or empty). */
	hint: string;
	fetch: () => Promise<string[]>;
}

export interface EngineDescriptor {
	id: string;
	name: string;
	/** Family line under the name — what this engine runs and how. */
	family: string;
	/** Longer description shown when the engine isn't installed. */
	description: string;
	installed: boolean;
	status: EngineStatusView;
	version: string | null;
	variant: string | null;
	/** Key/value rows rendered in the card body. */
	facts: EngineFact[];
	/** Primary lifecycle actions (install / start / stop / unload). */
	actions: EngineAction[];
	/** Engine-specific extras (variant switch, reinstall, refresh) — expandable detail. */
	detailActions: EngineAction[];
	/** Capability table rendered in the expandable detail, when the engine reports one. */
	pipelines: PipelineCapability[] | null;
	logs: EngineLogsView;
	/** Ledger residents owned by this engine (unload goes through `unloadResident`). */
	residents: ResidentInfo[];
}

// ---------------------------------------------------------------------------
// Module state (engines without a dedicated store yet)
// ---------------------------------------------------------------------------

// The llama.cpp install/reinstall flow runs through the existing InstallWizard
// modal; descriptor actions just flip this flag and the page hosts the modal.
let _llamaWizardOpen = $state(false);

export const enginesUi = {
	get llamaWizardOpen() {
		return _llamaWizardOpen;
	},
	set llamaWizardOpen(v: boolean) {
		_llamaWizardOpen = v;
	},
};

/**
 * Unload a ledger resident from any engines surface. Goes through the same
 * guards as the dock: refused while the model is in use or being replaced,
 * and a no-op error if the slot now holds a different model.
 */
export async function unloadResident(r: ResidentInfo): Promise<void> {
	const lock = unloadLock(r);
	if (lock) throw new Error(lock.reason);
	await orchestratorUnloadIf(r.id, r.model_id ?? null);
}

/** Idempotent: wire every per-engine store this adapter reads from. */
export async function initEnginesStore(): Promise<void> {
	await Promise.all([initEngineStore(), initWorkerStore(), initLedgerStore()]);
	await refreshNativeEngine();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function residentsOf(kind: ResidentKind): ResidentInfo[] {
	return ledgerStore.residents.filter((r) => r.kind === kind);
}

// ---------------------------------------------------------------------------
// llama.cpp (llama-server) — text generation
// ---------------------------------------------------------------------------

function llamaDescriptor(): EngineDescriptor {
	const s: EngineStatus = engineStore.nativeOwnsStatus ? { phase: 'idle' } : engineStore.status;
	const row = engineStore.activeEngine;
	const installed = engineStore.installed;

	const currentVariant: EngineVariant =
		row?.variant === 'cuda-12.4' || row?.variant === 'cpu'
			? row.variant
			: gpuStore.hasGpu
				? 'cuda-12.4'
				: 'cpu';
	const otherVariant: EngineVariant = currentVariant === 'cuda-12.4' ? 'cpu' : 'cuda-12.4';
	const otherLabel = otherVariant === 'cuda-12.4' ? 'CUDA 12.4 build' : 'CPU build';

	let status: EngineStatusView;
	switch (s.phase) {
		case 'installing':
			status = {
				phase: 'installing',
				label: 'Installing',
				tone: 'yellow',
				progress: { stage: s.stage, pct: s.pct, bytes: s.bytes, total: s.total },
			};
			break;
		case 'loading':
			status = {
				phase: 'loading',
				label: 'Loading model',
				tone: 'blue',
				detail: `Loading ${s.model_id}…`,
				transitioning: true,
			};
			break;
		case 'ready':
			status = { phase: 'ready', label: 'Ready', tone: 'green' };
			break;
		case 'failed':
			status = { phase: 'failed', label: 'Failed', tone: 'red', error: s.error };
			break;
		default:
			status = installed
				? {
						phase: 'idle',
						label: 'Idle',
						tone: 'slate',
						detail: 'No model loaded — llama-server starts when you load a text model.',
					}
				: { phase: 'idle', label: 'Not installed', tone: 'slate' };
	}
	if (status.error == null && engineStore.error) status.error = engineStore.error;

	const facts: EngineFact[] = [];
	if (row) {
		facts.push(
			{ label: 'Version', value: row.version, mono: true },
			{ label: 'Variant', value: currentVariant === 'cuda-12.4' ? 'CUDA 12.4' : 'CPU' },
			{ label: 'Path', value: row.path, mono: true },
		);
	}
	if (s.phase === 'ready') {
		facts.push(
			{ label: 'Model', value: s.model_id, mono: true },
			{ label: 'Port', value: String(s.port), mono: true },
			{ label: 'Context', value: `${s.ctx.toLocaleString()} tokens` },
		);
	}

	const installing = s.phase === 'installing';
	const actions: EngineAction[] = [];
	if (!installed && !installing) {
		actions.push({
			id: 'install',
			label: 'Install engine…',
			tone: 'primary',
			run: () => {
				_llamaWizardOpen = true;
			},
		});
	}
	if (s.phase === 'ready') {
		const text = ledgerStore.residents.find((r) => r.id === 'text');
		const lock = text ? unloadLock(text) : null;
		const modelId = s.model_id;
		actions.push({
			id: 'unload',
			label: 'Unload model',
			busyLabel: 'Unloading…',
			tone: 'secondary',
			disabled: lock !== null,
			reason: lock?.reason,
			run: () => (text ? unloadResident(text) : orchestratorUnloadIf('text', modelId)),
		});
	}

	const detailActions: EngineAction[] = installed
		? [
				{
					id: 'reinstall',
					label: 'Reinstall…',
					tone: 'secondary',
					disabled: installing,
					run: () => {
						_llamaWizardOpen = true;
					},
				},
				{
					id: 'switch-variant',
					label: `Switch to ${otherLabel}`,
					busyLabel: 'Switching…',
					tone: 'secondary',
					disabled: installing,
					run: async () => {
						await engineSwitchVariant(otherVariant);
						await refreshActiveEngine();
						addToast(`Switched to ${otherLabel}`, 'success');
					},
				},
			]
		: [];

	return {
		id: 'llama-server',
		name: 'llama.cpp engine',
		family: 'Text generation — llama-server for Chat and the local API',
		description:
			'Runs GGUF text models for Chat via llama-server. Pin-and-fetched on first run (~700 MB; CUDA or CPU build).',
		installed,
		status,
		version: row?.version ?? null,
		variant: row ? (currentVariant === 'cuda-12.4' ? 'CUDA 12.4' : 'CPU') : null,
		facts,
		actions,
		detailActions,
		pipelines: null,
		logs: {
			available: s.phase === 'ready',
			hint: 'No model loaded — llama-server is not running.',
			fetch: engineLogs,
		},
		residents: engineStore.nativeOwnsStatus ? [] : residentsOf('text_engine'),
	};
}

// ---------------------------------------------------------------------------
// Native engine — routed-expert models streamed from the SSD
// ---------------------------------------------------------------------------

const NATIVE_FOLDER = '%LOCALAPPDATA%\\OpenHarness\\engine\\native\\';

function folderOf(path: string): string {
	const cut = Math.max(path.lastIndexOf('\\'), path.lastIndexOf('/'));
	return cut > 0 ? path.slice(0, cut + 1) : NATIVE_FOLDER;
}

const yesNo = (v: boolean) => (v ? 'Yes' : 'No');

function nativeDescriptor(): EngineDescriptor {
	const n = engineStore.native;
	const s = engineStore.status;
	const owns = engineStore.nativeOwnsStatus;
	const text = residentsOf('text_engine')[0];

	let status: EngineStatusView;
	if (n === null) {
		status = { phase: 'idle', label: 'Checking', tone: 'slate' };
	} else if (!n.installed) {
		status = { phase: 'idle', label: 'Not installed', tone: 'slate', error: n.error ?? undefined };
	} else if (!engineStore.nativeStreaming) {
		status = {
			phase: 'failed',
			label: 'Not usable',
			tone: 'red',
			error: n.error ?? "This build doesn't report SSD expert streaming, so models can't load through it.",
		};
	} else if (owns && s.phase === 'loading') {
		status = {
			phase: 'loading',
			label: 'Loading model',
			tone: 'blue',
			detail: `Loading ${s.model_id}…`,
			transitioning: true,
		};
	} else if (owns && s.phase === 'ready') {
		status = { phase: 'ready', label: 'Ready', tone: 'green' };
	} else if (owns && s.phase === 'failed') {
		status = { phase: 'failed', label: 'Failed', tone: 'red', error: s.error };
	} else {
		status = {
			phase: 'idle',
			label: 'Installed',
			tone: 'slate',
			detail: 'No model loaded. It starts when you load a model whose plan uses it.',
		};
	}

	const facts: EngineFact[] = [];
	if (n?.installed) facts.push({ label: 'Path', value: n.path, mono: true });
	else if (n) facts.push({ label: 'Place files in', value: folderOf(n.path), mono: true });
	if (n?.info) {
		facts.push(
			{ label: 'Engine', value: `${n.info.engine} ${n.info.engine_version}`, mono: true },
			{ label: 'llama.cpp', value: `${n.info.llama_cpp.tag} (${n.info.llama_cpp.commit})`, mono: true },
			{ label: 'SSD expert streaming', value: yesNo(n.info.features.ssd_expert_streaming) },
			{ label: 'Graphics-memory expert tier', value: yesNo(n.info.features.cuda_vram_expert_tier) },
		);
		const devices = n.info.backends.flatMap((b) =>
			b.devices.map(
				(d) => `${b.name}: ${d.description || d.name} (${(d.memory_total_mib / 1024).toFixed(1)} GB)`,
			),
		);
		facts.push({ label: 'Devices', value: devices.length ? devices.join(', ') : 'None reported' });
	}
	if (owns && s.phase === 'ready') {
		facts.push(
			{ label: 'Model', value: s.model_id, mono: true },
			{ label: 'Port', value: String(s.port), mono: true },
			{ label: 'Context', value: `${s.ctx.toLocaleString()} tokens` },
		);
	}

	const actions: EngineAction[] = [];
	if (owns && s.phase === 'ready') {
		const lock = text ? unloadLock(text) : null;
		const modelId = s.model_id;
		actions.push({
			id: 'unload',
			label: 'Unload model',
			busyLabel: 'Unloading…',
			tone: 'secondary',
			disabled: lock !== null,
			reason: lock?.reason,
			run: () => (text ? unloadResident(text) : orchestratorUnloadIf('text', modelId)),
		});
	}
	actions.push({
		id: 'check',
		label: 'Check again',
		busyLabel: 'Checking…',
		tone: 'secondary',
		// The engine can't run a probe while a model is starting.
		disabled: s.phase === 'loading',
		reason: s.phase === 'loading' ? 'A model is starting. Check again when it has finished.' : undefined,
		run: () => refreshNativeEngine(true),
	});

	return {
		id: 'native',
		name: 'Native engine',
		family: 'Text generation — models larger than memory, streamed from the SSD',
		description:
			'No one-click download yet. Build the package with the "Engine package" GitHub workflow, or build it yourself (engine/README.md), then put openharness-engine-server.exe and its DLLs in the folder below and choose Check again.',
		installed: Boolean(n?.installed),
		status,
		version: n?.info?.engine_version ?? null,
		variant: null,
		facts,
		actions,
		detailActions: [],
		pipelines: null,
		logs: {
			available: owns && s.phase === 'ready',
			hint: 'No native model is running, so there are no logs to show.',
			fetch: engineLogs,
		},
		residents: owns ? residentsOf('text_engine') : [],
	};
}

// ---------------------------------------------------------------------------
// Python worker — Studio pipelines
// ---------------------------------------------------------------------------

function workerDescriptor(): EngineDescriptor {
	const w = workerStore.status;
	const variantLabel = (v: WorkerVariantSlug) => (v === 'cuda' ? 'CUDA' : 'CPU');

	// Fully automatic lifecycle: the app installs the venv on first run
	// (variant by GPU detection) and starts the process on demand, so this
	// card is a passive status surface — no install/start/stop buttons.
	let status: EngineStatusView;
	switch (w.phase) {
		case 'installing':
			status = {
				phase: 'installing',
				label: 'Installing',
				tone: 'yellow',
				detail: `Setting up the ${variantLabel(w.variant as WorkerVariantSlug)} runtime automatically — torch + diffusers + faster-whisper + TTS. The CUDA build is ~6 GB of wheels; this can take a while. You can keep using the app.`,
				progress: { stage: w.stage, pct: w.pct, bytes: w.bytes, total: w.total },
			};
			break;
		case 'stopped':
			status = {
				phase: 'stopped',
				label: 'On standby',
				tone: 'slate',
				detail: 'Installed — starts by itself the moment a Studio generation needs it.',
			};
			break;
		case 'starting':
			status = {
				phase: 'starting',
				label: 'Starting',
				tone: 'blue',
				detail: 'Starting worker process…',
				transitioning: true,
			};
			break;
		case 'ready':
			status = { phase: 'ready', label: 'Ready', tone: 'green' };
			break;
		case 'failed':
			status = { phase: 'failed', label: 'Failed', tone: 'red', error: w.error };
			break;
		default:
			// Auto-install kicks in right after launch — show that honestly
			// instead of a "Not installed" state the user can't act on.
			status = {
				phase: 'not_installed',
				label: 'Preparing',
				tone: 'blue',
				detail: 'Setting up automatically — the Python runtime installs itself on first run.',
				transitioning: true,
			};
	}

	const facts: EngineFact[] = [];
	if ('variant' in w) facts.push({ label: 'Variant', value: variantLabel(w.variant as WorkerVariantSlug) });
	if (w.phase === 'ready') facts.push({ label: 'Version', value: w.version, mono: true });

	return {
		id: 'worker',
		name: 'Python worker',
		family: 'Studio pipelines — image, video, speech, transcription',
		description:
			'Runs Studio generation (images, video, speech, transcription) in a managed Python venv. Fully automatic: it installs itself on first run and starts whenever a generation needs it — nothing to manage here.',
		installed: w.phase !== 'not_installed',
		status,
		version: w.phase === 'ready' ? w.version : null,
		variant: 'variant' in w ? variantLabel(w.variant as WorkerVariantSlug) : null,
		facts,
		actions: [],
		detailActions: [],
		pipelines: w.phase === 'ready' ? w.pipelines.filter((p) => !p.hidden) : null,
		logs: {
			available: w.phase === 'ready',
			hint: 'Worker is not running — no logs to show.',
			fetch: workerLogs,
		},
		residents: residentsOf('pipeline'),
	};
}

// ---------------------------------------------------------------------------
// Registry — append a builder here to add an engine card.
//
// The Qwen3-TTS sidecar (qwentts.cpp) is deliberately *not* listed: it is
// fully on-demand (lazy auto-install on first use) and surfaced only through
// the shared VRAM ledger's Unload control — no manual settings card.
// ---------------------------------------------------------------------------

const builders: (() => EngineDescriptor)[] = [llamaDescriptor, nativeDescriptor, workerDescriptor];

export const enginesStore = {
	/** Uniform descriptor list, derived live from the per-engine stores. */
	get engines(): EngineDescriptor[] {
		return builders.map((build) => build());
	},
};
