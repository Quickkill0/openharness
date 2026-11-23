/**
 * Svelte 5 rune-based engine store.
 *
 * Tracks the current EngineStatus broadcast by the backend actor.
 * Call initEngineStore() once from the root layout onMount.
 */
import {
	engineStatus as fetchStatus,
	engineActive as fetchActive,
	engineInstall,
	engineUnloadModel,
	nativeEngineInfo,
	orchestratorLoadModel,
	type LoadModelOpts,
} from '$lib/ipc/commands';
import { subscribeEngineStatus } from '$lib/ipc/channels';
import { requestEviction } from '$lib/stores/studio.svelte';
import type {
	EngineKind,
	EngineRowDto,
	EngineStatus,
	EngineVariant,
	NativeEngineStatus,
} from '$lib/types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _status = $state<EngineStatus>({ phase: 'idle' });
let _activeEngine = $state<EngineRowDto | null>(null);
/** Install and store-start errors. Load failures reach the owning card through the status. */
let _error = $state<string | null>(null);
let _initialized = $state(false);
/** Latest native-engine probe; null until it has answered once. */
let _native = $state<NativeEngineStatus | null>(null);

/** Mirrors `NativeStatus::support().streaming` in the backend. */
function nativeCanStream(n: NativeEngineStatus | null): boolean {
	return Boolean(n?.installed && n.info?.features.ssd_expert_streaming);
}

// ---------------------------------------------------------------------------
// Read-only store object
// ---------------------------------------------------------------------------

export const engineStore = {
	get status() { return _status; },
	get phase() { return _status.phase; },
	get isReady() { return _status.phase === 'ready'; },
	get isInstalling() { return _status.phase === 'installing'; },
	/** True when an engine is persisted in the catalog — independent of whether a model is loaded. */
	get installed() { return _activeEngine !== null; },
	/** The persisted active engine row (version/variant/path), or null. */
	get activeEngine() { return _activeEngine; },
	/** Latest native-engine probe (installed, path, versions, features), or null before it answers. */
	get native() { return _native; },
	/** True when the native engine is installed and can stream routed experts. */
	get nativeStreaming() { return nativeCanStream(_native); },
	/** What the load gates read: see `textEngineBlock` in modelMeta. */
	get textEngines() {
		return {
			llama: _activeEngine !== null,
			nativeChecked: _native !== null,
			nativeStreaming: nativeCanStream(_native),
		};
	},
	/** False only once we know neither engine can run text. Unknown doesn't nag. */
	get hasTextEngine() {
		return _activeEngine !== null || _native === null || nativeCanStream(_native);
	},
	/** Launcher of the running text model, as the engine actor reports it. */
	get runningKind(): EngineKind | null {
		return _status.phase === 'ready' ? _status.launcher : null;
	},
	/** True when the current load, running model or load failure is the native engine's. */
	get nativeOwnsStatus() {
		return (
			(_status.phase === 'loading' || _status.phase === 'ready' || _status.phase === 'failed') &&
			_status.launcher === 'ssd_stream'
		);
	},
	get error() { return _error; },
	get currentModelId() {
		return _status.phase === 'ready' || _status.phase === 'loading'
			? _status.model_id
			: null;
	},
	/** Context window (tokens) the running llama-server was launched with. */
	get loadedCtx() {
		return _status.phase === 'ready' ? _status.ctx : null;
	},
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

export async function initEngineStore(): Promise<void> {
	if (_initialized) return;
	try {
		_status = await fetchStatus();
		_activeEngine = await fetchActive();
		await refreshNativeEngine();
		await subscribeEngineStatus((s) => { _status = s; });
		_initialized = true;
	} catch (e) {
		_error = String(e);
	}
}

/** Re-query the persisted active engine (after install / variant switch). */
export async function refreshActiveEngine(): Promise<void> {
	try {
		_activeEngine = await fetchActive();
	} catch (e) {
		_error = String(e);
	}
}

/**
 * Read the native engine's state. It is placed by hand, so pages that depend
 * on it ask again on mount. The backend caches the probe per binary size and
 * mtime; `retry` makes it probe again after a failure, for when the fix was
 * outside the binary (a missing DLL).
 */
export async function refreshNativeEngine(retry = false): Promise<void> {
	try {
		_native = await nativeEngineInfo(retry);
	} catch (e) {
		_native = { installed: false, hosted: false, path: '', error: String(e), info: null };
	}
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export async function installEngine(variant?: EngineVariant): Promise<void> {
	_error = null;
	try {
		await engineInstall(variant);
		// Engine now persisted — refresh so `installed` flips true and the
		// UI can move past the "install engine" gate.
		await refreshActiveEngine();
	} catch (e) {
		_error = String(e);
		throw e;
	}
}

/**
 * Load a text model with explicit tuning args through the orchestrator's
 * VRAM gate (M16 W1 — the legacy direct engine path is gone). A
 * needs-eviction outcome opens the global EvictionDialog; the promise
 * resolves once the model loads, or rejects if the user cancels.
 */
export async function loadModel(modelId: string, opts: LoadModelOpts = {}): Promise<void> {
	const attempt = async (force: boolean): Promise<void> => {
		const outcome = await orchestratorLoadModel({
			model_id: modelId,
			ctx: opts.ctx,
			ngl: opts.ngl,
			batch_size: opts.batchSize,
			threads: opts.threads,
			flash_attn: opts.flashAttn,
			kv_cache_type: opts.kvCacheType,
			mmap: opts.mmap,
			mlock: opts.mlock,
			force,
			overrides: opts.overrides,
		});
		if (outcome.outcome === 'needs_eviction') {
			return new Promise<void>((resolve, reject) => {
				requestEviction({
					task: 'load_model',
					params: { model_id: modelId },
					requiredMb: outcome.required_mb,
					freeMb: outcome.free_mb,
					candidates: outcome.candidates,
					resubmit: (f) => void attempt(f).then(resolve, reject),
					cancel: () =>
						reject(
							new Error(
								`Not enough free VRAM (${outcome.required_mb} MB needed, ${outcome.free_mb} MB free).`,
							),
						),
				});
			});
		}
	};
	await attempt(false);
}

export async function unloadModel(): Promise<void> {
	await engineUnloadModel();
}
