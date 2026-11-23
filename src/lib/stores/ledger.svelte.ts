/**
 * Svelte 5 rune-based VRAM-ledger store (M16 W1 — load-first model flow).
 *
 * Single subscriber to the orchestrator's ledger channel; owns the
 * "which catalog models are loaded right now" view every surface consumes
 * (/models Load/Unload buttons, workspace pickers' Loaded-first grouping,
 * /settings engines page). Loads route through the unified `load_model`
 * command and share the global EvictionDialog via the studio store's
 * eviction prompt slot.
 */
import { subscribeOrchestratorLedger } from '$lib/ipc/channels';
import { loadModelById, unloadModelById, orchestratorLedger } from '$lib/ipc/commands';
import { requestEviction } from '$lib/stores/studio.svelte';
import type { LedgerSnapshot, ResidentInfo } from '$lib/types';

/** Null until the first ledger event arrives (lets the UI show a loading state). */
let _snapshot = $state<LedgerSnapshot | null>(null);
/** Catalog ids with a load in flight (spinner state on buttons/pickers). */
let _loading = $state<Set<string>>(new Set());
/** Slot id → catalog id for loads in flight whose target slot is known. */
let _pendingSlots = $state<Map<string, string>>(new Map());
let _initialized = false;

export const ledgerStore = {
	get snapshot() {
		return _snapshot;
	},
	get residents(): ResidentInfo[] {
		return _snapshot?.residents ?? [];
	},
	/** Catalog ids currently loading via `loadModel`. */
	get loadingIds() {
		return _loading;
	},
	/** Slot id → catalog id being loaded into it (the dock locks that slot's chip). */
	get pendingSlots(): ReadonlyMap<string, string> {
		return _pendingSlots;
	},
	isLoaded(modelId: string): boolean {
		return this.residents.some((r) => r.model_id === modelId);
	},
	isLoading(modelId: string): boolean {
		return _loading.has(modelId);
	},
	residentFor(modelId: string): ResidentInfo | null {
		return this.residents.find((r) => r.model_id === modelId) ?? null;
	},
};

/** Idempotent: wire the orchestrator-ledger channel. */
export async function initLedgerStore(): Promise<void> {
	if (_initialized) return;
	_initialized = true;
	try {
		await subscribeOrchestratorLedger((snap) => {
			_snapshot = snap;
		});
	} catch {
		_initialized = false; // backend not up yet — retry on next init call
	}
}

function setLoading(modelId: string, on: boolean) {
	const next = new Set(_loading);
	if (on) next.add(modelId);
	else next.delete(modelId);
	_loading = next;
}

/**
 * Load a model by catalog id through the unified flow. A needs-eviction
 * outcome opens the global EvictionDialog; the returned promise resolves
 * `true` once loaded, `false` when the user cancels out of the dialog.
 * Failures reject.
 */
export async function loadModel(modelId: string, slot: string | null = null): Promise<boolean> {
	if (_loading.has(modelId)) return false;
	setLoading(modelId, true);
	if (slot) _pendingSlots = new Map(_pendingSlots).set(slot, modelId);
	try {
		const attempt = async (force: boolean): Promise<boolean> => {
			const outcome = await loadModelById(modelId, force);
			if (outcome.outcome === 'needs_eviction') {
				return new Promise<boolean>((resolve, reject) => {
					requestEviction({
						task: 'load_model',
						params: { model_id: modelId },
						requiredMb: outcome.required_mb,
						freeMb: outcome.free_mb,
						candidates: outcome.candidates,
						resubmit: (f) => void attempt(f).then(resolve, reject),
						cancel: () => resolve(false),
					});
				});
			}
			return true;
		};
		return await attempt(false);
	} finally {
		setLoading(modelId, false);
		if (slot && _pendingSlots.get(slot) === modelId) {
			const next = new Map(_pendingSlots);
			next.delete(slot);
			_pendingSlots = next;
		}
	}
}

/** Unload a model by catalog id (no-op when it isn't resident). */
export async function unloadModel(modelId: string): Promise<void> {
	await unloadModelById(modelId);
	// Belt-and-suspenders: refresh the snapshot directly instead of waiting
	// solely on the broadcast channel. If the broadcast fails to land on a
	// surface (the "unloaded but still shows Loaded" report), this keeps the
	// ledger authoritative; if the backend genuinely still holds the resident,
	// this fetch reflects that honestly rather than masking it.
	try {
		_snapshot = await orchestratorLedger();
	} catch {
		/* broadcast will still deliver the update */
	}
}
