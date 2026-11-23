/**
 * Shared cache of `estimate_model_fit` results for the Models page: the
 * table's Fit column, the inspector, and the dock's "will load" preview all
 * read the same estimate instead of each firing its own request.
 *
 * Each estimate is requested at the context a plain Load would launch that
 * model with (`defaultCtx` in modelMeta, mirroring the orchestrator's
 * `min(context_train, 8192)`). With a context the backend folds the context
 * cache into both the estimate and the verdict, so Fits / Tight / Won't fit
 * describe the load the Load button would actually do.
 *
 * Free VRAM moves whenever something loads, unloads or another app takes
 * graphics memory; the Models page calls `invalidateFits()` for each case and
 * on every visit.
 */
import { estimateModelFit } from '$lib/ipc/commands';
import type { FitEstimate } from '$lib/types';

export interface FitTarget {
	id: string;
	/** Context length the estimate is for. */
	ctx: number;
}

type Entry = { ctx: number } & ({ kind: 'ok'; fit: FitEstimate } | { kind: 'error'; message: string });

let _entries = $state<Record<string, Entry>>({});
const _inflight = new Map<string, { ctx: number; p: Promise<void> }>();
/** Bumped on invalidate so late answers from an older generation are dropped. */
let _generation = 0;

export const fitStore = {
	get(modelId: string): FitEstimate | null {
		const e = _entries[modelId];
		return e?.kind === 'ok' ? e.fit : null;
	},
	/** The context length the cached estimate was computed at. */
	ctx(modelId: string): number | null {
		return _entries[modelId]?.ctx ?? null;
	},
	error(modelId: string): string | null {
		const e = _entries[modelId];
		return e?.kind === 'error' ? e.message : null;
	},
	has(modelId: string): boolean {
		return modelId in _entries;
	},
};

function fetchFit(id: string, ctx: number): Promise<void> {
	const gen = _generation;
	const p = estimateModelFit(id, ctx)
		.then((fit) => {
			if (gen === _generation) _entries = { ..._entries, [id]: { kind: 'ok', fit, ctx } };
		})
		.catch((e) => {
			if (gen === _generation) {
				_entries = { ..._entries, [id]: { kind: 'error', message: String(e), ctx } };
			}
		})
		.finally(() => {
			if (_inflight.get(id)?.p === p) _inflight.delete(id);
		});
	_inflight.set(id, { ctx, p });
	return p;
}

/** Fetch an estimate unless one for this context is cached or in flight. */
export function ensureFit(modelId: string, ctx: number): Promise<void> {
	if (_entries[modelId]?.ctx === ctx) return Promise.resolve();
	const pending = _inflight.get(modelId);
	if (pending?.ctx === ctx) return pending.p;
	return fetchFit(modelId, ctx);
}

/** Ask again after a failed check: drops the cached error, so the row reads as checking. */
export function retryFit(modelId: string, ctx: number): Promise<void> {
	if (_entries[modelId]?.kind === 'error') {
		const { [modelId]: _failed, ...rest } = _entries;
		_entries = rest;
	}
	return ensureFit(modelId, ctx);
}

/**
 * Drop every cached estimate (free VRAM changed) and refetch the given
 * targets. Old values stay visible until the new ones arrive, so the table
 * doesn't flash to empty on every load/unload.
 */
export function invalidateFits(refetch: FitTarget[]): void {
	_generation++;
	_inflight.clear();
	const previous = _entries;
	for (const t of refetch) void fetchFit(t.id, t.ctx);
	// Keep stale values only for ids being refetched.
	const kept: Record<string, Entry> = {};
	for (const t of refetch) if (previous[t.id]) kept[t.id] = previous[t.id];
	_entries = kept;
}
