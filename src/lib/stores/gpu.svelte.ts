/**
 * Svelte 5 rune-based GPU store.
 *
 * Tracks the current GpuStatus broadcast by the backend GPU actor (1 Hz).
 * Call initGpuStore() once from the root layout onMount.
 */
import { gpuStatus as fetchStatus } from '$lib/ipc/commands';
import { subscribeGpuMetrics } from '$lib/ipc/channels';
import type { GpuStatus } from '$lib/types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _status = $state<GpuStatus>({ kind: 'no_gpu' });
let _initialized = $state(false);

// ---------------------------------------------------------------------------
// Read-only store object
// ---------------------------------------------------------------------------

export const gpuStore = {
	get status() { return _status; },
	get hasGpu() { return _status.kind === 'available'; },
	get vramTotal() { return _status.kind === 'available' ? _status.vram_total_mb : 0; },
	get vramUsed() { return _status.kind === 'available' ? _status.vram_used_mb : 0; },
	get vramFree() { return _status.kind === 'available' ? _status.vram_free_mb : 0; },
	get name() { return _status.kind === 'available' ? _status.name : 'No GPU'; },
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

export async function initGpuStore(): Promise<void> {
	if (_initialized) return;
	try {
		_status = await fetchStatus();
		await subscribeGpuMetrics((s) => { _status = s; });
		_initialized = true;
	} catch (e) {
		console.error('gpu store init failed:', e);
	}
}
