/**
 * Svelte 5 rune-based system-memory store (Workbench dock).
 *
 * `system_memory` is one cheap OS call with no backend actor behind it, so
 * this store polls it every few seconds rather than holding a channel open.
 * `memory` stays null until the first answer, and stays null for good when
 * the platform can't report it — the dock hides the meter in that case.
 */
import { systemMemory } from '$lib/ipc/commands';
import type { SystemMemory } from '$lib/types';

const POLL_MS = 3000;

let _memory = $state<SystemMemory | null>(null);
let _timer: ReturnType<typeof setInterval> | null = null;

export const systemStore = {
	get memory() {
		return _memory;
	},
};

async function refresh(): Promise<void> {
	try {
		_memory = await systemMemory();
	} catch {
		// Keep the last good reading; a transient failure shouldn't blank the dock.
	}
}

/** Idempotent: start polling. Call once from the root layout. */
export function initSystemStore(): void {
	if (_timer) return;
	void refresh();
	_timer = setInterval(() => void refresh(), POLL_MS);
}
