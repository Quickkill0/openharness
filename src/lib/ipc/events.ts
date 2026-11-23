/**
 * Typed listen() helpers for Tauri events.
 * Empty in M1 — events are used starting M3 (download progress).
 */
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export type { UnlistenFn };

/**
 * Generic typed listen wrapper.
 * Returns the unlisten function so the caller can clean up in onDestroy().
 */
export async function listenEvent<T>(
	event: string,
	handler: (payload: T) => void
): Promise<UnlistenFn> {
	return listen<T>(event, (e) => handler(e.payload));
}
