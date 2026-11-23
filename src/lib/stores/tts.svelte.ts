/**
 * Shared Qwen3-TTS sidecar status store.
 *
 * Mirrors `worker.svelte.ts`: the Rust side keeps a single
 * `subscribe_tts_status` forwarder slot, so every consumer reads this one
 * store and exactly one channel subscription is ever made. Drives the
 * Qwen3-TTS engine card on /settings/engines (status + pin-and-fetch install
 * progress).
 */
import { subscribeTtsStatus } from '$lib/ipc/channels';
import type { TtsStatus } from '$lib/types';

let _status = $state<TtsStatus>({ phase: 'idle' });
let wired = false;

export const ttsStore = {
	get status(): TtsStatus {
		return _status;
	},
};

/** Idempotent: subscribe the single tts-status channel. */
export async function initTtsStore(): Promise<void> {
	if (wired) return;
	wired = true;
	await subscribeTtsStatus((s) => {
		_status = s;
	});
}
