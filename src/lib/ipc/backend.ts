/**
 * IPC backend shim.
 *
 * In the Tauri webview, `invoke` and `subscribe` route straight to the Rust
 * backend. In a plain browser (`pnpm dev` + http://localhost:1420) there is no
 * Tauri bridge, so calls route to the in-memory mock backend (`./mock`) —
 * realistic canned data + streaming chat and Studio simulators. This lets the UI be
 * iterated in a normal browser with HMR, no Rust build in the loop.
 *
 * The mock module is loaded lazily so production (Tauri) bundles never pull in
 * the fixture data.
 */
import { invoke as tauriInvoke, Channel, convertFileSrc } from '@tauri-apps/api/core';

export const IS_TAURI = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

/**
 * Webview-loadable URL for a local file (asset protocol). In the browser
 * mock, images get a deterministic SVG placeholder so galleries render;
 * audio/video return an empty source (players show their idle state).
 */
export function fileSrc(path: string): string {
	if (IS_TAURI) return convertFileSrc(path);
	const name = path.split(/[\\/]/).pop() ?? path;
	if (/\.(png|jpe?g|webp|bmp|gif)$/i.test(name)) {
		// Hash the name into a hue so mock thumbnails are distinguishable.
		let h = 0;
		for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360;
		const svg =
			`<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">` +
			`<rect width="512" height="512" fill="hsl(${h},35%,22%)"/>` +
			`<text x="50%" y="50%" fill="hsl(${h},45%,70%)" font-family="monospace" font-size="20" text-anchor="middle">${name}</text>` +
			`</svg>`;
		return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
	}
	if (/\.txt$/i.test(name)) {
		const text =
			'[00:00.000 → 00:14.200] Welcome back to the weekly standup.\n' +
			'[00:14.200 → 00:27.900] The downloader actor landed resumable chunked fetches last sprint.\n' +
			'[00:27.900 → 00:42.500] Next up is the studio milestone — image, speech, and video generation.';
		return `data:text/plain;charset=utf-8,${encodeURIComponent(text)}`;
	}
	return '';
}

/** Minimal common surface of a real Tauri Channel and the mock subscription. */
export interface ChannelLike<T> {
	onmessage: (msg: T) => void;
}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
	if (IS_TAURI) return tauriInvoke<T>(cmd, args);
	const { mockInvoke } = await import('./mock');
	return mockInvoke<T>(cmd, args);
}

/**
 * Strong references to every Channel we hand to Tauri. The Tauri IPC bridge
 * does NOT keep the JS `Channel` alive on its own, so a caller that discards
 * the returned handle (as every status store does) lets V8 garbage-collect the
 * channel — which severs the Rust-side forwarder and silently freezes the
 * stream (observed as "the ledger still shows a model loaded after unloading"
 * and other stale-status bugs). Every subscription in this app is an
 * app-lifetime singleton, so retaining each channel here for the process
 * lifetime is correct rather than a leak, and removes a footgun no caller can
 * be relied upon to avoid.
 */
const retainedChannels = new Set<unknown>();

/**
 * Subscribe to a Channel-streaming command (`subscribe_*` / `*_subscribe_*`).
 * The channel is retained internally for the app lifetime so the Rust-side
 * forwarder is never severed by GC; the returned handle is still provided for
 * callers that want it.
 */
export async function subscribe<T>(
	cmd: string,
	onmessage: (msg: T) => void,
): Promise<ChannelLike<T>> {
	if (IS_TAURI) {
		const channel = new Channel<T>();
		channel.onmessage = onmessage;
		retainedChannels.add(channel);
		await tauriInvoke(cmd, { channel });
		return channel;
	}
	const { mockSubscribe } = await import('./mock');
	return mockSubscribe(cmd, onmessage);
}
