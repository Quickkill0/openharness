/**
 * Channel subscriptions for Tauri streaming commands.
 *
 * All subscriptions route through the backend shim (`./backend`), which falls
 * back to the in-browser mock backend when no Tauri bridge is present (plain
 * `pnpm dev` in a browser, for fast UI iteration).
 *
 * Each function returns the live channel handle. Channels live for the app
 * lifetime; the Rust bridging task holds the broadcast receiver until the JS
 * Channel is GC'd, which only happens when the layout unmounts (i.e., never).
 */
import { subscribe, type ChannelLike } from './backend';
import type {
	ChatEvent,
	DownloadProgress,
	EngineStatus,
	GenerationEvent,
	GpuStatus,
	LedgerSnapshot,
	RuntimeStats,
	TtsStatus,
	WorkerStatus,
} from '$lib/types';

export type { ChannelLike };

/** Subscribe to live download progress events. */
export async function subscribeDownloadProgress(
	onProgress: (progress: DownloadProgress) => void,
): Promise<ChannelLike<DownloadProgress>> {
	return subscribe('subscribe_download_progress', onProgress);
}

// ---------------------------------------------------------------------------
// Engine + GPU channel subscriptions (M4)
// ---------------------------------------------------------------------------

/** Subscribe to live engine status updates. Sends the current status immediately. */
export async function subscribeEngineStatus(
	onStatus: (status: EngineStatus) => void,
): Promise<ChannelLike<EngineStatus>> {
	return subscribe('engine_subscribe_status', onStatus);
}

/** Subscribe to live GPU metrics updates (~1 Hz). */
export async function subscribeGpuMetrics(
	onStatus: (status: GpuStatus) => void,
): Promise<ChannelLike<GpuStatus>> {
	return subscribe('gpu_subscribe_metrics', onStatus);
}

// ---------------------------------------------------------------------------
// Chat channel subscription (M5)
// ---------------------------------------------------------------------------

/** Subscribe to chat events (token stream, message lifecycle). */
export async function subscribeChatEvents(
	onEvent: (event: ChatEvent) => void,
): Promise<ChannelLike<ChatEvent>> {
	return subscribe('subscribe_chat_events', onEvent);
}

// ---------------------------------------------------------------------------
// Runtime v2 channel subscriptions (M8)
// ---------------------------------------------------------------------------

/** Subscribe to Python-worker lifecycle updates. Sends the current status immediately. */
export async function subscribeWorkerStatus(
	onStatus: (status: WorkerStatus) => void,
): Promise<ChannelLike<WorkerStatus>> {
	return subscribe('subscribe_worker_status', onStatus);
}

/** Subscribe to VRAM ledger updates. Sends the current snapshot immediately. */
export async function subscribeOrchestratorLedger(
	onSnapshot: (snapshot: LedgerSnapshot) => void,
): Promise<ChannelLike<LedgerSnapshot>> {
	return subscribe('subscribe_orchestrator_ledger', onSnapshot);
}

/** Subscribe to generation job events (progress/done/error for ALL jobs). */
export async function subscribeGenerationProgress(
	onEvent: (event: GenerationEvent) => void,
): Promise<ChannelLike<GenerationEvent>> {
	return subscribe('subscribe_generation_progress', onEvent);
}

/** Subscribe to Qwen3-TTS sidecar status (incl. pin-and-fetch install
 * progress). Sends the current status immediately. */
export async function subscribeTtsStatus(
	onStatus: (status: TtsStatus) => void,
): Promise<ChannelLike<TtsStatus>> {
	return subscribe('subscribe_tts_status', onStatus);
}

/** Subscribe to live actuals for the loaded text model (decode/prompt tok/s,
 * VRAM, RAM) next to its plan's estimate. Sends the current snapshot
 * immediately, then about once a second while a model is loaded. */
export async function subscribeRuntimeStats(
	onStats: (stats: RuntimeStats) => void,
): Promise<ChannelLike<RuntimeStats>> {
	return subscribe('subscribe_runtime_stats', onStats);
}
