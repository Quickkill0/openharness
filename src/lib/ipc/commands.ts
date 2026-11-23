/**
 * Typed wrappers around Tauri invoke() calls.
 * All commands that can fail return a Result-style object rather than throwing,
 * so the UI can handle errors gracefully without try/catch at every call-site.
 *
 * Theme is NOT routed through these commands. It is the exclusive responsibility
 * of `$lib/stores/theme.ts` via tauri-plugin-store (settings.json).
 */
import { invoke } from './backend';
import type {
	AppPaths,
	Asset,
	AssetKind,
	ContextMemory,
	ContextParams,
	ConversationSummaryDto,
	CuratedPick,
	DownloadGroup,
	EngineRowDto,
	EngineStatus,
	EngineVariant,
	FitEstimate,
	GpuStatus,
	HardwareProfile,
	HfFile,
	HfModelDetails,
	HfSearchParams,
	HfSearchResponse,
	HfWhoami,
	InstalledModel,
	Job,
	LedgerSnapshot,
	LoadOutcome,
	LoadPlan,
	ModelLoadOutcome,
	NativeEngineStatus,
	MessageDto,
	PipelineCapability,
	PlanOverrides,
	QuantFit,
	Runnability,
	RunnabilityQuery,
	SearchHitDto,
	SsdMeasurement,
	SubmitOutcome,
	TextLoadRequest,
	TtsStatus,
	WorkerStatus,
	WorkerVariantSlug,
	ConversationSettings,
	GenerationJob,
	SystemMemory,
} from '$lib/types';
/** Optional load-time tuning knobs for a tuned text-model load. */
export interface LoadModelOpts {
	ctx?: number;
	ngl?: number;
	batchSize?: number;
	threads?: number;
	flashAttn?: boolean;
	kvCacheType?: string;
	mmap?: boolean;
	mlock?: boolean;
	/** Planner choices such as the engine; ignored when `ngl` is set. */
	overrides?: PlanOverrides;
}

// ---------------------------------------------------------------------------
// Settings (key/value store backed by SQLite — for non-theme settings)
// ---------------------------------------------------------------------------

export async function getSetting(key: string): Promise<string | null> {
	return invoke<string | null>('get_setting', { key });
}

export async function setSetting(key: string, value: string): Promise<void> {
	return invoke<void>('set_setting', { key, value });
}

// ---------------------------------------------------------------------------
// HF Token (stored in Windows Credential Manager via keyring crate)
// ---------------------------------------------------------------------------

export async function getHfToken(): Promise<string | null> {
	return invoke<string | null>('get_hf_token');
}

export async function setHfToken(token: string): Promise<void> {
	return invoke<void>('set_hf_token', { token });
}

export async function clearHfToken(): Promise<void> {
	return invoke<void>('clear_hf_token');
}

export async function validateHfToken(): Promise<HfWhoami> {
	return invoke<HfWhoami>('validate_hf_token');
}

// ---------------------------------------------------------------------------
// App paths
// ---------------------------------------------------------------------------

export async function getAppPaths(): Promise<AppPaths> {
	return invoke<AppPaths>('app_paths');
}

/**
 * Returns true exactly once if the Hangar→OpenHarness data migration ran this
 * launch. Used to show a one-off toast on boot.
 */
export async function consumeMigrationNotice(): Promise<boolean> {
	return invoke<boolean>('consume_migration_notice');
}

// ---------------------------------------------------------------------------
// HuggingFace model browser (M2)
// ---------------------------------------------------------------------------

export async function hfSearch(params: HfSearchParams): Promise<HfSearchResponse> {
	return invoke<HfSearchResponse>('hf_search', { params });
}

export async function hfModelDetails(repoId: string): Promise<HfModelDetails> {
	return invoke<HfModelDetails>('hf_model_details', { repoId });
}

export async function hfModelTree(repoId: string, revision: string): Promise<HfFile[]> {
	return invoke<HfFile[]>('hf_model_tree', { repoId, revision });
}

// ---------------------------------------------------------------------------
// Downloads (M3)
// ---------------------------------------------------------------------------

export async function startDownload(
	repoId: string,
	revision: string,
	file: HfFile,
): Promise<string> {
	return invoke<string>('start_download', { repoId, revision, file });
}

export async function pauseDownload(id: string): Promise<void> {
	return invoke<void>('pause_download', { id });
}

export async function resumeDownload(id: string): Promise<void> {
	return invoke<void>('resume_download', { id });
}

export async function cancelDownload(id: string): Promise<void> {
	return invoke<void>('cancel_download', { id });
}

export async function deleteDownload(id: string): Promise<void> {
	return invoke<void>('delete_download', { id });
}

export async function listDownloads(): Promise<Job[]> {
	return invoke<Job[]>('list_downloads');
}

// ---------------------------------------------------------------------------
// Repo download groups (M9 §9.2)
// ---------------------------------------------------------------------------

/** Start a whole-repo download (one job per file). Returns the group id. */
export async function startRepoDownload(opts: {
	repoId: string;
	revision: string;
	files: HfFile[];
	format: string;
	task?: string | null;
}): Promise<string> {
	return invoke<string>('start_repo_download', {
		repoId: opts.repoId,
		revision: opts.revision,
		files: opts.files,
		format: opts.format,
		task: opts.task,
	});
}

export async function pauseDownloadGroup(id: string): Promise<void> {
	return invoke<void>('pause_download_group', { id });
}

export async function resumeDownloadGroup(id: string): Promise<void> {
	return invoke<void>('resume_download_group', { id });
}

export async function cancelDownloadGroup(id: string): Promise<void> {
	return invoke<void>('cancel_download_group', { id });
}

export async function deleteDownloadGroup(id: string): Promise<void> {
	return invoke<void>('delete_download_group', { id });
}

export async function listDownloadGroups(): Promise<DownloadGroup[]> {
	return invoke<DownloadGroup[]>('list_download_groups');
}

// ---------------------------------------------------------------------------
// Store intelligence (M9 §9.4)
// ---------------------------------------------------------------------------

/** Batch runnability resolution — one verdict per query, in order. */
export async function storeRunnability(items: RunnabilityQuery[]): Promise<Runnability[]> {
	return invoke<Runnability[]>('store_runnability', { items });
}

/** Curated picks, optionally filtered to one tab's task. */
export async function storeCuratedPicks(task?: string): Promise<CuratedPick[]> {
	return invoke<CuratedPick[]>('store_curated_picks', { task });
}

/** Per-GGUF-file VRAM fit verdicts (§9.6). Empty when no GPU is detected. */
export async function storeQuantFits(files: HfFile[]): Promise<QuantFit[]> {
	return invoke<QuantFit[]>('store_quant_fits', { files });
}

// ---------------------------------------------------------------------------
// Local model catalog (M3)
// ---------------------------------------------------------------------------

export async function listLocalModels(): Promise<InstalledModel[]> {
	return invoke<InstalledModel[]>('list_local_models');
}

export async function deleteLocalModel(id: string): Promise<void> {
	return invoke<void>('delete_local_model', { id });
}

// ---------------------------------------------------------------------------
// Engine (M4)
// ---------------------------------------------------------------------------

export async function engineStatus(): Promise<EngineStatus> {
	return invoke<EngineStatus>('engine_status');
}

/**
 * Return the persisted active engine row, or null if no engine is installed.
 * Distinguishes "installed but idle" from "never installed" — the live
 * EngineStatus union reports `idle` for both.
 */
export async function nativeEngineInfo(retry = false): Promise<NativeEngineStatus> {
	return invoke('native_engine_info', { retry });
}

export async function nativeEngineInstall(): Promise<void> {
	return invoke('native_engine_install');
}

export async function engineActive(): Promise<EngineRowDto | null> {
	return invoke<EngineRowDto | null>('engine_active');
}

export async function engineInstall(variant?: EngineVariant): Promise<EngineRowDto> {
	return invoke<EngineRowDto>('engine_install', { variant });
}

export async function engineSwitchVariant(variant: EngineVariant): Promise<void> {
	return invoke<void>('engine_switch_variant', { variant });
}

export async function engineUnloadModel(): Promise<void> {
	return invoke<void>('engine_unload_model');
}

/** Last ~1000 stdout/stderr lines from the running llama-server (empty when no model is loaded). */
export async function engineLogs(): Promise<string[]> {
	return invoke<string[]>('engine_logs');
}

// ---------------------------------------------------------------------------
// GPU (M4)
// ---------------------------------------------------------------------------

export async function gpuStatus(): Promise<GpuStatus> {
	return invoke<GpuStatus>('gpu_status');
}

/** Physical RAM total/used; null when the OS query is unavailable. */
export async function systemMemory(): Promise<SystemMemory | null> {
	return invoke<SystemMemory | null>('system_memory');
}

/** Open Explorer at an installed model's folder / weights file. */
export async function revealModel(modelId: string): Promise<void> {
	return invoke<void>('reveal_model', { modelId });
}

export async function estimateModelFit(modelId: string, ctx?: number): Promise<FitEstimate> {
	return invoke<FitEstimate>('estimate_model_fit', { modelId, ctx });
}

// ---------------------------------------------------------------------------
// Chat (M5)
// ---------------------------------------------------------------------------

export async function listConversations(): Promise<ConversationSummaryDto[]> {
	return invoke<ConversationSummaryDto[]>('list_conversations');
}

export async function createConversation(): Promise<ConversationSummaryDto> {
	return invoke<ConversationSummaryDto>('create_conversation');
}

export async function getConversation(id: string): Promise<MessageDto[]> {
	return invoke<MessageDto[]>('get_conversation', { id });
}

export async function deleteConversation(id: string): Promise<void> {
	return invoke<void>('delete_conversation', { id });
}

export async function sendMessage(
	convId: string,
	content: string): Promise<void> {
	return invoke<void>('send_message', { convId, content });
}

export async function cancelStream(convId: string): Promise<void> {
	return invoke<void>('cancel_stream', { convId });
}

/** Delete a message and everything after it (delete-from-here). */
export async function chatTruncateFrom(convId: string, fromSeq: number): Promise<void> {
	return invoke<void>('chat_truncate_from', { convId, fromSeq });
}

/** Regenerate an assistant message: truncate from its seq, then re-stream. */
export async function chatRegenerate(convId: string, seq: number): Promise<void> {
	return invoke<void>('chat_regenerate', { convId, seq });
}

export async function searchMessages(query: string): Promise<SearchHitDto[]> {
	return invoke<SearchHitDto[]>('search_messages', { query });
}

export async function getConversationSettings(convId: string): Promise<ConversationSettings> {
	return invoke<ConversationSettings>('get_conversation_settings', { convId });
}

export async function setConversationSettings(
	convId: string,
	settings: ConversationSettings,
): Promise<void> {
	return invoke<void>('set_conversation_settings', { convId, settings });
}

// ---------------------------------------------------------------------------
// Python worker (M8)
// ---------------------------------------------------------------------------

export async function workerStatus(): Promise<WorkerStatus> {
	return invoke<WorkerStatus>('worker_status');
}

/** Install (or rebuild) the worker venv. Resolves when the install finishes. */
export async function workerInstall(variant: WorkerVariantSlug): Promise<void> {
	return invoke<void>('worker_install', { variant });
}

/** Start the worker process. Resolves with its pipeline capabilities. */
export async function workerStart(): Promise<PipelineCapability[]> {
	return invoke<PipelineCapability[]>('worker_start');
}

export async function workerStop(): Promise<void> {
	return invoke<void>('worker_stop');
}

/** Last ~1000 stderr lines from the running worker (empty when stopped). */
export async function workerLogs(): Promise<string[]> {
	return invoke<string[]>('worker_logs');
}

// ---------------------------------------------------------------------------
// Orchestrator (M8 §8.4)
// ---------------------------------------------------------------------------

/**
 * Load a text model through the orchestrator's VRAM gate. Returns
 * `needs_eviction` (nothing loaded) when it would only fit after freeing a
 * resident — unload a candidate and retry, or re-send with `force: true`.
 */
export async function orchestratorLoadModel(req: TextLoadRequest): Promise<LoadOutcome> {
	return invoke<LoadOutcome>('orchestrator_load_model', { req });
}

/** Unload a resident by ledger id: `"text"` or `"pipeline:<task>"`. */
export async function orchestratorUnload(residentId: string): Promise<void> {
	return invoke<void>('orchestrator_unload', { residentId });
}

/**
 * Unload a resident only if it still holds `expectedModelId` (null = an
 * uncorrelated resident). A no-op when the slot is already empty; rejects
 * with a plain message when another model has taken the slot since the UI
 * last looked, so a stale × can't unload the newcomer.
 */
export async function orchestratorUnloadIf(
	residentId: string,
	expectedModelId: string | null,
): Promise<void> {
	return invoke<void>('orchestrator_unload_if', { residentId, expectedModelId,
	});
}

/**
 * Unified load-first flow (M16 W1): load any runnable model into VRAM,
 * routed by its capability record (llama-server / worker preload / qwen-tts).
 * `needs_eviction` means nothing was loaded — present the eviction choice.
 */
export async function loadModelById(
	modelId: string,
	force?: boolean,
	plan?: LoadPlan,
	overrides?: PlanOverrides,
): Promise<ModelLoadOutcome> {
	return invoke<ModelLoadOutcome>('load_model', {
		modelId,
		force,
		plan,
		overrides,
	});
}

// ---------------------------------------------------------------------------
// Load planner
// ---------------------------------------------------------------------------

/** GPU, RAM, the drive holding the models folder and the planner's bandwidths. */
export async function hardwareProfile(): Promise<HardwareProfile> {
	return invoke<HardwareProfile>('hardware_profile');
}

/**
 * Measure the sequential read speed of the drive holding a model (or the
 * models folder). Reads at most 4 GB of one model file; explicit user action
 * only, never automatic. The result feeds later plans.
 */
export async function benchmarkSsd(modelId?: string): Promise<SsdMeasurement> {
	return invoke<SsdMeasurement>('benchmark_ssd', { modelId });
}

/**
 * Where a text model's weights would go (graphics memory / system memory /
 * SSD), the llama-server flags that realise it and a speed estimate. No
 * overrides = the auto best settings the one-click Load uses.
 */
export async function planLoad(modelId: string, overrides?: PlanOverrides): Promise<LoadPlan> {
	return invoke<LoadPlan>('plan_load', { modelId, overrides });
}

/**
 * What a context costs for a model (KV, sliding-window, indexer and
 * recurrent-state memory plus compute buffers), at `params.ctx` or at each
 * length in `contexts`. Same function the planner uses.
 */
export async function contextMemory(
	modelId: string,
	params?: Partial<ContextParams>,
	contexts?: number[],
): Promise<ContextMemory[]> {
	return invoke<ContextMemory[]>('context_memory', {
		modelId,
		params: params ? { ...DEFAULT_CONTEXT_PARAMS, ...params } : undefined,
		contexts,
	});
}

/** Defaults of planner::context::ContextParams. */
export const DEFAULT_CONTEXT_PARAMS: ContextParams = {
	ctx: 4096,
	kv_k: 'f16',
	kv_v: 'f16',
	n_parallel: 1,
	ubatch: 512,
	experts_on_host: false,
};

/** Unload a model by catalog id (no-op when it isn't resident). */
export async function unloadModelById(modelId: string): Promise<void> {
	return invoke<void>('unload_model', { modelId });
}

/** One-shot VRAM ledger snapshot: GPU totals + everything resident. */
export async function orchestratorLedger(): Promise<LedgerSnapshot> {
	return invoke<LedgerSnapshot>('orchestrator_ledger');
}

// ---------------------------------------------------------------------------
// Generation (M8 §8.5)
// ---------------------------------------------------------------------------

export async function generateImage(
	params: Record<string, unknown>,
	force?: boolean,
): Promise<SubmitOutcome> {
	return invoke<SubmitOutcome>('generate_image', { params, force });
}

export async function generateVideo(
	params: Record<string, unknown>,
	force?: boolean,
): Promise<SubmitOutcome> {
	return invoke<SubmitOutcome>('generate_video', { params, force });
}

export async function synthesizeSpeech(
	params: Record<string, unknown>,
	force?: boolean,
): Promise<SubmitOutcome> {
	return invoke<SubmitOutcome>('synthesize_speech', { params, force });
}

export async function transcribeAudio(
	params: Record<string, unknown>,
	force?: boolean,
): Promise<SubmitOutcome> {
	return invoke<SubmitOutcome>('transcribe_audio', { params, force });
}

/** A job the backend is running (or has reserved and is about to start). */
export interface ActiveGenerationJob {
	id: string;
	task: string;
}

/**
 * Every running job, with no history cutoff, plus jobs whose reservation is
 * granted but whose row isn't written yet. Startup healing has already closed
 * rows orphaned by a crash, so each entry ends with a real terminal event.
 */
export async function activeGenerationJobs(): Promise<ActiveGenerationJob[]> {
	return invoke<ActiveGenerationJob[]>('active_generation_jobs');
}

/** Job history, newest first. `task` filters to one pipeline. */
export async function listGenerationJobs(
	task?: string,
	limit?: number): Promise<GenerationJob[]> {
	return invoke<GenerationJob[]>('list_generation_jobs', { task, limit });
}

/** Request cancellation; the terminal `cancelled` status lands via job sync. */
export async function cancelGenerationJob(jobId: string): Promise<void> {
	return invoke<void>('cancel_generation_job', { jobId });
}

/** Delete a job row; `deleteFiles` also removes its outputs from disk. */
export async function deleteGenerationJob(jobId: string, deleteFiles?: boolean): Promise<void> {
	return invoke<void>('delete_generation_job', { jobId, deleteFiles });
}

/**
 * Persist a microphone capture (base64 audio from MediaRecorder) so the
 * transcribe pipeline can read it. Returns the absolute file path (M11 §11.3).
 */
export async function saveVoiceCapture(dataB64: string, ext?: string): Promise<string> {
	return invoke<string>('save_voice_capture', { dataB64, ext });
}

// ---------------------------------------------------------------------------
// Qwen3-TTS engine
// ---------------------------------------------------------------------------

/** Sidecar status; `ready` carries the port and the talker's speaker names. */
export async function ttsStatus(): Promise<TtsStatus> {
	return invoke<TtsStatus>('tts_status');
}

/** Whether the qwentts engine files are present on disk. */
export async function ttsInstalled(): Promise<boolean> {
	return invoke<boolean>('tts_installed');
}

/** Whether the one-shot qwen-tts CLI (voice cloning, base talkers) is present. */
export async function ttsCliInstalled(): Promise<boolean> {
	return invoke<boolean>('tts_cli_installed');
}

/** Stop the sidecar and release its VRAM (ledger updates too). */
export async function ttsUnload(): Promise<void> {
	return invoke<void>('tts_unload');
}

/**
 * Provision the qwentts runtime via pin-and-fetch. Resolves when the install
 * finishes; progress streams on `subscribeTtsStatus`. NOTE: the binaries are
 * not hosted yet, so a real call rejects with an honest "not hosted" error.
 */
export async function ttsInstall(): Promise<void> {
	return invoke<void>('tts_install');
}

/** Recent tts-server log lines. */
export async function ttsLogs(): Promise<string[]> {
	return invoke<string[]>('tts_logs');
}

// ---------------------------------------------------------------------------
// Asset library (M10)
// ---------------------------------------------------------------------------

/** Library contents, newest first. `kind` filters to image/video/audio/text. */
export async function listAssets(
	kind?: AssetKind,
	limit?: number,
	offset?: number,
): Promise<Asset[]> {
	return invoke<Asset[]>('list_assets', { kind, limit, offset });
}

/** Full-text search over prompt/model/task (prefix matching). */
export async function searchAssets(query: string, limit?: number): Promise<Asset[]> {
	return invoke<Asset[]>('search_assets', { query, limit });
}

/** Delete an asset row; `deleteFile` also removes the artifact from disk. */
export async function deleteAsset(assetId: string, deleteFile?: boolean): Promise<void> {
	return invoke<void>('delete_asset', { assetId, deleteFile });
}

/** Open Windows Explorer with the asset's file selected. */
export async function revealAsset(assetId: string): Promise<void> {
	return invoke<void>('reveal_asset', { assetId });
}