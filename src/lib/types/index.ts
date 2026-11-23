/** Application theme setting */
export type Theme = 'light' | 'dark' | 'system';

/** Persisted application settings (non-secret) */
export interface Settings {
	theme: Theme;
}

/** App path information returned by app_paths command */
export interface AppPaths {
	app_data_dir: string;
	db_path: string;
	models_dir: string;
	transcripts_dir: string;
	engine_dir: string;
}

/** Result of HF token validation */
export interface HfWhoami {
	name: string;
	fullname?: string;
	email?: string;
}

/** Tauri command result wrapper */
export type CommandResult<T> = { ok: true; data: T }
	| { ok: false; error: string };

// ---------------------------------------------------------------------------
// HuggingFace model browser types (M2)
// Mirror the Rust structs in src-tauri/src/models/hf.rs
// ---------------------------------------------------------------------------

export interface HfModelCard {
	id: string;
	author?: string;
	pipeline_tag?: string;
	library_name?: string;
	tags: string[];
	downloads: number;
	likes: number;
	trending_score?: number;
	gated: boolean;
	private: boolean;
	last_modified?: string;
	created_at?: string;
}

export interface HfFile {
	path: string;
	/** Resolved byte size (from LFS metadata when available) */
	size: number;
	sha256?: string;
	is_lfs: boolean;
}

export interface HfGgufMeta {
	architecture?: string;
	context_length?: number;
	total_params?: number;
}

/** Repo layout classification (M9) — mirrors models/layout.rs */
export type LayoutKind =
	'gguf'
	| 'safetensors-single'
	| 'safetensors-shards'
	| 'diffusers-dir'
	| 'unknown';

export interface RepoLayout {
	kind: LayoutKind;
	/** Rel-paths a "download repo" action fetches (empty for gguf/unknown) */
	download_files: string[];
}

export interface HfModelDetails {
	card: HfModelCard;
	/** Raw README markdown, truncated to 50 KB */
	readme?: string;
	gguf?: HfGgufMeta;
	siblings: HfFile[];
	layout: RepoLayout;
}

export interface HfSearchParams {
	query?: string;
	/** text-generation | embedding | vision-text | speech | image | any */
	capability?: string;
	/** Raw HF pipeline_tag passthrough — supersedes capability when set */
	task?: string;
	/** gguf | transformers | diffusers | safetensors | any */
	library?: string;
	/** trending | downloads | likes | recent */
	sort?: string;
	/** 1-based page number */
	page: number;
	per_page: number;
}

export interface HfSearchResponse {
	results: HfModelCard[];
	page: number;
	has_next: boolean;
}

// ---------------------------------------------------------------------------
// Downloader types (M3)
// Mirror the Rust structs in src-tauri/src/downloader/
// ---------------------------------------------------------------------------

export type JobStatus = 'queued'
	| 'running'
	| 'paused'
	| 'done'
	| 'failed'
	| 'cancelled';

export interface ChunkState {
	start: number;
	end: number;
	completed: number;
}

export interface Job {
	id: string;
	repo_id: string;
	revision: string;
	file: string;
	dest_path: string;
	total_bytes: number;
	downloaded: number;
	chunks: ChunkState[];
	status: JobStatus;
	error?: string;
	expected_sha256?: string;
	/** Parent download-group id for repo downloads; null for single-GGUF jobs. */
	group_id?: string | null;
	created_at: number;
	updated_at: number;
}

/** A repo download group — mirrors db::downloads::GroupRow (M9 §9.2). */
export interface DownloadGroup {
	id: string;
	repo_id: string;
	revision: string;
	/** gguf | safetensors | diffusers-dir */
	format: string;
	task: string | null;
	created_at: number;
}

export type DownloadPhase = 'downloading' | 'verifying' | 'finalizing';

export interface DownloadProgress {
	id: string;
	status: JobStatus;
	/** Current failure reason; null clears the previous attempt's error. */
	error?: string | null;
	downloaded: number;
	total: number;
	bytes_per_sec: number;
	eta_secs?: number;
	/** Phase of the running job — flips to 'verifying' / 'finalizing' after all bytes are written. */
	phase?: DownloadPhase;
}

// ---------------------------------------------------------------------------
// Store intelligence types (M9 §9.4)
// Mirror the Rust types in src-tauri/src/store.rs
// ---------------------------------------------------------------------------

/** Which local runtime can run a repo. */
export type StoreRuntime = 'llama-cpp' | 'worker';

/** Runnability verdict for one search row / detail pane. */
export interface Runnability {
	runnable: boolean;
	runtime: StoreRuntime | null;
	/** Worker pipeline task when runtime === 'worker' (e.g. "text_to_image"). */
	pipeline: string | null;
	/** Pipeline deps currently importable (always true for llama.cpp). */
	available: boolean;
	/** Honest badge/tooltip copy. */
	reason: string;
}

/** One row to resolve — card-level signals plus optional layout sharpening. */
export interface RunnabilityQuery {
	repo_id?: string;
	/** HF pipeline_tag. */
	task?: string;
	/** HF library_name. */
	library?: string;
	tags?: string[];
	layout?: LayoutKind;
}

/** One curated pick — mirrors store::CuratedPick. */
export interface CuratedPick {
	repo_id: string;
	display_name: string;
	/** HF pipeline_tag-style task — matches the store tab filter. */
	task: string;
	/** gguf | diffusers-dir | ctranslate2-dir | kokoro-repo */
	format: string;
	/** Recommended file for single-file (GGUF) picks; null for dir layouts. */
	recommended_file: string | null;
	min_vram_mb: number;
	blurb: string;
	/** Whether the pick fits the detected GPU's total VRAM. */
	fits: boolean;
}

/** Per-GGUF-file fit verdict (M9 §9.6) — mirrors store::QuantFit. */
export interface QuantFit {
	path: string;
	/** Parsed quant label (e.g. "Q4_K_M"); null when unrecognizable. */
	label: string | null;
	size: number;
	/** Coarse pre-download VRAM estimate in MiB (no GGUF metadata yet). */
	estimated_mb: number;
	verdict: FitVerdict;
	/** The recommended one-click pick — at most one entry per repo. */
	best: boolean;
	/** Part of a multi-file split GGUF; never `best`. */
	split: boolean;
	set_key: string | null;
	set_size: number | null;
	part_count: number | null;
}

// ---------------------------------------------------------------------------
// Installed model catalog types (M3)
// ---------------------------------------------------------------------------

export type Capability =
	'text-generation'
	| 'embedding'
	| 'vision-text'
	| 'speech'
	| 'image-gen'
	| 'other';

/** Catalog storage format (M9 §9.1). */
export type ModelFormat = 'gguf' | 'safetensors' | 'diffusers-dir';

// ---------------------------------------------------------------------------
// Model capability records (M15) — mirror src-tauri/src/models/capabilities.rs
// ---------------------------------------------------------------------------

/** Engine family that runs a model. */
export type RuntimeKind = 'llama-server' | 'worker' | 'qwen-tts';

/** Workflows a model can power, in display order. */
export type Workflow = 'chat'
	| 'image'
	| 'speech'
	| 'transcribe'
	| 'video'
	| 'embeddings';

export interface VoiceInfo {
	id: string;
	label: string;
}

/** TTS descriptor: which engine speaks and which controls to render. */
export interface SpeechCaps {
	/** 'kokoro' | 'qwen3-tts' | 'vibevoice' — matches the `engine` job param. */
	engine: string;
	/** Static voice list; UI may override with a live one (qwentts /v1/voices). Empty for cloning engines. */
	voices: VoiceInfo[];
	supports_speed: boolean;
	supports_instructions: boolean;
	/** Clones a voice from a reference audio sample (VibeVoice, Qwen3-TTS base talkers). */
	supports_voice_cloning: boolean;
	/** Cloning also needs a transcript of the reference sample (Qwen3-TTS `--ref-text`). */
	cloning_requires_ref_text: boolean;
}

/** Resolved capability record attached to every catalog model (computed, never stored). */
export interface ModelCaps {
	runtime: RuntimeKind | null;
	/** Worker pipeline name when runtime === 'worker'. */
	pipeline: string | null;
	workflows: Workflow[];
	speech: SpeechCaps | null;
	/** Diffusion architecture hint ('sdxl' | 'sd' | 'flux') for image models. */
	image_arch: string | null;
	/**
	 * True when this is a single-file GGUF diffusion transformer (DiT only)
	 * that needs companion VAE + text-encoder files to run.
	 */
	gguf_diffusion: boolean;
	/** Companion-pack key ('qwen_image' | 'ltxv' | 'wan') when gguf_diffusion. */
	companion_arch: string | null;
	/** Honest copy explaining why the model can't run, when it can't. */
	reason: string | null;
}

export interface InstalledModel {
	id: string;
	repo_id: string;
	revision: string;
	display_name: string;
	architecture?: string;
	capability: Capability;
	format: ModelFormat;
	/** HF pipeline_tag-style task ('text-generation', 'text-to-image', …). */
	task: string | null;
	/** Root directory for dir-layout models; null for single-blob GGUF. */
	local_path: string | null;
	params_b?: number;
	quantization?: string;
	context_train?: number;
	installed_at: number;
	primary_file_path: string;
	primary_file_size: number;
	/**
	 * The primary file's path inside the source repo. For single GGUFs this is
	 * the original HF filename; the file on disk is named `<sha256>.gguf`.
	 */
	primary_rel_path: string | null;
	/** SHA-256 of the primary file when known. */
	primary_sha256: string | null;
	/** Number of registered files (1 for single-blob GGUF). */
	file_count: number;
	/** Sum of all registered file sizes in bytes. */
	total_size: number;
	pinned: boolean;
	/** Resolved capability record (M15). */
	caps: ModelCaps;
}

// ---------------------------------------------------------------------------
// Engine + GPU types (M4)
// Mirror the Rust types in src-tauri/src/engine/ and src-tauri/src/gpu/
// ---------------------------------------------------------------------------

export type EngineVariant = 'cuda-12.4' | 'cpu';

export type EnginePhase = 'idle' | 'installing' | 'loading' | 'ready' | 'failed';

export type EngineStatus =
	| { phase: 'idle' }
	| { phase: 'installing'; variant: string; pct: number; bytes: number; total: number; stage: string;
	  }
	| { phase: 'loading'; model_id: string; launcher: EngineKind }
	| { phase: 'ready'; model_id: string; port: number; ctx: number; launcher: EngineKind }
	/** `launcher` is absent when no load was involved (an install failure). */
	| { phase: 'failed'; error: string; launcher?: EngineKind };

export interface EngineRowDto {
	id: string;
	version: string;
	variant: string;
	path: string;
	active: boolean;
	installed_at: number;
}

export interface GpuMetrics {
	name: string;
	driver_version: string;
	cuda_major: number;
	cuda_minor: number;
	vram_total_mb: number;
	vram_used_mb: number;
	vram_free_mb: number;
	sampled_at: number;
}

// GpuStatus uses #[serde(tag = "kind")] on a tuple-like Available(GpuMetrics)
// variant. Serde flattens the inner struct fields into the tagged object, so
// the JSON is { kind: "available", name, driver_version, ... } — no nesting.
export type GpuStatus = { kind: 'no_gpu' }
	| ({ kind: 'available' } & GpuMetrics);

/** Physical RAM snapshot in MiB — mirrors commands::system::SystemMemory. */
export interface SystemMemory {
	total_mb: number;
	used_mb: number;
	available_mb: number;
}

export type FitVerdict = 'fits' | 'tight' | 'wont_fit';

export interface FitEstimate {
	/** Installed text admission/placement; not the store quant VRAM-only estimate. */
	plan?: LoadPlan | null;
	/** Planned graphics footprint, or host footprint on CPU-only machines, MiB.
	 * Legacy non-text estimates retain their weights/overhead formula. */
	estimated_mb: number;
	/** Context cache on that tier. Native GPU KV is zero; see plan.context for host context. */
	kv_at_ctx_mb: number;
	free_mb: number;
	/** VRAM held by the loaded text engine — released by a same-slot swap,
	 *  so it counts as available when judging a (re)load. */
	reclaimable_mb: number;
	verdict: FitVerdict;
	gpu: boolean;
}

// ---------------------------------------------------------------------------
// Load planner — mirrors src-tauri/src/planner/ (snake_case, bytes as numbers)
// ---------------------------------------------------------------------------

export type DriveKind = 'ssd' | 'hdd' | 'unknown';

export interface PlannerGpuInfo {
	name: string;
	vram_total_bytes: number;
	vram_free_bytes: number;
}

export interface StorageInfo {
	/** Mount point holding the models folder, e.g. `C:\`. */
	mount: string;
	/** Device model when Windows reports it, e.g. "Samsung SSD 990 PRO 2TB". */
	model: string | null;
	kind: DriveKind;
	free_bytes: number;
	total_bytes: number;
}

/** Bandwidths the speed estimate uses, in GB/s (10^9 bytes). */
export interface Bandwidths {
	vram_gbps: number;
	pcie_gbps: number;
	ram_gbps: number;
	ssd_gbps: number;
	/** True when `ssd_gbps` came from the SSD test on this drive. */
	ssd_measured: boolean;
}

/** Result of hardware_profile — mirrors planner::hardware::HardwareProfile. */
export interface HardwareProfile {
	/** null when no NVIDIA GPU is present (CPU-only planning). */
	gpu: PlannerGpuInfo | null;
	ram_total_bytes: number;
	ram_available_bytes: number;
	/** Commit Windows can still hand out (RAM plus pagefile); null elsewhere. */
	commit_available_bytes?: number | null;
	storage: StorageInfo | null;
	bandwidth: Bandwidths;
	cpu_threads: number;
}

/** Result of benchmark_ssd — mirrors planner::hardware::SsdMeasurement. */
export interface SsdMeasurement {
	mount: string;
	gbps: number;
	bytes_read: number;
	seconds: number;
	/** Unix seconds. */
	measured_at: number;
	file: string;
}

export type KvType = 'f16' | 'q8_0' | 'q4_0';
export type SpeedLabel = 'slow' | 'usable' | 'fast';
/** Which tiers a plan touches, fastest first. */
export type TierMix = 'vram' | 'ram' | 'ssd';
export type EngineKind = 'llama_server' | 'ssd_stream';
export type Bottleneck = 'vram' | 'pcie' | 'ram' | 'ssd' | 'cpu' | 'overhead';

/** Decode-speed thresholds behind SpeedLabel (mirrors planner::speed). */
export const FAST_TPS = 30;
export const USABLE_TPS = 8;

/** User choices that pin parts of a plan; unset fields are planner-chosen. */
export interface PlanOverrides {
	/** Auto keeps llama-server when it fits; ssd_stream forces native. */
	engine?: 'auto' | EngineKind;
	ctx?: number;
	/** K cache type (and the V type unless kv_type_v is set). */
	kv_type?: KvType;
	/** V cache type. MLA models keep no V cache and always use the K type. */
	kv_type_v?: KvType;
	/** MoE layers whose routed experts stay entirely in graphics memory. */
	gpu_expert_layers?: number;
	/** GPU cache for hot experts of host-resident layers, MiB (0 = off). */
	expert_cache_mib?: number;
	/** Allow weights beyond free RAM to be paged from the SSD (default true). */
	allow_ssd?: boolean;
	/** Most system memory the load may use, MiB. Caps the planner's own budget, never raises it. */
	ram_limit_mib?: number;
	/** Most graphics memory the load may use, MiB, overhead included. 0 keeps the model off the card. */
	vram_limit_mib?: number;
}

/** Inputs that size a context (mirrors planner::context::ContextParams). */
export interface ContextParams {
	/** Tokens; padded to 256 like llama.cpp. */
	ctx: number;
	kv_k: KvType;
	/** Ignored for MLA models (V follows K). */
	kv_v: KvType;
	/** Parallel sequences (`-np`). */
	n_parallel: number;
	/** Micro-batch (`-ub`). */
	ubatch: number;
	/** Some routed experts live in host memory. */
	experts_on_host: boolean;
}

/**
 * What a context costs (mirrors planner::context::ContextMemory). Bytes for
 * the whole process. The planner uses the same numbers, so a context slider
 * and a load agree.
 */
export interface ContextMemory {
	/** Context llama.cpp allocates (padded to 256). */
	ctx: number;
	/** V type actually used (MLA forces the K type). */
	kv_v: KvType;
	/** Full-attention K+V cache (MLA: the compressed latent). */
	kv_bytes: number;
	/** Sliding-window K+V cache (stops growing at the window). */
	swa_bytes: number;
	/** Sparse-MLA lightning-indexer cache. */
	indexer_bytes: number;
	/** Fixed recurrent state of linear-attention / SSM layers. */
	state_bytes: number;
	/** Worst-case compute buffer on the GPU (system memory without one). */
	compute_bytes: number;
	/** Host-side compute buffer next to a GPU. */
	host_compute_bytes: number;
	/** Bytes added per extra token (the slider's slope). */
	bytes_per_token: number;
}

/** Bytes per memory tier. The four vram_* parts add up to the GPU footprint. */
export interface TierBytes {
	vram_weights: number;
	vram_expert_cache: number;
	vram_kv: number;
	vram_overhead: number;
	ram: number;
	ssd: number;
}

export interface SpeedEstimate {
	decode_tps: number;
	prompt_tps: number;
	label: SpeedLabel;
	raw_decode_tps: number;
	raw_prompt_tps: number;
	/** Measured samples blended in (0 = pure model). */
	calibration_samples: number;
	bottleneck: Bottleneck;
}

export interface Reclaimable {
	id: string;
	label: string;
	bytes: number;
}

export interface EvictAlternative {
	evicts: Reclaimable[];
	vram_bytes: number;
	tier_mix: TierMix;
	decode_tps: number;
	label: SpeedLabel;
}

export interface NativePlan {
	ram_budget_mib: number;
	host_headroom_mib: number;
	expert_cache_mib: number;
	vram_budget_bytes: number;
	vram_reserve_mib: number;
	prefetch_depth: number;
	/** Windows process memory envelope, MiB: the host budget plus the graphics-memory tier. */
	host_budget_mib?: number | null;
	estimate_basis: string;
}

/** Cached --engine-info; free-device memory is the probe snapshot, not a live budget. */
export interface NativeEngineStatus {
	installed: boolean;
	hosted: boolean;
	path: string;
	error: string | null;
	info: {
		engine: string;
		engine_version: string;
		llama_cpp: { tag: string; commit: string };
		features: { ssd_expert_streaming: boolean; cuda_vram_expert_tier: boolean };
		backends: {
			name: string;
			devices: {
				name: string;
				description: string;
				type: string;
				memory_total_mib: number;
				memory_free_mib: number;
			}[];
		}[];
	} | null;
}

/** Result of plan_load — mirrors planner::LoadPlan. */
export interface LoadPlan {
	engine_kind: EngineKind;
	/** Present only for native SSD plans. Context is entirely host-resident. */
	native: NativePlan | null;
	architecture: string;
	ctx: number;
	/** Trained context (0 when unknown). */
	ctx_train: number;
	/** K cache type (`-ctk`). */
	kv_type: KvType;
	/** V cache type (`-ctv`). */
	kv_type_v: KvType;
	/** Parallel sequences (`-np`, always 1 from the planner). */
	n_parallel: number;
	/** Context memory and compute buffers at these settings. */
	context: ContextMemory;
	/** When the context doesn't fit: the largest one that does (null if none). */
	max_ctx: number | null;
	/** Why the plan can't load; set exactly when feasible is false. */
	infeasible_reason: string | null;
	/** `-ngl`; -1 = all layers. */
	gpu_layers: number;
	gpu_expert_layers: number;
	cpu_expert_layers: number;
	n_cpu_moe: number | null;
	expert_cache_mib: number;
	ubatch: number;
	tiers: TierBytes;
	tier_mix: TierMix;
	/** False when the model can't be placed with these settings. */
	feasible: boolean;
	speed: SpeedEstimate;
	/** llama-server flags the plan uses (display). */
	args: string[];
	/** Plain-language warnings (eviction, SSD streaming, context capped…). */
	warnings: string[];
	evict_alternative: EvictAlternative | null;
	overrides: PlanOverrides;
}

export interface EstimateSummary {
	decode_tps: number;
	prompt_tps: number;
	label: SpeedLabel;
	tier_mix: TierMix;
	vram_bytes: number;
	ram_bytes: number;
	ssd_bytes: number;
}

/** Live actuals for the loaded text model (subscribe_runtime_stats, ~1/s). */
export interface RuntimeStats {
	model_id: string | null;
	decode_tps: number | null;
	prompt_tps: number | null;
	decode_label: SpeedLabel | null;
	busy: boolean;
	tokens_generated: number;
	vram_mb: number | null;
	ram_bytes: number | null;
	estimate: EstimateSummary | null;
	/** Unix seconds. */
	sampled_at: number;
}

// ---------------------------------------------------------------------------
// Chat types (M5)
// Mirror the Rust DTOs in src-tauri/src/chat/ and src-tauri/src/commands/chat.rs
// ---------------------------------------------------------------------------

export type ChatRole = 'user' | 'assistant' | 'system';

export interface MessageDto {
	seq: number;
	role: ChatRole;
	content: string;
	ts: number;
	prompt_tokens?: number;
	completion_tokens?: number;
	finish_reason?: string;
}

export interface ConversationSummaryDto {
	id: string;
	title: string | null;
	model_id: string | null;
	created_at: number;
	updated_at: number;
	message_count: number;
}

export interface SearchHitDto {
	conversation_id: string;
	seq: number;
	snippet: string;
}

// Per-conversation chat configuration (M6 §6.4).
// Mirrors src-tauri/src/chat/settings.rs.
export interface SamplingParams {
	temperature: number;
	top_p: number;
	top_k: number;
	min_p: number;
	max_tokens?: number;
	repeat_penalty: number;
	presence_penalty: number;
	frequency_penalty: number;
	seed?: number;
	stop: string[];
}

export interface LoadConfig {
	ctx: number;
	ngl: number;
	batch_size?: number;
	threads?: number;
	flash_attn: boolean;
	kv_cache_type?: string;
	mmap: boolean;
	mlock: boolean;
}

export interface ConversationSettings {
	system_prompt?: string | null;
	sampling: SamplingParams;
	load_config?: LoadConfig | null;
}

/** Default sampling values — mirrors `SamplingParams::default()` in Rust. */
export const DEFAULT_SAMPLING: SamplingParams = {
	temperature: 0.7,
	top_p: 0.9,
	top_k: 40,
	min_p: 0.05,
	repeat_penalty: 1.1,
	presence_penalty: 0.0,
	frequency_penalty: 0.0,
	stop: [],
};

export type ChatEvent =
	| { type: 'user_message_stored'; conv_id: string; seq: number }
	| { type: 'assistant_start'; conv_id: string; seq: number }
	| { type: 'token'; conv_id: string; seq: number; delta: string }
	| { type: 'thinking'; conv_id: string; seq: number; delta: string }
	| { type: 'assistant_done'; conv_id: string; seq: number; prompt_tokens: number; completion_tokens: number; finish_reason: string;
	  }
	| { type: 'truncated'; conv_id: string; from_seq: number }
	| { type: 'error'; conv_id: string; error: string };

// ---------------------------------------------------------------------------
// Runtime v2 types (M8)
// Mirror the Rust types in src-tauri/src/worker/, src-tauri/src/orchestrator/
// and src-tauri/src/generation.rs
// ---------------------------------------------------------------------------

/** Worker venv flavor — torch+CUDA wheels vs CPU-only. */
export type WorkerVariantSlug = 'cuda' | 'cpu';

/** One pipeline the Python worker can run — mirrors worker::PipelineCapability. */
export interface PipelineCapability {
	task: string;
	summary: string;
	available: boolean;
	/** Python packages missing for this pipeline (empty when available). */
	missing: string[];
	/** Internal pipelines (echo) — never shown in the UI. */
	hidden: boolean;
	/** Model formats the `model` param accepts (drives M9 store badges). */
	formats: string[];
}

/** Python-worker lifecycle — mirrors worker::WorkerStatus (tag = "phase"). */
export type WorkerStatus =
	| { phase: 'not_installed' }
	| { phase: 'installing'; variant: string; stage: string; pct: number; bytes: number; total: number;
	  }
	| { phase: 'stopped'; variant: string }
	| { phase: 'starting' }
	| {
			phase: 'ready';
			variant: string;
			version: string;
			pipelines: PipelineCapability[];
	  }
	| { phase: 'failed'; error: string };

export type ResidentKind = 'text_engine' | 'tts_engine' | 'pipeline';

/** One model resident on the GPU — mirrors orchestrator::ResidentInfo. */
export interface ResidentInfo {
	/** `"text"`, `"tts"`, or `"pipeline:<task>"` — pass to orchestratorUnload. */
	id: string;
	kind: ResidentKind;
	label: string;
	/** Originating catalog model id, where known (M16 W1 loaded-state). */
	model_id: string | null;
	est_mb: number;
	/** NVML-actual VRAM after load, where known. */
	actual_mb: number | null;
}

/** VRAM ledger snapshot — mirrors orchestrator::LedgerSnapshot. */
export interface LedgerSnapshot {
	/** False when no NVIDIA GPU is present (ledger is display-only then). */
	gpu: boolean;
	total_mb: number;
	free_mb: number;
	residents: ResidentInfo[];
}

/** Gated text-model load request — mirrors orchestrator::TextLoadRequest.
 * Field names are snake_case because the struct deserializes them verbatim. */
export interface TextLoadRequest {
	model_id: string;
	ctx?: number;
	ngl?: number;
	batch_size?: number;
	threads?: number;
	flash_attn?: boolean;
	kv_cache_type?: string;
	mmap?: boolean;
	mlock?: boolean;
	/** Skip the eviction gate and load with whatever fits. */
	force?: boolean;
	/** A plan from plan_load to load with (used when it still fits).
	 *  Without `ngl` and without a plan, the planner picks the settings. */
	plan?: LoadPlan;
	/** Planner overrides (ctx / kv_cache_type above are folded in too). */
	overrides?: PlanOverrides;
}

/** Result of orchestratorLoadModel — mirrors orchestrator::LoadOutcome. */
export type LoadOutcome =
	| {
			outcome: 'loaded';
			port: number;
			model_id: string;
			ctx: number;
			ngl: number;
			plan?: LoadPlan;
	  }
	| {
			outcome: 'needs_eviction';
			required_mb: number;
			free_mb: number;
			candidates: ResidentInfo[];
	  };

/** Result of the unified loadModel command — mirrors commands::orchestrator::ModelLoadOutcome (M16 W1). */
export type ModelLoadOutcome =
	| {
			outcome: 'loaded';
			model_id: string;
			resident_id: string;
			plan?: LoadPlan;
	  }
	| {
			outcome: 'needs_eviction';
			required_mb: number;
			free_mb: number;
			candidates: ResidentInfo[];
	  };

/** Result of a generation submit — mirrors generation::SubmitOutcome. */
export type SubmitOutcome =
	| { outcome: 'submitted'; job_id: string }
	| {
			outcome: 'needs_eviction';
			required_mb: number;
			free_mb: number;
			candidates: ResidentInfo[];
	  };

/** Pipeline task slugs accepted by the generation commands. */
export type GenerationTask = 'text_to_image' | 'text_to_video' | 'text_to_speech' | 'transcribe';

/** Qwen3-TTS sidecar lifecycle — mirrors tts::TtsStatus (tag = "phase"). */
export type TtsStatus =
	| { phase: 'not_installed' }
	| {
			phase: 'installing';
			stage: string;
			pct: number;
			bytes: number;
			total: number;
	  }
	| { phase: 'idle' }
	| { phase: 'loading'; model_id: string }
	| { phase: 'ready'; model_id: string; port: number; voices: string[] }
	| { phase: 'failed'; error: string };

export type GenerationJobStatus = 'running' | 'done' | 'error' | 'cancelled';

/** A persisted generation job — mirrors db::generation_jobs::GenerationJob. */
export interface GenerationJob {
	id: string;
	task: string;
	status: GenerationJobStatus;
	params: unknown;
	outputs: string[];
	meta: unknown;
	error: string | null;
	created_at: number;
	finished_at: number | null;
}

/** Live job events from subscribeGenerationProgress — mirrors
 * worker::protocol::WorkerEvent (tag = "event"; Log is filtered out). */
export type GenerationEvent =
	| { event: 'job_progress'; job_id: string; pct: number; stage: string; step?: number | null; total_steps?: number | null;
	  }
	| {
			event: 'job_done';
			job_id: string;
			outputs: string[];
			meta: unknown;
			task?: string | null;
	  }
	| { event: 'job_error'; job_id: string; error: string };

// ---------------------------------------------------------------------------
// Asset library (M10)
// Mirror the Rust DTOs in src-tauri/src/db/assets.rs
// ---------------------------------------------------------------------------

export type AssetKind = 'image' | 'video' | 'audio' | 'text';

/** A generated artifact registered in the shared asset library. */
export interface Asset {
	id: string;
	/** Source generation job (soft reference — survives job deletion). */
	job_id: string | null;
	kind: AssetKind;
	/** Absolute path inside the assets dir. */
	path: string;
	/** Source pipeline task ('text_to_image', …). */
	task: string;
	/** Prompt / spoken text, when the job had one. */
	prompt: string | null;
	/** Human-readable model label. */
	model: string | null;
	params: unknown;
	meta: unknown;
	size: number;
	created_at: number;
}