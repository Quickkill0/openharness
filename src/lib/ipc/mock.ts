/**
 * In-browser mock backend for UI development.
 *
 * Active only when the app runs WITHOUT the Tauri bridge (plain `pnpm dev`
 * opened in a browser). Provides realistic canned data for every command the
 * app uses, including streaming chat and Studio jobs.
 *
 * Never bundled into the Tauri build — `./backend` imports it lazily and only
 * on the non-Tauri path.
 */
import { splitGgufPart, splitGgufFiles } from '$lib/splitGguf';
import type {
	Asset,
	AssetKind,
	Bottleneck,
	ChatEvent,
	ConversationSummaryDto,
	CuratedPick,
	DownloadGroup,
	DownloadProgress,
	GenerationEvent,
	GenerationJob,
	HardwareProfile,
	HfFile,
	HfModelCard,
	HfModelDetails,
	Job,
	ContextMemory,
	ContextParams,
	KvType,
	LedgerSnapshot,
	LoadPlan,
	MessageDto,
	ModelCaps,
	PipelineCapability,
	PlanOverrides,
	QuantFit,
	Reclaimable,
	RepoLayout,
	ResidentInfo,
	Runnability,
	SystemMemory,
	RunnabilityQuery,
	RuntimeStats,
	SpeedLabel,
	SsdMeasurement,
	TextLoadRequest,
	TierBytes,
	TierMix,
	TtsStatus,
	VoiceInfo,
	Workflow,
	WorkerStatus,
} from '$lib/types';
import { FAST_TPS, USABLE_TPS } from '../types';
import { KOKORO_VOICES } from '../speech';
import { mockServerInvoke, mockServerSubscribe } from './server.mock';

const now = () => Math.floor(Date.now() / 1000);

// ---------------------------------------------------------------------------
// Event bus (mock channels)
// ---------------------------------------------------------------------------

type Listener = (msg: unknown) => void;
const listeners = new Map<string, Set<Listener>>();

export function mockSubscribe<T>(cmd: string, onmessage: (msg: T) => void,
): { onmessage: (msg: T) => void } {
	if (cmd === 'server_subscribe_request_log') return mockServerSubscribe(onmessage as (msg: import('./server').ServerRequestLog) => void) as { onmessage: (msg: T) => void;
		};
	let set = listeners.get(cmd);
	if (!set) {
		set = new Set();
		listeners.set(cmd, set);
	}
	const handle = { onmessage };
	set.add(onmessage as Listener);
	// Subscriptions that snapshot immediately in the real backend:
	if (cmd === 'engine_subscribe_status') queueMicrotask(() => onmessage(engineState as T));
	if (cmd === 'gpu_subscribe_metrics') queueMicrotask(() => onmessage(mockGpuFeed() as T));
	if (cmd === 'subscribe_worker_status') {
		queueMicrotask(() => onmessage(workerState as T));
		// First contact kicks off the auto install→start walk (the backend
		// equivalent runs at app startup, before any subscriber exists).
		startWorkerAutoFlow();
	}
	if (cmd === 'subscribe_orchestrator_ledger') queueMicrotask(() => onmessage(ledgerSnapshot() as T));
	if (cmd === 'subscribe_tts_status') queueMicrotask(() => onmessage(ttsState as T));
	if (cmd === 'subscribe_runtime_stats') {
		queueMicrotask(() => onmessage(runtimeStatsNow() as T));
		startRuntimeStats();
	}
	return handle;
}

function emit(cmd: string, msg: unknown) {
	for (const l of listeners.get(cmd) ?? []) l(msg);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Fixtures — engine / GPU / models / settings
// ---------------------------------------------------------------------------

const MODEL_ID = 'qwen2.5-coder-7b-instruct-q4_k_m';

const ENGINE_READY = {
	phase: 'ready',
	model_id: MODEL_ID,
	port: 8732,
	ctx: 8192,
	launcher: 'llama_server',
};

/** Live llama-server state — mutated by load_model/unload (M16 W1). */
// oh.mock.status starts the app in a state the UI didn't cause, such as a
// native load started through the OpenAI API.
let engineState: Record<string, unknown> = JSON.parse(
	globalThis.localStorage?.getItem('oh.mock.status') ?? 'null',
) ?? { ...ENGINE_READY };

/** Installed engine variant — mutated by engine_install/engine_switch_variant (M16 W6). */
let engineVariant = 'cuda-12.4';

function setEngineState(s: Record<string, unknown>) {
	engineState = s;
	emit('engine_subscribe_status', s);
}

type MockNative = 'ok' | 'missing' | 'no_capability';

/**
 * Which text engines the mock machine has: set localStorage `oh.mock.engines`
 * to `both` (default), `native-only`, `llama-only`, `neither` or
 * `native-no-capability` and reload.
 */
function mockEngineSetup(): { llama: boolean; native: MockNative } {
	let v: string | null = null;
	try {
		v = globalThis.localStorage?.getItem('oh.mock.engines') ?? null;
	} catch {
		// No storage: the default machine has both.
	}
	switch (v) {
		case 'native-only':
			return { llama: mockLlamaInstalled, native: 'ok' };
		case 'llama-only':
			return { llama: true, native: 'missing' };
		case 'neither':
			return { llama: mockLlamaInstalled, native: 'missing' };
		case 'native-no-capability':
			return { llama: mockLlamaInstalled, native: 'no_capability' };
		default:
			return { llama: true, native: 'ok' };
	}
}

/** The install wizard ran in this page load, whatever the setup says. */
let mockLlamaInstalled = false;

/** Mirrors `native_engine_info` for the mock machine's setup. */
function mockNativeInfo() {
	const path = 'C:\\mock\\engine\\native\\openharness-engine-server.exe';
	const state = mockEngineSetup().native;
	if (state === 'missing') return { installed: false, hosted: false, path, error: null, info: null };
	const streaming = state === 'ok';
	return {
		installed: true,
		hosted: false,
		path,
		error: streaming ? null : 'The native engine does not report ssd_expert_streaming.',
		info: {
			engine: 'openharness-engine',
			engine_version: '0.1.0-mock',
			llama_cpp: { tag: 'b12192', commit: 'mock' },
			features: { ssd_expert_streaming: streaming, cuda_vram_expert_tier: streaming },
			backends: [
				{
					name: 'CUDA',
					devices: [
						{
							name: 'CUDA0',
							description: 'Mock RTX 4090',
							type: 'gpu',
							memory_total_mib: 24576,
							memory_free_mib: 22528,
						},
					],
				},
			],
		},
	};
}

function mockEngineRow() {
	return {
		id: 'mock-engine', version: 'b12182', variant: engineVariant,
		path: 'C:\\mock\\engine\\b12182', active: true, installed_at: 1748000000,
	};
}

/** Simulate an engine (re)install: installing ticks on the status channel, then idle. */
async function simulateEngineInstall(variant: string) {
	const total = variant === 'cpu' ? 600_000_000 : 700_000_000;
	const stages: [string, number][] = [
		['resolving release', 5],
		['downloading engine', 35],
		['downloading engine', 75],
		['extracting', 92],
		['verifying', 100],
	];
	for (const [stage, pct] of stages) {
		setEngineState({
			phase: 'installing', variant, pct,
			bytes: Math.round((total * pct) / 100), total, stage,
		});
		await sleep(400);
	}
	engineVariant = variant;
	// A (re)install replaces the engine binaries — any running llama-server dies.
	setEngineState({ phase: 'idle' });
	residents = residents.filter((r) => r.id !== 'text');
	emitLedger();
}

const ENGINE_LOG_LINES = [
	'build: b12182 (mock) with MSVC 19.40 for x64',
	'system info: n_threads = 16, total_threads = 32 | CUDA : ARCHS = 890 | F16 = 1',
	`llama_model_loader: loaded meta data with 38 key-value pairs from ${MODEL_ID}.gguf`,
	'load_tensors: offloaded 33/33 layers to GPU',
	'llama_context: n_ctx = 8192, n_batch = 2048, flash_attn = 0',
	'main: server is listening on http://127.0.0.1:8732 - starting the main loop',
	'srv  update_slots: all slots are idle',
];

const GPU_OK = {
	kind: 'available',
	name: 'NVIDIA GeForce RTX 4090 (mock)',
	driver_version: '560.94',
	cuda_major: 12,
	cuda_minor: 6,
	vram_total_mb: 24576,
	vram_used_mb: 9801,
	vram_free_mb: 14775,
	sampled_at: Math.floor(Date.now() / 1000),
};

// --- Capability records (M15) — mirrors src-tauri/src/models/capabilities.rs ---

const QWEN_TTS_VOICE_INFOS: VoiceInfo[] = [
	{ id: 'serena', label: 'Serena' },
	{ id: 'vivian', label: 'Vivian' },
	{ id: 'uncle_fu', label: 'Uncle Fu' },
	{ id: 'ryan', label: 'Ryan' },
	{ id: 'aiden', label: 'Aiden' },
	{ id: 'ono_anna', label: 'Ono Anna' },
	{ id: 'sohee', label: 'Sohee' },
	{ id: 'eric', label: 'Eric' },
	{ id: 'dylan', label: 'Dylan' },
];

// VibeVoice curated built-in presets — mirrors BUILTIN_VOICES in
// src-tauri/src/models/vibevoice_voices.rs.
const VIBEVOICE_VOICE_INFOS: VoiceInfo[] = [
	{ id: 'alice', label: 'Alice — English female' },
	{ id: 'maya', label: 'Maya — English female' },
	{ id: 'carter', label: 'Carter — English male' },
	{ id: 'frank', label: 'Frank — English male' },
	{ id: 'samuel', label: 'Samuel — English male' },
];

interface CapsInput {
	format: string;
	capability: string;
	task: string | null;
	architecture?: string;
	repo_id: string;
	display_name: string;
}

// Actionable verdict for the original (non-GGUF) Qwen3-TTS safetensors release.
// Mirrors QWEN_TTS_SAFETENSORS_REASON in capabilities.rs / store.rs.
const QWEN_TTS_SAFETENSORS_REASON =
	'This is the original safetensors build — OpenHarness runs Qwen3-TTS from its GGUF build instead. Download the GGUF version of this voice from the Store.';

/** Mirrors models::capability::capability_for_task for repo registration. */
function capabilityForTask(task: string | null): string {
	switch (task) {
		case 'text-generation': return 'text-generation';
		case 'feature-extraction':
		case 'sentence-similarity': return 'embedding';
		case 'automatic-speech-recognition':
		case 'text-to-speech':
		case 'audio-to-audio': return 'speech';
		case 'text-to-image':
		case 'image-to-image':
		case 'text-to-video':
		case 'image-to-video': return 'image-gen';
		case 'image-text-to-text':
		case 'visual-question-answering': return 'vision-text';
		default: return 'other';
	}
}

function resolveMockCaps(m: CapsInput): ModelCaps {
	const none: ModelCaps = {
		runtime: null,
		pipeline: null,
		workflows: [],
		speech: null,
		image_arch: null,
		gguf_diffusion: false,
		companion_arch: null,
		reason: null,
	};
	if (m.architecture === 'qwen3-tts') {
		// Mirror qwen_talker_mode() in capabilities.rs: variant token from the
		// repo id + display name, separators collapsed; "base" word-bounded.
		const qhay = `${m.repo_id} ${m.display_name}`.toLowerCase();
		const collapsed = qhay.replace(/[^a-z0-9]/g, '');
		const mode = collapsed.includes('voicedesign')
			? 'voice_design'
			: collapsed.includes('customvoice')
				? 'custom_voice'
				: qhay.split(/[^a-z0-9]+/).includes('base')
					? 'base'
					: 'custom_voice';
		return {
			...none,
			runtime: 'qwen-tts',
			workflows: ['speech'],
			speech: {
				engine: 'qwen3-tts',
				voices: mode === 'custom_voice' ? QWEN_TTS_VOICE_INFOS : [],
				supports_speed: false,
				supports_instructions: mode !== 'base',
				supports_voice_cloning: mode === 'base',
				cloning_requires_ref_text: mode === 'base',
			},
		};
	}
	if (m.format === 'gguf') {
		if (m.capability === 'text-generation' || m.capability === 'vision-text') {
			return { ...none, runtime: 'llama-server', workflows: ['chat'] };
		}
		if (m.capability === 'embedding') {
			return { ...none, runtime: 'llama-server', workflows: ['embeddings'] };
		}
		if (m.capability === 'image-gen') {
			return { ...none, reason: "GGUF diffusion checkpoints aren't supported — install the safetensors or diffusers release instead" };
		}
		if (m.capability === 'speech') {
			return { ...none, reason: "GGUF speech models aren't supported — transcription runs faster-whisper (CTranslate2) repos" };
		}
		// Capability 'other' (card gave no signal) — fall back on the parsed
		// GGUF architecture; unknown decoder archs default to text.
		const arch = (m.architecture ?? '').toLowerCase();
		if (!arch) return { ...none, reason: 'No local runtime recognizes this model' };
		if (arch.includes('whisper')) {
			return {
				...none,
				reason:
					"GGUF speech models aren't supported — transcription runs faster-whisper (CTranslate2) repos",
			};
		}
		if (arch.includes('stable-diffusion') || arch.includes('sdxl') || arch.includes('flux')) {
			return {
				...none,
				reason:
					"GGUF diffusion checkpoints aren't supported — install the safetensors or diffusers release instead",
			};
		}
		if (arch === 'qwen3tts') {
			return {
				...none,
				reason:
					"This talker uses the unsupported 'qwen3tts' GGUF layout — install a release with the 'qwen3-tts' architecture instead",
			};
		}
		if (arch.includes('bert')) {
			return { ...none, runtime: 'llama-server', workflows: ['embeddings'] };
		}
		// ComfyUI-style single-file diffusion GGUFs (DiT only). Route to the
		// Studio worker pipelines before the Chat/Agent catch-all — they aren't
		// decoder LLMs. Mirrors resolve_diffusion_gguf() in capabilities.rs.
		const diffusion: Record<string, { pipeline: 'text_to_image' | 'text_to_video'; workflow: Workflow }> =
			{
				qwen_image: { pipeline: 'text_to_image', workflow: 'image' },
				ltxv: { pipeline: 'text_to_video', workflow: 'video' },
				wan: { pipeline: 'text_to_video', workflow: 'video' },
			};
		if (diffusion[arch]) {
			const { pipeline, workflow } = diffusion[arch];
			return {
				...none,
				runtime: 'worker',
				pipeline,
				workflows: [workflow],
				gguf_diffusion: true,
				companion_arch: arch,
			};
		}
		// Known-but-unsupported DiT archs: honest not-runnable, never Chat/Agent.
		const unsupportedDiffusion = ['hunyuan-video', 'mochi', 'cogvideox', 'sd3', 'auraflow', 'lumina2',
		];
		if (unsupportedDiffusion.includes(arch)) {
			return {
				...none,
				reason:
					"This is a single-file GGUF diffusion model OpenHarness can't run yet — its architecture isn't supported. Use a supported diffusion GGUF (Qwen-Image, LTX, Wan) or install the full diffusers release.",
			};
		}
		return { ...none, runtime: 'llama-server', workflows: ['chat'] };
	}
	const hay = `${m.architecture ?? ''} ${m.repo_id} ${m.display_name}`.toLowerCase();
	// Non-GGUF Qwen3-TTS = the original safetensors release, which has no local
	// runtime (real talkers carry the qwen3-tts arch and are handled above).
	// Detect the family by name and refuse with a pointer to the GGUF build —
	// otherwise it would fall into the Kokoro text-to-speech arm. Mirrors
	// is_qwen_tts() in capabilities.rs.
	if (`${m.repo_id} ${m.display_name}`.toLowerCase().replace(/[^a-z0-9]/g, '').match(/qwen3tts|qwentts/)) {
		return { ...none, reason: QWEN_TTS_SAFETENSORS_REASON };
	}
	// VibeVoice checkpoints identify by name — cards carry the generic
	// text-to-speech tag (or none on partial mirrors).
	if (hay.includes('vibevoice') && (m.task === 'text-to-speech' || m.task === null)) {
		return {
			...none,
			runtime: 'worker',
			pipeline: 'vibevoice',
			workflows: ['speech'],
			speech: {
				engine: 'vibevoice',
				voices: VIBEVOICE_VOICE_INFOS, // curated preset clips; cloning still offered
				supports_speed: false,
				supports_instructions: false,
				supports_voice_cloning: true,
				cloning_requires_ref_text: false,
			},
		};
	}
	switch (m.task) {
		case 'text-to-image':
		case 'image-to-image':
			return {
				...none,
				runtime: 'worker',
				pipeline: 'text_to_image',
				workflows: ['image'],
				image_arch: hay.includes('flux')
					? 'flux'
					: hay.includes('xl') || hay.includes('turbo')
						? 'sdxl'
						: 'sd',
			};
		case 'text-to-video':
		case 'image-to-video':
			return {
				...none,
				runtime: 'worker',
				pipeline: 'text_to_video',
				workflows: ['video'],
			};
		case 'automatic-speech-recognition':
			return {
				...none,
				runtime: 'worker',
				pipeline: 'transcribe',
				workflows: ['transcribe'],
			};
		case 'text-to-speech':
			return {
				...none,
				runtime: 'worker',
				pipeline: 'text_to_speech',
				workflows: ['speech'],
				speech: {
					engine: 'kokoro',
					voices: KOKORO_VOICES,
					supports_speed: true,
					supports_instructions: false,
					supports_voice_cloning: false,
					cloning_requires_ref_text: false,
				},
			};
		case 'text-generation':
			return { ...none, reason: 'Text models run from GGUF here — install the GGUF release instead',
			};
		default:
			return { ...none, reason: `No local runtime for task '${m.task}' yet` };
	}
}

const MODELS_RAW = [
	{
		id: MODEL_ID,
		repo_id: 'Qwen/Qwen2.5-Coder-7B-Instruct-GGUF',
		revision: 'main',
		display_name: 'Qwen2.5 Coder 7B Instruct Q4_K_M',
		architecture: 'qwen2',
		capability: 'text-generation',
		format: 'gguf' as const,
		task: 'text-generation',
		local_path: null,
		params_b: 7.6,
		quantization: 'Q4_K_M',
		context_train: 32768,
		installed_at: 1748000000,
		primary_file_path: 'C:\\mock\\models\\qwen2.5-coder-7b.gguf',
		primary_file_size: 4_900_000_000,
		primary_rel_path: 'qwen2.5-coder-7b-instruct-q4_k_m.gguf',
		primary_sha256: MODEL_ID,
		file_count: 1,
		total_size: 4_900_000_000,
		pinned: false,
	},
	{
		id: 'llama-3.1-8b-instruct-q5_k_m',
		repo_id: 'meta-llama/Llama-3.1-8B-Instruct-GGUF',
		revision: 'main',
		display_name: 'Llama 3.1 8B Instruct Q5_K_M',
		architecture: 'llama',
		capability: 'text-generation',
		format: 'gguf' as const,
		task: 'text-generation',
		local_path: null,
		params_b: 8.0,
		quantization: 'Q5_K_M',
		context_train: 131072,
		installed_at: 1747000000,
		primary_file_path: 'C:\\mock\\models\\llama-3.1-8b.gguf',
		primary_file_size: 5_700_000_000,
		primary_rel_path: 'llama-3.1-8b.gguf',
		primary_sha256: null,
		file_count: 1,
		total_size: 5_700_000_000,
		pinned: true,
	},
	{
		id: 'glm-5.3-flash-ud-q2_k_xl',
		repo_id: 'unsloth/GLM-5.3-Flash-GGUF',
		revision: 'main',
		display_name: 'GLM 5.3 Flash UD-Q2_K_XL',
		architecture: 'glm5-next',
		capability: 'text-generation',
		format: 'gguf' as const,
		task: 'text-generation',
		local_path: null,
		params_b: 368,
		quantization: 'UD-Q2_K_XL',
		context_train: 1048576,
		installed_at: 1749000000,
		primary_file_path: 'C:\\mock\\models\\GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf',
		primary_file_size: 9_429_984,
		primary_rel_path: 'UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf',
		file_count: 4,
		total_size: 108_720_071_552,
		pinned: false,
	},
	{
		id: 'qwen3.6-35b-a3b-ud-q4_k_m',
		repo_id: 'unsloth/Qwen3.6-35B-A3B-GGUF',
		revision: 'main',
		display_name: 'Qwen3.6 35B A3B UD-Q4_K_M',
		architecture: 'qwen35moe',
		capability: 'text-generation',
		format: 'gguf' as const,
		task: 'text-generation',
		local_path: null,
		params_b: 35,
		quantization: 'UD-Q4_K_M',
		context_train: 262144,
		installed_at: 1749100000,
		primary_file_path: 'C:\\mock\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
		primary_file_size: 22_134_528_992,
		file_count: 1,
		total_size: 22_134_528_992,
		pinned: false,
	},
	{
		id: 'qwen3-tts-1-7b-customvoice-q8_0',
		repo_id: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF',
		revision: 'main',
		display_name: 'Qwen3-TTS 1.7B CustomVoice Q8_0',
		architecture: 'qwen3-tts',
		capability: 'speech',
		format: 'gguf' as const,
		task: 'text-to-speech',
		local_path: null,
		params_b: 1.7,
		quantization: 'Q8_0',
		context_train: undefined,
		installed_at: 1749700000,
		primary_file_path: 'C:\\mock\\models\\qwen3-tts-1.7b-customvoice-q8_0.gguf',
		primary_file_size: 1_840_000_000,
		primary_rel_path: 'qwen3-tts-1.7b-customvoice-q8_0.gguf',
		primary_sha256: null,
		file_count: 1,
		total_size: 1_840_000_000,
		pinned: false,
	},
	// Base talker — the cloning-capable variant (speaker encoder ships
	// only in Base checkpoints); exercises the clone UI in Studio Speech.
	{
		id: 'qwen3-tts-1-7b-base-q8_0',
		repo_id: 'Qwen/Qwen3-TTS-12Hz-1.7B-Base-GGUF',
		revision: 'main',
		display_name: 'Qwen3-TTS 1.7B Base Q8_0',
		architecture: 'qwen3-tts',
		capability: 'speech',
		format: 'gguf' as const,
		task: 'text-to-speech',
		local_path: null,
		params_b: 1.7,
		quantization: 'Q8_0',
		context_train: undefined,
		installed_at: 1749710000,
		primary_file_path: 'C:\\mock\\models\\qwen3-tts-1.7b-base-q8_0.gguf',
		primary_file_size: 1_840_000_000,
		primary_rel_path: 'qwen3-tts-1.7b-base-q8_0.gguf',
		primary_sha256: null,
		file_count: 1,
		total_size: 1_840_000_000,
		pinned: false,
	},
	// Dir-layout installs (M9) — exercise format/task badges + the Studio label.
	{
		id: 'sdxl-turbo',
		repo_id: 'stabilityai/sdxl-turbo',
		revision: 'main',
		display_name: 'SDXL-Turbo',
		architecture: undefined,
		capability: 'image-gen',
		format: 'diffusers-dir' as const,
		task: 'text-to-image',
		local_path: 'C:\\mock\\models\\sdxl-turbo',
		params_b: undefined,
		quantization: undefined,
		context_train: undefined,
		installed_at: 1749000000,
		primary_file_path: 'C:\\mock\\models\\sdxl-turbo\\model_index.json',
		primary_file_size: 600,
		primary_rel_path: 'model_index.json',
		primary_sha256: null,
		file_count: 12,
		total_size: 15_500_000_000,
		pinned: false,
	},
	{
		id: 'sd15',
		repo_id: 'stable-diffusion-v1-5/stable-diffusion-v1-5',
		revision: 'main',
		display_name: 'Stable Diffusion 1.5',
		architecture: undefined,
		capability: 'image-gen',
		format: 'diffusers-dir' as const,
		task: 'text-to-image',
		local_path: 'C:\\mock\\models\\sd15',
		params_b: undefined,
		quantization: undefined,
		context_train: undefined,
		installed_at: 1748900000,
		primary_file_path: 'C:\\mock\\models\\sd15\\model_index.json',
		primary_file_size: 540,
		primary_rel_path: 'model_index.json',
		primary_sha256: null,
		file_count: 11,
		total_size: 5_200_000_000,
		pinned: false,
	},
	{
		id: 'faster-whisper-large-v3',
		repo_id: 'Systran/faster-whisper-large-v3',
		revision: 'main',
		display_name: 'faster-whisper large-v3',
		architecture: undefined,
		capability: 'speech',
		format: 'safetensors' as const,
		task: 'automatic-speech-recognition',
		local_path: 'C:\\mock\\models\\faster-whisper-large-v3',
		params_b: undefined,
		quantization: undefined,
		context_train: undefined,
		installed_at: 1749100000,
		primary_file_path: 'C:\\mock\\models\\faster-whisper-large-v3\\model.bin',
		primary_file_size: 3_090_000_000,
		primary_rel_path: 'model.bin',
		primary_sha256: null,
		file_count: 5,
		total_size: 3_095_000_000,
		pinned: false,
	},
	{
		id: 'wan21-t2v-1-3b',
		repo_id: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
		revision: 'main',
		display_name: 'Wan2.1 T2V 1.3B',
		architecture: undefined,
		capability: 'video-gen',
		format: 'diffusers-dir' as const,
		task: 'text-to-video',
		local_path: 'C:\\mock\\models\\wan21-t2v-1.3b',
		params_b: 1.3,
		quantization: undefined,
		context_train: undefined,
		installed_at: 1749200000,
		primary_file_path: 'C:\\mock\\models\\wan21-t2v-1.3b\\model_index.json',
		primary_file_size: 700,
		primary_rel_path: 'model_index.json',
		primary_sha256: null,
		file_count: 9,
		total_size: 5_400_000_000,
		pinned: false,
	},
	// Single-file diffusion GGUFs (ComfyUI-style — DiT transformer only).
	// Routed to Studio Image/Video by the resolver with gguf_diffusion + a
	// companion_arch; generation is gated on companion files (P2-P4). Mirrors
	// the three real installed DB rows.
	{
		id: 'qwen-image-2512-gguf',
		repo_id: 'unsloth/Qwen-Image-2512-GGUF',
		revision: 'main',
		display_name: 'Qwen-Image 2512 Q6_K',
		architecture: 'qwen_image',
		capability: 'other',
		format: 'gguf' as const,
		task: undefined,
		local_path: null,
		params_b: undefined,
		quantization: 'Q6_K',
		context_train: undefined,
		installed_at: 1750500000,
		primary_file_path: 'C:\\mock\\models\\qwen-image-2512-Q6_K.gguf',
		primary_file_size: 16_800_000_000,
		primary_rel_path: 'qwen-image-2512-Q6_K.gguf',
		primary_sha256: null,
		file_count: 1,
		total_size: 16_800_000_000,
		pinned: false,
	},
	{
		id: 'ltx-2-3-gguf',
		repo_id: 'unsloth/LTX-2.3-GGUF',
		revision: 'main',
		display_name: 'LTX 2.3 22B dev UD Q5_K_M',
		architecture: 'ltxv',
		capability: 'other',
		format: 'gguf' as const,
		task: undefined,
		local_path: null,
		params_b: undefined,
		quantization: 'Q5_K_M',
		context_train: undefined,
		installed_at: 1750510000,
		primary_file_path: 'C:\\mock\\models\\ltx-2.3-22b-dev-UD-Q5_K_M.gguf',
		primary_file_size: 18_300_000_000,
		primary_rel_path: 'ltx-2.3-22b-dev-UD-Q5_K_M.gguf',
		primary_sha256: null,
		file_count: 1,
		total_size: 18_300_000_000,
		pinned: false,
	},
	{
		id: 'wan22-i2v-a14b-gguf',
		repo_id: 'QuantStack/Wan2.2-I2V-A14B-GGUF',
		revision: 'main',
		display_name: 'Wan2.2 I2V A14B LowNoise Q8_0',
		architecture: 'wan',
		capability: 'other',
		format: 'gguf' as const,
		task: undefined,
		local_path: null,
		params_b: undefined,
		quantization: 'Q5_1',
		context_train: undefined,
		installed_at: 1750520000,
		primary_file_path: 'C:\\mock\\models\\Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf',
		primary_file_size: 15_400_000_000,
		primary_rel_path: 'Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf',
		primary_sha256: null,
		file_count: 1,
		total_size: 15_400_000_000,
		pinned: false,
	},
	{
		id: 'vibevoice-1-5b',
		repo_id: 'vibevoice/VibeVoice-1.5B',
		revision: 'main',
		display_name: 'VibeVoice 1.5B',
		architecture: undefined,
		capability: 'speech',
		format: 'safetensors' as const,
		task: 'text-to-speech',
		local_path: 'C:\\mock\\models\\vibevoice-1.5b',
		params_b: 2.7,
		quantization: undefined,
		context_train: undefined,
		installed_at: 1749400000,
		primary_file_path: 'C:\\mock\\models\\vibevoice-1.5b\\model-00001-of-00003.safetensors',
		primary_file_size: 4_980_000_000,
		primary_rel_path: 'model-00001-of-00003.safetensors',
		primary_sha256: null,
		file_count: 9,
		total_size: 5_400_000_000,
		pinned: false,
	},
	{
		id: 'kokoro-82m',
		repo_id: 'hexgrad/Kokoro-82M',
		revision: 'main',
		display_name: 'Kokoro 82M',
		architecture: undefined,
		capability: 'speech',
		format: 'safetensors' as const,
		task: 'text-to-speech',
		local_path: 'C:\\mock\\models\\kokoro-82m',
		params_b: 0.082,
		quantization: undefined,
		context_train: undefined,
		installed_at: 1749300000,
		primary_file_path: 'C:\\mock\\models\\kokoro-82m\\kokoro-v1_0.pth',
		primary_file_size: 327_000_000,
		primary_rel_path: 'kokoro-v1_0.pth',
		primary_sha256: null,
		file_count: 3,
		total_size: 340_000_000,
		pinned: false,
	},
];

/** Catalog fixtures with resolved capability records, as the backend returns them. */
const MODELS = MODELS_RAW.map((m) => ({
	...m,
	task: m.task as string | null,
	caps: resolveMockCaps({ ...m, task: m.task ?? null }),
}));

const settings = new Map<string, string>();

// ---------------------------------------------------------------------------
// Fixtures — Runtime v2 (M8): Python worker, VRAM ledger, generation jobs
// ---------------------------------------------------------------------------

const WORKER_PIPELINES: PipelineCapability[] = [
	{
		task: 'text_to_image', summary: 'Diffusers image pipelines (SD 1.5 / SDXL / Flux)',
		available: true, missing: [], hidden: false, formats: ['diffusers-dir'],
	},
	{
		task: 'text_to_video', summary: 'Diffusers video pipelines (LTX-Video / Wan)',
		available: true, missing: [], hidden: false, formats: ['diffusers-dir'],
	},
	{
		task: 'text_to_speech', summary: 'Kokoro TTS (82M, voice picker)',
		available: true, missing: [], hidden: false, formats: ['kokoro-repo'],
	},
	{
		task: 'vibevoice', summary: 'Voice cloning TTS from a reference audio sample (VibeVoice)',
		available: true, missing: [], hidden: false, formats: ['vibevoice-dir'],
	},
	{
		task: 'transcribe', summary: 'faster-whisper speech-to-text with timestamps',
		available: false, missing: ['faster_whisper'], hidden: false, formats: ['ctranslate2-dir'],
	},
	{ task: 'echo', summary: 'Internal test pipeline', available: true, missing: [], hidden: true, formats: [],
	},
];

// The worker is auto-managed (installs on first run, starts on demand) — the
// mock starts from scratch and walks not_installed → installing → starting →
// ready by itself on first contact, mirroring `worker::spawn_auto_manage`.
let workerState: WorkerStatus = { phase: 'not_installed' };
let workerAutoFlowStarted = false;

function startWorkerAutoFlow() {
	if (workerAutoFlowStarted) return;
	workerAutoFlowStarted = true;
	void (async () => {
		await sleep(600);
		if (workerState.phase !== 'not_installed') return; // someone raced us
		const stages: [string, number][] = [
			['resolving python', 8],
			['downloading wheels', 35],
			['downloading wheels', 72],
			['installing packages', 90],
			['verifying venv', 100],
		];
		const total = 6_400_000_000;
		for (const [stage, pct] of stages) {
			setWorkerState({
				phase: 'installing', variant: 'cuda', stage, pct,
				bytes: Math.round((total * pct) / 100), total,
			});
			await sleep(450);
		}
		setWorkerState({ phase: 'starting' });
		await sleep(900);
		setWorkerState({ phase: 'ready', variant: 'cuda', version: '0.1.0', pipelines: WORKER_PIPELINES,
		});
	})();
}

const WORKER_LOG_LINES = [
	'[worker] openharness-worker 0.1.0 starting (torch 2.7.0+cu124)',
	'[worker] CUDA available: NVIDIA GeForce RTX 4090 (mock)',
	'[worker] pipelines registered: text_to_image, text_to_video, text_to_speech, vibevoice, transcribe, echo',
	'[worker] ready — waiting for requests on stdin',
];

function setWorkerState(s: WorkerStatus) {
	workerState = s;
	emit('subscribe_worker_status', s);
}

// VRAM ledger — text engine pre-resident to match ENGINE_READY, plus one
// speech pipeline resident so the studio pickers demo loaded-first grouping
// (Kokoro loaded ●, VibeVoice/whisper installed-but-unloaded) out of the box.
let residents: ResidentInfo[] = [
	{
		id: 'text',
		kind: 'text_engine',
		label: 'Qwen2.5 Coder 7B Instruct Q4_K_M',
		model_id: MODEL_ID,
		est_mb: 7421,
		actual_mb: null,
	},
	{
		id: 'pipeline:text_to_speech',
		kind: 'pipeline',
		label: 'Kokoro 82M',
		model_id: 'kokoro-82m',
		est_mb: 1200,
		actual_mb: 1180,
	},
];

// The pre-loaded Qwen2.5 is a llama.cpp model, so it can't be running without that engine.
if (!mockEngineSetup().llama) {
	residents = residents.filter((r) => r.id !== 'text');
	engineState = { phase: 'idle' };
}

/**
 * Simulate a machine without an NVIDIA GPU: set localStorage
 * `oh.mock.gpu` to `none` and reload. NVML reports no GPU, the ledger has no
 * VRAM total, and fit estimates come back `gpu: false` (CPU inference).
 */
function mockNoGpu(): boolean {
	try {
		return globalThis.localStorage?.getItem('oh.mock.gpu') === 'none';
	} catch {
		return false;
	}
}

/**
 * Stream mock chat replies slowly so in-flight states (the dock's "In use"
 * lock, switching conversations mid-reply) can be inspected: set localStorage
 * `oh.mock.slowchat` to a per-token delay in milliseconds and reload.
 */
function mockTokenDelayMs(): number {
	try {
		const v = Number(globalThis.localStorage?.getItem('oh.mock.slowchat'));
		return Number.isFinite(v) && v > 0 ? v : 28;
	} catch {
		return 28;
	}
}

/** The NVML feed as the GPU actor would report it. */
function mockGpuFeed() {
	return mockNoGpu() ? { kind: 'no_gpu' as const } : mockGpuStatus();
}

function ledgerSnapshot(): LedgerSnapshot {
	const used = residents.reduce((s, r) => s + (r.actual_mb ?? r.est_mb), 0);
	if (mockNoGpu()) {
		return { gpu: false, total_mb: 0, free_mb: 0, residents: residents.map((r) => ({ ...r })),
		};
	}
	return {
		gpu: true,
		total_mb: GPU_OK.vram_total_mb,
		free_mb: Math.max(0, GPU_OK.vram_total_mb - used - 1200 /* desktop/other processes */),
		residents: residents.map((r) => ({ ...r })),
	};
}

function emitLedger() {
	emit('subscribe_orchestrator_ledger', ledgerSnapshot());
	// NVML would see the residents come and go; keep the mock GPU feed honest.
	emit('gpu_subscribe_metrics', mockGpuFeed());
}

/** Desktop + other processes holding VRAM outside the ledger. */
const MOCK_VRAM_BASELINE_MB = 1200;

/** GPU metrics whose used/free track the mock ledger's residents. */
function mockGpuStatus() {
	const resident = residents.reduce((s, r) => s + (r.actual_mb ?? r.est_mb), 0);
	const used = Math.min(GPU_OK.vram_total_mb, MOCK_VRAM_BASELINE_MB + resident);
	return {
		...GPU_OK,
		vram_used_mb: used,
		vram_free_mb: GPU_OK.vram_total_mb - used,
		sampled_at: now(),
	};
}

/** Physical RAM — mirrors commands::system::system_memory (64 GB box). */
function mockSystemMemory(): SystemMemory {
	const total = 65_536;
	const used = 21_340 + Math.round(Math.random() * 180);
	return { total_mb: total, used_mb: used, available_mb: total - used };
}

function removeResident(id: string) {
	residents = residents.filter((r) => r.id !== id);
	emitLedger();
}

/**
 * Mirror the orchestrator's eviction gate: needs-eviction only when the load
 * doesn't fit free VRAM (own slot reclaimed) but WOULD fit after freeing the
 * other residents. Returns the needs_eviction payload, or null to proceed.
 */
function mockGate(residentId: string, est: number, force: boolean) {
	if (force) return null;
	const snap = ledgerSnapshot();
	const own = residents.find((r) => r.id === residentId);
	const free = snap.free_mb + (own ? (own.actual_mb ?? own.est_mb) : 0);
	const others = snap.residents.filter((r) => r.id !== residentId);
	const reclaimable = others.reduce((s, r) => s + (r.actual_mb ?? r.est_mb), 0);
	if (est > free && others.length > 0 && est <= free + reclaimable) {
		return { outcome: 'needs_eviction' as const, required_mb: est, free_mb: free, candidates: others,
		};
	}
	return null;
}

// ---------------------------------------------------------------------------
// Load planner — a simplified mirror of src-tauri/src/planner/. Same shapes and
// warning strings; the numbers follow the same model (bytes touched per token
// per tier over that tier's bandwidth) with fewer cases, tuned so the canned
// RTX 4090 / 64 GB / 990 PRO profile gives realistic plans.
// ---------------------------------------------------------------------------

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

let ssdMeasurement: SsdMeasurement | null = {
	mount: 'C:\\',
	gbps: 6.8,
	bytes_read: 4 * GIB,
	seconds: 0.63,
	measured_at: 1762300000,
	file: 'C:\\mock\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
};

function mockHardware(): HardwareProfile {
	const free = ledgerSnapshot().free_mb;
	return {
		gpu: mockNoGpu() ? null : {
			name: GPU_OK.name,
			vram_total_bytes: GPU_OK.vram_total_mb * MIB,
			vram_free_bytes: free * MIB,
		},
		ram_total_bytes: 64 * GIB,
		ram_available_bytes: 48_000_000_000,
		commit_available_bytes: 64 * GIB,
		storage: {
			mount: 'C:\\',
			model: 'Samsung SSD 990 PRO 2TB',
			kind: 'ssd',
			free_bytes: 812_000_000_000,
			total_bytes: 2_000_398_934_016,
		},
		bandwidth: {
			vram_gbps: 1008,
			pcie_gbps: 25,
			ram_gbps: 60,
			ssd_gbps: ssdMeasurement?.gbps ?? 3.5,
			ssd_measured: ssdMeasurement !== null,
		},
		cpu_threads: 32,
	};
}

/** Per-model tensor accounting the real backend reads from the GGUF header. */
interface MockAccounting {
	architecture: string;
	layers: number;
	/** token_embd: llama.cpp keeps it in host RAM. */
	token_embd: number;
	/** Always-used weights that go to the GPU (attention, norms, shared, output). */
	always: number;
	/** Routed experts (0 = dense). */
	experts: number;
	/** top-k / expert count. */
	active_frac: number;
	/** Context sizing (mirrors planner::context): full-attention layers… */
	full_layers: number;
	/** …their K and V elements per token (MLA: latent as K, no V)… */
	k_per_layer: number;
	v_per_layer: number;
	/** …and DSA indexer elements per token (0 = none). */
	idx_per_layer: number;
	mla: boolean;
	/** Sliding-window layers (same K/V widths) and their window. */
	swa_layers?: number;
	swa_window?: number;
	/** Fixed recurrent state per sequence, bytes. */
	state_per_seq: number;
	n_vocab: number;
	n_embd: number;
	expert_used: number;
	expert_ffn: number;
	/** Largest routed-expert tensor (host-expert prompt batches copy it). */
	max_expert_tensor: number;
}

const MOCK_ACCOUNTING: Record<string, MockAccounting> = {
	[MODEL_ID]: {
		architecture: 'qwen2',
		layers: 28,
		token_embd: 300_000_000,
		always: 4_600_000_000,
		experts: 0,
		active_frac: 0,
		full_layers: 28,
		k_per_layer: 4 * 128,
		v_per_layer: 4 * 128,
		idx_per_layer: 0,
		mla: false,
		state_per_seq: 0,
		n_vocab: 152064,
		n_embd: 3584,
		expert_used: 0,
		expert_ffn: 0,
		max_expert_tensor: 0,
	},
	'llama-3.1-8b-instruct-q5_k_m': {
		architecture: 'llama',
		layers: 32,
		token_embd: 380_000_000,
		always: 5_320_000_000,
		experts: 0,
		active_frac: 0,
		full_layers: 32,
		k_per_layer: 8 * 128,
		v_per_layer: 8 * 128,
		idx_per_layer: 0,
		mla: false,
		state_per_seq: 0,
		n_vocab: 128256,
		n_embd: 4096,
		expert_used: 0,
		expert_ffn: 0,
		max_expert_tensor: 0,
	},
	// 45 decode blocks: 34 KDA linear-attention (fixed f32 state) + 11 sparse
	// MLA (512-wide latent, 3 × 128 indexer keys); top-8 of 288 experts.
	'glm-5.3-flash-ud-q2_k_xl': {
		architecture: 'glm5-next',
		layers: 45,
		token_embd: 620_000_000,
		always: 5_800_000_000,
		experts: 102_300_000_000,
		active_frac: 8 / 288,
		full_layers: 11,
		k_per_layer: 512,
		v_per_layer: 0,
		idx_per_layer: 384,
		mla: true,
		state_per_seq: 152_633_344,
		n_vocab: 154880,
		n_embd: 4096,
		expert_used: 8,
		expert_ffn: 2048,
		max_expert_tensor: 1_283_457_024,
	},
	// 40 blocks: 30 gated delta-net + 10 full attention (2 KV heads × 256);
	// every layer MoE, top-8 of 256.
	'qwen3.6-35b-a3b-ud-q4_k_m': {
		architecture: 'qwen35moe',
		layers: 40,
		token_embd: 540_000_000,
		always: 1_600_000_000,
		experts: 19_994_528_992,
		active_frac: 8 / 256,
		full_layers: 10,
		k_per_layer: 512,
		v_per_layer: 512,
		idx_per_layer: 0,
		mla: false,
		state_per_seq: 65_863_680,
		n_vocab: 248320,
		n_embd: 2048,
		expert_used: 8,
		expert_ffn: 512,
		max_expert_tensor: 220_200_960,
	},
};

/** Bytes of one cache row of `elems` elements (ggml block sizes). */
function kvRowBytes(t: KvType, elems: number): number {
	if (t === 'f16') return elems * 2;
	return Math.ceil(elems / 32) * (t === 'q8_0' ? 34 : 18);
}

/** Mirror of `planner::context::context_memory`. */
function mockContextMemory(acct: MockAccounting, p: ContextParams): ContextMemory {
	const nSeq = Math.max(1, p.n_parallel);
	const ub = Math.max(1, p.ubatch);
	const padTo = (n: number, to: number) => Math.ceil(n / to) * to;
	const ctx = padTo(Math.max(p.ctx, 256), 256 * nSeq);
	const cells = ctx / nSeq;
	const kvV = acct.mla ? p.kv_k : p.kv_v;
	const row = kvRowBytes(p.kv_k, acct.k_per_layer) + kvRowBytes(kvV, acct.v_per_layer);
	const kvRow = acct.full_layers * row;
	const idxRow = acct.full_layers * kvRowBytes(p.kv_k, acct.idx_per_layer);
	const swaLayers = acct.swa_layers ?? 0;
	const swaCells = swaLayers > 0 ? Math.min(cells, padTo((acct.swa_window ?? 0) + ub, 256)) : 0;
	const swaBytes = swaLayers * row * swaCells * nSeq;
	const maskB = acct.idx_per_layer > 0 ? 1 : 2;
	const masks = (kvRow > 0 ? cells * ub * maskB : 0) + (swaBytes > 0 ? swaCells * ub * maskB : 0);
	const act = ub * acct.n_embd * 16;
	const recurrent = acct.state_per_seq > 0 ? ub * acct.n_embd * 4 : 0;
	const base = ub * acct.n_vocab * 4 + masks + act + recurrent;
	const split =
		p.experts_on_host && p.ubatch >= 32
			? acct.max_expert_tensor +
				ub * Math.max(1, acct.expert_used) * Math.max(1, acct.expert_ffn) * 24 +
				masks
			: 0;
	return {
		ctx,
		kv_v: kvV,
		kv_bytes: kvRow * cells * nSeq,
		swa_bytes: swaBytes,
		indexer_bytes: idxRow * cells * nSeq,
		state_bytes: acct.state_per_seq * nSeq,
		compute_bytes: Math.max(base, split),
		host_compute_bytes: masks + act,
		bytes_per_token: kvRow + idxRow + (kvRow > 0 ? (ub * maskB) / nSeq : 0),
	};
}

function contextCacheBytes(c: ContextMemory): number {
	return c.kv_bytes + c.swa_bytes + c.indexer_bytes + c.state_bytes;
}

/** Planner's compute reservation: prediction plus its 8 % safety margin. */
const COMPUTE_MARGIN = 1.08;
const CUDA_CONTEXT = 450 * MIB;

function mockAccountingFor(modelId: string): MockAccounting {
	const model = MODELS.find((m) => m.id === modelId);
	if (!model) throw new Error(`Model not found: ${modelId}`);
	if (model.caps.runtime !== 'llama-server') {
		throw new Error('Load plans cover llama-server GGUF models only.');
	}
	return (
		MOCK_ACCOUNTING[modelId] ?? {
			architecture: model.architecture ?? 'unknown',
			layers: 32,
			token_embd: model.total_size * 0.05,
			always: model.total_size * 0.95,
			experts: 0,
			active_frac: 0,
			full_layers: 32,
			k_per_layer: 1024,
			v_per_layer: 1024,
			idx_per_layer: 0,
			mla: false,
			state_per_seq: 0,
			n_vocab: 128000,
			n_embd: 4096,
			expert_used: 0,
			expert_ffn: 0,
			max_expert_tensor: 0,
		}
	);
}

const TIER_RANK: Record<TierMix, number> = { vram: 0, ram: 1, ssd: 2 };

function speedLabel(tps: number): SpeedLabel {
	return tps >= FAST_TPS ? 'fast' : tps >= USABLE_TPS ? 'usable' : 'slow';
}

interface MockBudget {
	total: number;
	free: number;
	reclaimable: Reclaimable[];
}

/** Free VRAM for a text load: free now plus the text slot it replaces. */
function mockTextBudget(): MockBudget {
	const snap = ledgerSnapshot();
	const own = residents.find((r) => r.id === 'text');
	const ownMb = own ? (own.actual_mb ?? own.est_mb) : 0;
	return {
		total: snap.total_mb * MIB,
		free: (snap.free_mb + ownMb) * MIB,
		reclaimable: residents
			.filter((r) => r.id !== 'text')
			.map((r) => ({
				id: r.id,
				label: r.label,
				bytes: (r.actual_mb ?? r.est_mb) * MIB,
			})),
	};
}

/** Graphics memory a text plan may use: what is free, capped by the user's limit. */
function mockVramUsable(budget: MockBudget, ov: PlanOverrides): { usable: number; limited: boolean } {
	const free = Math.max(0, budget.free - Math.max(512 * MIB, budget.total * 0.03));
	const limit = ov.vram_limit_mib === undefined ? undefined : ov.vram_limit_mib * MIB;
	const limited = limit !== undefined && limit <= free;
	return { usable: limited ? limit : free, limited };
}

/** System memory a text plan may use: what is free after headroom, capped by the user's limit. */
function mockRamAvail(hw: HardwareProfile, ov: PlanOverrides): { avail: number; limited: boolean } {
	const free = Math.max(0, hw.ram_available_bytes - 4 * GIB);
	const limited = ov.ram_limit_mib !== undefined && ov.ram_limit_mib * MIB < free;
	return { avail: limited ? ov.ram_limit_mib! * MIB : free, limited };
}

/**
 * The binding system memory limit, when the context memory that must stay in
 * RAM is larger than it (the RAM tier is then that context memory alone).
 */
function mockRamLimitShort(hw: HardwareProfile, ov: PlanOverrides, plan: LoadPlan): number | null {
	const { avail, limited } = mockRamAvail(hw, ov);
	return limited && plan.tiers.ram > avail ? avail : null;
}

const OFF_CARD_WARNING =
	'The graphics memory limit is below what llama-server needs to use the card at all, so the model runs on the CPU and the card is left alone.';

/**
 * llama-server placement under the user's memory limits. A graphics memory
 * limit too small for the card to be used at all plans on the CPU with the
 * card left out, rather than refusing. The limit alone decides that: when
 * free graphics memory is what falls short the plan is refused as it is
 * without a limit, and a pinned GPU expert placement stays pinned.
 */
function mockPlace(
	acct: MockAccounting,
	hw: HardwareProfile,
	budget: MockBudget,
	ctx: number,
	kv: KvType,
	ov: PlanOverrides,
): LoadPlan {
	const onCard = (t: TierBytes) => t.vram_weights + t.vram_expert_cache + t.vram_kv + t.vram_overhead;
	const limit = ov.vram_limit_mib === undefined ? undefined : ov.vram_limit_mib * MIB;
	const pinsGpu = (ov.gpu_expert_layers ?? 0) > 0 || (ov.expert_cache_mib ?? 0) > 0;
	// Off the card only when no on-card layout fits the limit on an otherwise
	// empty card (placement under the limit reaches down to zero layers).
	const offCard =
		!!hw.gpu &&
		limit !== undefined &&
		!pinsGpu &&
		onCard(mockPlaceOn(acct, hw, { ...budget, free: Infinity }, ctx, kv, ov).tiers) > limit;
	const plan = mockPlaceOn(acct, offCard ? { ...hw, gpu: null } : hw, budget, ctx, kv, ov);
	if (offCard)
		plan.warnings = plan.warnings.map((w) => (w.startsWith('No NVIDIA GPU') ? OFF_CARD_WARNING : w));
	const p = plan.tiers;
	if (ov.vram_limit_mib !== undefined && p.vram_weights + p.vram_expert_cache + p.vram_kv + p.vram_overhead === 0) {
		// A CUDA build would still create its context on the card with `-ngl 0`.
		plan.args.splice(plan.args.indexOf('-ub') + 2, 0, '--device', 'none');
	}
	return plan;
}

function mockPlaceOn(
	acct: MockAccounting,
	hw: HardwareProfile,
	budget: MockBudget,
	ctx: number,
	kv: KvType,
	ov: PlanOverrides,
): LoadPlan {
	const bw = hw.bandwidth;
	const { usable, limited: vramLimited } = mockVramUsable(budget, ov);
	const { avail: ramAvail, limited: ramLimited } = mockRamAvail(hw, ov);
	const kvV: KvType = acct.mla ? kv : (ov.kv_type_v ?? kv);
	const ctxParams = (ubatch: number, expertsOnHost: boolean): ContextParams => ({
		ctx,
		kv_k: kv,
		kv_v: kvV,
		n_parallel: 1,
		ubatch,
		experts_on_host: expertsOnHost,
	});
	// Context memory and compute claim graphics memory before any expert cache.
	let context = mockContextMemory(acct, ctxParams(512, false));
	const kvBytes = contextCacheBytes(context);
	const overheadFor = (c: ContextMemory) => CUDA_CONTEXT + Math.round(c.compute_bytes * COMPUTE_MARGIN);
	const perToken = acct.always + acct.experts * acct.active_frac;
	const allowSsd = ov.allow_ssd ?? true;
	const layerOverhead = acct.layers * 1e-4;
	const moeLayers = acct.experts > 0 ? acct.layers : 0;

	let tiers: TierBytes;
	let gpuLayers = -1;
	let gpuExpertLayers = moeLayers;
	let cacheBytes = 0;
	let ubatch = 512;
	let parts: [Bottleneck, number][];
	let promptTps: number;
	// Context memory and compute that must stay in RAM; never streamed.
	let hostFixed = context.host_compute_bytes;

	const fullVram = acct.always + acct.experts + kvBytes + overheadFor(context);
	if (hw.gpu && fullVram <= usable && ov.gpu_expert_layers === undefined) {
		tiers = {
			vram_weights: acct.always + acct.experts,
			vram_expert_cache: 0,
			vram_kv: kvBytes,
			vram_overhead: overheadFor(context),
			ram: acct.token_embd + context.host_compute_bytes,
			ssd: 0,
		};
		parts = [
			['vram', perToken / (bw.vram_gbps * 1e9 * 0.8)],
			['overhead', layerOverhead],
		];
		promptTps = Math.min(8000, 2.2e13 / perToken);
	} else if (
		hw.gpu &&
		acct.experts > 0 &&
		// Every always-used weight on the GPU: only when that fits, or when
		// the expert placement is pinned (then it is honoured and refused).
		(ov.gpu_expert_layers !== undefined ||
			ov.expert_cache_mib !== undefined ||
			acct.always + kvBytes + overheadFor(mockContextMemory(acct, ctxParams(2048, true))) <= usable)
	) {
		// Experts on the CPU (all or --n-cpu-moe), hot ones cached on the GPU.
		ubatch = 2048;
		context = mockContextMemory(acct, ctxParams(ubatch, true));
		const overhead = overheadFor(context);
		const perLayer = acct.experts / moeLayers;
		gpuExpertLayers = Math.min(moeLayers, ov.gpu_expert_layers ?? 0);
		const gpuExperts = gpuExpertLayers * perLayer;
		const spare = usable - acct.always - gpuExperts - kvBytes - overhead;
		cacheBytes =
			ov.expert_cache_mib !== undefined
				? ov.expert_cache_mib * MIB
				: spare >= 512 * MIB
					? Math.floor(spare / (256 * MIB)) * 256 * MIB
					: 0;
		const hostExperts = acct.experts - gpuExperts;
		// Experts held by the cache go cold in the page cache.
		const hostNeed = hostExperts - cacheBytes + acct.token_embd;
		hostFixed = context.host_compute_bytes;
		const ram = Math.min(hostNeed, Math.max(0, ramAvail - context.host_compute_bytes));
		tiers = {
			vram_weights: acct.always + gpuExperts,
			vram_expert_cache: cacheBytes,
			vram_kv: kvBytes,
			vram_overhead: overhead,
			ram: ram + context.host_compute_bytes,
			ssd: Math.max(0, hostNeed - ram),
		};
		// Hottest-first hit rate: a tier holding fraction f of the experts
		// serves f^0.15 of expert reads.
		const hit = (f: number) => Math.min(1, Math.max(0, f)) ** 0.15;
		const hc = hostExperts > 0 ? hit(cacheBytes / hostExperts) : 0;
		const hr = hostExperts > 0 ? hit((cacheBytes + ram - acct.token_embd) / hostExperts) - hc : 0;
		const hostWeightsInRam = ram;
		const hs = Math.max(0, 1 - hc - hr);
		const traffic = hostExperts * acct.active_frac;
		parts = [
			[
				'vram',
				(acct.always + gpuExperts * acct.active_frac + traffic * hc) / (bw.vram_gbps * 1e9 * 0.8),
			],
			['ram', (traffic * hr) / (bw.ram_gbps * 1e9 * 0.6)],
			['ssd', (traffic * hs) / (bw.ssd_gbps * 1e9 * 0.45)],
			['overhead', layerOverhead],
		];
		// Prompt batches stream every host expert to the GPU once per ubatch.
		const stream =
			(hostWeightsInRam - acct.token_embd) / (bw.pcie_gbps * 1e9) + tiers.ssd / (bw.ssd_gbps * 1e9);
		promptTps = Math.min(8000, (ubatch / Math.max(stream, 1e-3)) * 0.5);
	} else if (hw.gpu) {
		// Partial -ngl: as many layers as fit go on the GPU, each with its own
		// context cache, down to none (the card then holds only the CUDA
		// context and compute buffers). A MoE model that cannot keep all its
		// always-used weights on the GPU lands here with every expert on the host.
		const moe = acct.experts > 0;
		if (moe) {
			ubatch = 2048;
			context = mockContextMemory(acct, ctxParams(ubatch, true));
			gpuExpertLayers = 0;
		}
		const overhead = overheadFor(context);
		const perLayer = acct.always / acct.layers;
		const kvPerLayer = kvBytes / acct.layers;
		const n = Math.max(
			0,
			Math.min(acct.layers, Math.floor((usable - overhead) / (perLayer + kvPerLayer))),
		);
		gpuLayers = n;
		const gpuKv = Math.floor(n * kvPerLayer);
		// Context caches of the layers left on the host stay in RAM.
		hostFixed = context.host_compute_bytes + (kvBytes - gpuKv);
		const hostWeights = acct.always - n * perLayer + acct.experts;
		const host = hostWeights + acct.token_embd;
		const ram = Math.min(host, Math.max(0, ramAvail - hostFixed));
		tiers = {
			vram_weights: n * perLayer,
			vram_expert_cache: 0,
			vram_kv: gpuKv,
			vram_overhead: overhead,
			ram: ram + hostFixed,
			ssd: Math.max(0, host - ram),
		};
		// Per token the host reads its dense layers and the active share of its experts.
		const read = moe
			? (hostWeights - acct.experts * (1 - acct.active_frac)) / Math.max(hostWeights, 1)
			: 1;
		parts = [
			['vram', (n * perLayer) / (bw.vram_gbps * 1e9 * 0.8)],
			['cpu', (Math.max(0, ram - acct.token_embd) * read) / (bw.ram_gbps * 1e9 * 0.6)],
			['ssd', (tiers.ssd * read) / (bw.ssd_gbps * 1e9 * 0.45)],
			['overhead', layerOverhead],
		];
		promptTps = Math.min(2000, 6e12 / perToken);
	} else {
		gpuLayers = 0;
		gpuExpertLayers = 0;
		hostFixed = kvBytes + context.compute_bytes;
		const host = acct.always + acct.experts + acct.token_embd + hostFixed;
		// Weights fill what the context memory leaves; the rest streams.
		const ram = hostFixed + Math.min(host - hostFixed, Math.max(0, ramAvail - hostFixed));
		tiers = {
			vram_weights: 0,
			vram_expert_cache: 0,
			vram_kv: 0,
			vram_overhead: 0,
			ram,
			ssd: Math.max(0, host - ram),
		};
		parts = [
			['cpu', perToken / (bw.ram_gbps * 1e9 * 0.6)],
			['ssd', (tiers.ssd * (acct.experts > 0 ? acct.active_frac : 1)) / (bw.ssd_gbps * 1e9 * 0.45)],
		];
		promptTps = Math.min(400, 2e12 / perToken);
	}

	const vramTotal =
		tiers.vram_weights + tiers.vram_expert_cache + tiers.vram_kv + tiers.vram_overhead;
	const fitsVram = !hw.gpu || vramTotal <= usable;
	const fitsRam = hostFixed <= ramAvail;
	const feasible = fitsVram && fitsRam && (tiers.ssd === 0 || allowSsd);
	const decode = 1 / parts.reduce((s, [, t]) => s + t, 0);
	const bottleneck = parts.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
	const ramWeights = tiers.ram - acct.token_embd - context.host_compute_bytes;
	const tierMix: TierMix =
		tiers.ssd > 0 ? 'ssd' : ramWeights > 1 || gpuLayers >= 0 ? 'ram' : 'vram';
	const cpuExpertLayers = hw.gpu ? moeLayers - gpuExpertLayers : moeLayers;
	const cacheMib = Math.round(cacheBytes / MIB);

	const extra: string[] = ['-np', '1'];
	if (kv !== 'f16' || kvV !== 'f16') extra.push('-ctk', kv, '-ctv', kvV);
	if (cpuExpertLayers > 0)
		extra.push(...(gpuExpertLayers > 0 ? ['--n-cpu-moe', String(cpuExpertLayers)] : ['--cpu-moe']));
	if (cacheMib > 0) extra.push('--moe-cache-mib', String(cacheMib));
	extra.push('-ub', String(ubatch), '--fit', 'off', '--metrics');
	const args = ['-c', String(ctx), '-ngl', gpuLayers < 0 ? 'all' : String(gpuLayers), '-fa', 'on'];
	args.push('-b', String(Math.max(ubatch, 2048)), ...extra);

	const gb = (b: number) => `${(b / 1e9).toFixed(1)} GB`;
	const warnings: string[] = [];
	if (!hw.gpu) warnings.push('No NVIDIA GPU detected: the model runs on the CPU.');
	if (tiers.ssd > 0) {
		warnings.push(
			`SSD streaming active: ${gb(tiers.ssd)} of weights don't fit in memory and are read from the SSD as needed.`,
		);
		if (!hw.bandwidth.ssd_measured) {
			warnings.push('SSD speed is a default guess; run the SSD benchmark for a better estimate.');
		}
		if (ramLimited)
			warnings.push(
				'The system memory limit sets what is planned to stay in memory. llama-server maps the whole model file, so Windows may keep more of it cached while RAM is free.',
			);
	}
	if (!fitsVram)
		warnings.push(
			vramLimited
				? `These settings need ${gb(vramTotal)} of graphics memory but the graphics memory limit allows ${gb(usable)}.`
				: `These settings need ${gb(vramTotal)} of graphics memory but only ${gb(usable)} is free.`,
		);
	else if (!fitsRam)
		warnings.push(
			ramLimited
				? `These settings need ${gb(tiers.ram)} of system memory for the context window but the system memory limit allows ${gb(ramAvail)}.`
				: "Doesn't fit: not enough memory for the context window. Lower the context length.",
		);
	else if (tiers.ssd > 0 && !allowSsd)
		warnings.push("Doesn't fit in graphics and system memory, and SSD streaming is off.");
	if (gpuLayers >= 0 && hw.gpu) {
		warnings.push(`Only ${gpuLayers} of ${acct.layers} layers fit in graphics memory.`);
	}

	return {
		engine_kind: 'llama_server',
		native: null,
		architecture: acct.architecture,
		ctx,
		ctx_train: 0,
		kv_type: kv,
		kv_type_v: kvV,
		n_parallel: 1,
		context,
		max_ctx: null,
		infeasible_reason: null,
		gpu_layers: gpuLayers,
		gpu_expert_layers: gpuExpertLayers,
		cpu_expert_layers: cpuExpertLayers,
		n_cpu_moe: cpuExpertLayers > 0 && gpuExpertLayers > 0 ? cpuExpertLayers : null,
		expert_cache_mib: cacheMib,
		ubatch,
		tiers,
		tier_mix: tierMix,
		feasible,
		speed: {
			decode_tps: decode,
			prompt_tps: promptTps,
			label: speedLabel(decode),
			raw_decode_tps: decode,
			raw_prompt_tps: promptTps,
			calibration_samples: 0,
			bottleneck,
		},
		args,
		warnings,
		evict_alternative: null,
		overrides: { ...ov },
	};
}

function planRank(p: LoadPlan): [number, number, number] {
	return [p.feasible ? 0 : 1, TIER_RANK[p.tier_mix], -p.speed.decode_tps];
}

function betterPlan(a: LoadPlan, b: LoadPlan): boolean {
	const ra = planRank(a);
	const rb = planRank(b);
	for (let i = 0; i < ra.length; i++) if (ra[i] !== rb[i]) return ra[i] < rb[i];
	return false;
}

/** Mirror of `planner::plan` / `auto_plan` (no overrides = auto best settings). */
/** Native fixture: dense/prefill/KV stay on the CPU, experts stream. */
function mockNativePlan(
	acct: MockAccounting,
	hw: HardwareProfile,
	p: LoadPlan,
	ov: PlanOverrides,
): LoadPlan {
	p.engine_kind = 'ssd_stream';
	p.evict_alternative = null;
	const native = mockEngineSetup().native;
	if (acct.experts > 0 && native !== 'ok') {
		p.feasible = false;
		p.infeasible_reason =
			native === 'missing'
				? "The native engine isn't installed. Install it in engine/native to stream this model without overcommitting RAM."
				: 'The native engine does not report ssd_expert_streaming.';
		p.warnings.push(p.infeasible_reason);
		return p;
	}
	if (acct.experts === 0 || ov.allow_ssd === false) {
		p.feasible = false;
		p.infeasible_reason =
			acct.experts === 0
				? 'Native streaming requires routed experts.'
				: 'SSD streaming is disabled in this plan.';
		p.warnings = [p.infeasible_reason];
		return p;
	}
	p.ubatch = 512;
	p.kv_type = ov.kv_type ?? 'f16';
	p.kv_type_v = acct.mla ? p.kv_type : (ov.kv_type_v ?? p.kv_type);
	const context = (ctx: number) =>
		mockContextMemory(acct, {
			ctx,
			kv_k: p.kv_type,
			kv_v: p.kv_type_v,
			n_parallel: 1,
			ubatch: 512,
			experts_on_host: false,
		});
	const ramFree = Math.max(0, hw.ram_available_bytes - 12 * GIB);
	const ramLimit = ov.ram_limit_mib === undefined ? undefined : ov.ram_limit_mib * MIB;
	const ram = Math.floor(Math.min(ramFree, ramLimit ?? ramFree) / MIB) * MIB;
	// The graphics-memory tier is sized first: Windows charges it to commit.
	// The text model this load replaces gives its graphics memory back; other
	// residents keep their charge.
	const text = residents.find((r) => r.id === 'text');
	const textVram = hw.gpu && text ? (text.actual_mb ?? text.est_mb) * MIB : 0;
	const freeNow = hw.gpu ? Math.min(hw.gpu.vram_total_bytes, hw.gpu.vram_free_bytes + textVram) : 0;
	const vramFree = Math.max(0, freeNow - 5 * GIB);
	const vramAllowed =
		ov.vram_limit_mib === undefined ? vramFree : Math.min(vramFree, ov.vram_limit_mib * MIB);
	const minimumTier = acct.experts / (acct.expert_used / acct.active_frac);
	// The expert tier once `spent` bytes of the allowance hold something else.
	const tierAfter = (spent: number) => {
		const tier = Math.floor(Math.max(0, vramAllowed - 450 * MIB - spent) / MIB) * MIB;
		return tier >= minimumTier ? tier : 0;
	};
	const hostTier = tierAfter(0);
	const limitDropsTier =
		hostTier === 0 && Math.floor(Math.max(0, vramFree - 450 * MIB) / MIB) * MIB >= minimumTier;
	const dense = acct.always + acct.token_embd;
	const margined = (bytes: number) => Math.floor(bytes * 1.08);
	const cardCost = (c: ContextMemory) =>
		acct.token_embd + margined(c.host_compute_bytes) + 3584 * MIB;
	// Measured GLM runs on an RTX 4090: dense layers on the CPU, then on the card.
	const seed = (card: boolean) => (card ? [3.5, 10 * GIB] : [2.25, 12 * GIB]);
	const seededTps = (card: boolean, tier: number) =>
		seed(card)[0] * (tier === 0 ? 0.25 : Math.min(1, tier / seed(card)[1]));
	// [dense layers and context on the card, expert tier]. The dense layers go on
	// the card when the tier left beside them still seeds at least the speed of
	// the CPU layout's larger tier; [0, tier] leaves them on the CPU.
	const layout = (c: ContextMemory): [number, number] => {
		const onCard = acct.always + contextCacheBytes(c) + margined(c.compute_bytes);
		const tier = tierAfter(onCard);
		return tier > 0 && cardCost(c) <= ram && seededTps(true, tier) >= seededTps(false, hostTier)
			? [onCard, tier]
			: [0, hostTier];
	};
	// RAM the engine holds besides the expert cache, then its commit: the mapped
	// model file takes RAM but none.
	const cost = (c: ContextMemory) =>
		layout(c)[0] > 0
			? cardCost(c)
			: dense + contextCacheBytes(c) + margined(c.compute_bytes) + 1536 * MIB;
	const privateBytes = (c: ContextMemory) =>
		layout(c)[0] > 0 ? margined(c.host_compute_bytes) + 1536 * MIB : cost(c) - dense;
	// Windows charges everything the engine puts on the card to its commit.
	const vramCommit = (c: ContextMemory) => {
		const [onCard, tier] = layout(c);
		return tier > 0 ? tier + 450 * MIB + onCard : 0;
	};
	const commit = hw.commit_available_bytes ?? null;
	const commitRoom = (c: ContextMemory) =>
		commit === null ? null : Math.max(0, commit - (8 + 3) * GIB - vramCommit(c));
	const fitsRam = (ctx: number) => cost(context(ctx)) <= ram;
	const fitsCommit = (ctx: number) => {
		const c = context(ctx);
		const room = commitRoom(c);
		return room === null || privateBytes(c) <= room;
	};
	const fits = (ctx: number) => fitsRam(ctx) && fitsCommit(ctx);
	let lo = 0;
	let hi = Math.floor(p.ctx / 256);
	while (lo < hi) {
		const mid = lo + Math.ceil((hi - lo) / 2);
		if (fits(mid * 256)) lo = mid;
		else hi = mid - 1;
	}
	if (ov.ctx === undefined) p.ctx = Math.max(256, Math.min(p.ctx, lo * 256));
	p.context = context(p.ctx);
	p.feasible = fits(p.ctx);
	p.max_ctx = !p.feasible && lo > 0 ? lo * 256 : null;
	const bound = !fitsRam(p.ctx)
		? ramLimit !== undefined && ramLimit < ramFree
			? `within the ${(ramLimit / 1e9).toFixed(1)} GB system memory limit`
			: 'while leaving 12 GiB RAM free'
		: "within Windows' commit limit (RAM plus pagefile) while leaving 8 GiB of it free";
	p.infeasible_reason = p.feasible
		? null
		: p.max_ctx
			? `Native host context does not fit ${bound}. Largest fitting context: ${p.max_ctx} tokens.`
			: `No native context fits ${bound}.`;
	const [onCard, vram] = layout(p.context);
	const room = commitRoom(p.context);
	const cacheByRam =
		Math.floor(Math.min(acct.experts, Math.max(0, ram - cost(p.context))) / MIB) * MIB;
	const cache =
		room === null
			? cacheByRam
			: Math.min(
					cacheByRam,
					Math.floor(Math.max(0, room - privateBytes(p.context)) / MIB) * MIB,
				);
	const [seedTps, seedTier] = seed(onCard > 0);
	const basis = `Estimate seeded from GLM-5.3-Flash at ${seedTps} tok/s (RTX 4090, ${seedTier / GIB} GiB expert tier); conservatively scaled. Prefill speed is unknown. Live /metrics calibrates decode when available.`;
	p.native = {
		// The engine's receipt measures the process, and Windows counts what it
		// holds on the card in that.
		ram_budget_mib: Math.floor((ram + onCard) / MIB),
		host_headroom_mib: 512,
		expert_cache_mib: cache / MIB,
		vram_budget_bytes: vram,
		vram_reserve_mib: 5120,
		prefetch_depth: acct.architecture === 'qwen3moe' ? 2 : acct.architecture === 'glm5-next' ? 4 : 0,
		host_budget_mib:
			commit === null
				? null
				: ram / MIB + Math.ceil((vramCommit(p.context) + 3 * GIB) / MIB),
		estimate_basis: basis,
	};
	p.gpu_layers = onCard > 0 ? -1 : 0;
	p.gpu_expert_layers = 0;
	p.cpu_expert_layers = acct.layers;
	p.n_cpu_moe = null;
	p.expert_cache_mib = 0;
	p.tiers = {
		vram_weights: onCard > 0 ? acct.always : 0,
		vram_expert_cache: vram,
		vram_kv: onCard > 0 ? contextCacheBytes(p.context) : 0,
		vram_overhead:
			(vram > 0 ? 450 * MIB : 0) + (onCard > 0 ? margined(p.context.compute_bytes) : 0),
		ram: Math.min(ram, cost(p.context) + cache),
		ssd: Math.max(0, acct.experts - cache - vram),
	};
	p.tier_mix = 'ssd';
	const decode =
		seededTps(onCard > 0, vram) *
		Math.min(1, Math.max(0.05, hw.bandwidth.ssd_gbps / 6.8)) *
		Math.min(1, Math.max(0.1, hw.bandwidth.ram_gbps / 60));
	p.speed = {
		decode_tps: decode,
		raw_decode_tps: decode,
		prompt_tps: 0,
		raw_prompt_tps: 0,
		label: speedLabel(decode),
		bottleneck: 'ssd',
		calibration_samples: 0,
	};
	p.args = [
		'-c',
		String(p.ctx),
		'-ngl',
		onCard > 0 ? 'all' : '0',
		'-b',
		'512',
		'--ssd',
		'-np',
		'1',
		'--metrics',
		'--no-warmup',
		// Dense layers on the card keep every routed expert on the host.
		...(onCard > 0 ? ['--cpu-moe'] : ['--no-op-offload', '--no-kv-offload']),
		'-ctk',
		p.kv_type,
		'-ctv',
		p.kv_type_v,
		'-ub',
		'512',
	];
	for (const [key, value] of Object.entries({
		'engine.ram_budget_mib': p.native.ram_budget_mib,
		'host.headroom_mib': 512,
		'engine.expert_cache_mib': cache / MIB,
		'prefetch.depth': p.native.prefetch_depth,
		'vram.budget': vram,
		...(vram > 0 ? { 'vram.reserve_mib': 5120 } : {}),
		...(p.native.host_budget_mib ? { 'host.budget_mib': p.native.host_budget_mib } : {}),
	}))
		p.args.push('--ssd-set', `${key}=${value}`);
	// A limit that leaves nothing on the card: a CUDA build would still create
	// its context there.
	if (ov.vram_limit_mib !== undefined && vram === 0 && onCard === 0) p.args.push('--device', 'none');
	p.warnings = [
		basis,
		onCard > 0
			? 'Dense layers and the context cache run in graphics memory; routed experts are prefilled on the CPU. Expert layouts (including F32) are validated by the engine; MTP/draft is not enabled.'
			: 'Dense layers, prefill and KV stay on the CPU. Expert layouts (including F32) are validated by the engine; MTP/draft is not enabled.',
		...(limitDropsTier
			? [
					`The graphics memory limit is below the ${((minimumTier + 450 * MIB) / 1e9).toFixed(1)} GB this model's expert tier needs at minimum, so no experts are kept in graphics memory.`,
				]
			: []),
		...(cacheByRam - cache >= GIB
			? [
					`Windows' commit limit (RAM plus pagefile) holds the system memory expert cache to ${(cache / 1e9).toFixed(1)} GB. A larger pagefile would let it use ${((cacheByRam - cache) / 1e9).toFixed(1)} GB more of the free RAM.`,
				]
			: []),
		...(p.infeasible_reason ? [p.infeasible_reason] : []),
	];
	return p;
}

function mockPlan(modelId: string, ov: PlanOverrides = {}): LoadPlan {
	const acct = mockAccountingFor(modelId);
	const model = MODELS.find((m) => m.id === modelId)!;
	const hw = mockHardware();
	const budget = mockTextBudget();
	const ctxTrain = model.context_train ?? 0;
	const cap = model.total_size < 8 * GIB ? 131072 : 32768;
	const wanted = ov.ctx ?? Math.min(ctxTrain || 4096, cap);
	const ctx = Math.max(
		256,
		ov.ctx !== undefined ? wanted : ctxTrain > 0 ? Math.min(wanted, ctxTrain) : wanted,
	);

	let plan = mockPlace(acct, hw, budget, ctx, ov.kv_type ?? 'f16', ov);
	if (!ov.kv_type && plan.tier_mix !== 'vram') {
		// q8_0 KV only when it buys a better tier.
		const q8 = mockPlace(acct, hw, budget, ctx, 'q8_0', ov);
		if (TIER_RANK[q8.tier_mix] < TIER_RANK[plan.tier_mix] || (q8.feasible && !plan.feasible))
			plan = q8;
	}
	plan.ctx_train = ctxTrain;
	if (
		ov.engine === 'ssd_stream' ||
		(acct.experts > 0 && (plan.tiers.ssd > 0 || !plan.feasible) && ov.engine !== 'llama_server')
	) {
		return mockNativePlan(acct, hw, plan, ov);
	}
	// A plan llama-server already refuses keeps its own reason below (a memory
	// limit the context cannot fit is not fixed by changing engines).
	if (acct.experts > 0 && plan.tiers.ssd > 0 && plan.feasible && ov.engine === 'llama_server') {
		plan.feasible = false;
		plan.infeasible_reason =
			'This llama-server plan overcommits RAM. Choose auto or the native streaming engine.';
		plan.warnings.push(plan.infeasible_reason);
		return plan;
	}
	if (!plan.feasible) {
		// Largest context that fits with the same cache types (binary search
		// over multiples of 256 strictly below the request; `hi` is exclusive).
		const fits = (c: number) => mockPlace(acct, hw, budget, c, plan.kv_type, ov).feasible;
		let lo = 0;
		let hi = Math.ceil(plan.ctx / 256);
		if (hi > 0 && fits(256)) {
			lo = 1;
			while (hi - lo > 1) {
				const mid = lo + Math.floor((hi - lo) / 2);
				if (fits(mid * 256)) lo = mid;
				else hi = mid;
			}
		}
		plan.max_ctx = lo > 0 ? lo * 256 : null;
		const c = plan.context;
		const gb = (b: number) => `${(b / 1e9).toFixed(1)} GB`;
		// A system memory limit smaller than the context window is named.
		const short = mockRamLimitShort(hw, ov, plan);
		const within = short === null ? '' : ` within the ${gb(short)} system memory limit`;
		const ram = mockRamAvail(hw, ov);
		const reason =
			plan.max_ctx !== null
				? `A ${plan.ctx} token context doesn't fit${within}, even with no expert cache: its context memory and compute buffers take ${gb(contextCacheBytes(c) + c.compute_bytes)}. The largest context that fits is ${plan.max_ctx} tokens.`
				: short !== null
					? `No context length fits${within}: the smallest context window needs more system memory than that. Raise the limit.`
					: (ov.allow_ssd ?? true) === false && plan.tiers.ssd > 0
						? ram.limited
							? `The weights don't fit in graphics memory and the ${gb(ram.avail)} system memory limit, and SSD streaming is off.`
							: "The weights don't fit in graphics and system memory, and SSD streaming is off."
						: "The model's weights don't fit in the available memory at any context length.";
		plan.warnings.push(reason);
		plan.infeasible_reason = reason;
	}
	if (ov.ctx !== undefined && ov.ctx > ctx) {
		plan.warnings.push(`Context capped at ${ctx} tokens (the model supports ${ctxTrain}).`);
	}
	if (budget.reclaimable.length > 0 && (plan.tier_mix !== 'vram' || !plan.feasible)) {
		const freed = budget.reclaimable.reduce((s, r) => s + r.bytes, 0);
		const alt = mockPlace(
			acct,
			hw,
			{ ...budget, free: budget.free + freed, reclaimable: [] },
			plan.ctx,
			plan.kv_type,
			ov,
		);
		if (TIER_RANK[alt.tier_mix] < TIER_RANK[plan.tier_mix] || (alt.feasible && !plan.feasible)) {
			const t = alt.tiers;
			plan.evict_alternative = {
				evicts: budget.reclaimable,
				vram_bytes: t.vram_weights + t.vram_expert_cache + t.vram_kv + t.vram_overhead,
				tier_mix: alt.tier_mix,
				decode_tps: alt.speed.decode_tps,
				label: alt.speed.label,
			};
			const what =
				alt.tier_mix === 'vram'
					? 'run entirely in graphics memory'
					: alt.tier_mix === 'ram'
						? 'use less system memory'
						: 'faster';
			plan.warnings.push(
				`Unloading ${budget.reclaimable.map((r) => r.label).join(', ')} first would make this ${what} (${alt.speed.decode_tps.toFixed(0)} tok/s).`,
			);
		}
	}
	return plan;
}

/** Plan currently backing the loaded text model (drives runtime stats). */
let textPlan: { modelId: string; plan: LoadPlan } | null = null;

const IDLE_STATS: RuntimeStats = {
	model_id: null,
	decode_tps: null,
	prompt_tps: null,
	decode_label: null,
	busy: false,
	tokens_generated: 0,
	vram_mb: null,
	ram_bytes: null,
	estimate: null,
	sampled_at: 0,
};
let runtimeTimer: ReturnType<typeof setInterval> | null = null;
let tokensGenerated = 0;

function runtimeStatsNow(): RuntimeStats {
	if (engineState.phase !== 'ready' || typeof engineState.model_id !== 'string') {
		return { ...IDLE_STATS, sampled_at: now() };
	}
	const modelId = engineState.model_id;
	if (!textPlan || textPlan.modelId !== modelId) {
		try {
			textPlan = { modelId, plan: mockPlan(modelId) };
		} catch {
			textPlan = null;
		}
	}
	const p = textPlan?.plan;
	const t = p?.tiers;
	// Actuals land within a few percent of the estimate.
	const jitter = 0.92 + Math.random() * 0.1;
	const decode = p ? p.speed.decode_tps * jitter : null;
	tokensGenerated += decode ? Math.round(decode * 0.2) : 0;
	return {
		model_id: modelId,
		decode_tps: decode,
		prompt_tps: p ? p.speed.prompt_tps * jitter : null,
		decode_label: decode !== null ? speedLabel(decode) : null,
		busy: false,
		tokens_generated: tokensGenerated,
		vram_mb: t
			? Math.round((t.vram_weights + t.vram_expert_cache + t.vram_kv + t.vram_overhead) / MIB)
			: null,
		ram_bytes: t ? t.ram : null,
		estimate:
			p && t
				? {
						decode_tps: p.speed.decode_tps,
						prompt_tps: p.speed.prompt_tps,
						label: p.speed.label,
						tier_mix: p.tier_mix,
						vram_bytes: t.vram_weights + t.vram_expert_cache + t.vram_kv + t.vram_overhead,
						ram_bytes: t.ram,
						ssd_bytes: t.ssd,
					}
				: null,
		sampled_at: now(),
	};
}

function startRuntimeStats() {
	if (runtimeTimer) return;
	runtimeTimer = setInterval(() => {
		if (!listeners.get('subscribe_runtime_stats')?.size) return;
		emit('subscribe_runtime_stats', runtimeStatsNow());
	}, 1000);
}

/** Text-slot load through the planner, mirroring `load_text_planned`. */
function mockPlannedTextLoad(
	modelId: string,
	force: boolean,
	plan?: LoadPlan,
	overrides?: PlanOverrides,
):
	| {
			outcome: 'needs_eviction';
			required_mb: number;
			free_mb: number;
			candidates: ResidentInfo[];
	  }
	| { outcome: 'ready'; plan: LoadPlan } {
	const fresh = mockPlan(modelId, overrides ?? plan?.overrides ?? {});
	// A supplied plan is a preview, not a memory reservation.
	const chosen = fresh;
	if (chosen.evict_alternative && !force) {
		const ids = new Set(chosen.evict_alternative.evicts.map((r) => r.id));
		return {
			outcome: 'needs_eviction',
			required_mb: Math.round(chosen.evict_alternative.vram_bytes / MIB),
			free_mb: Math.round(mockTextBudget().free / MIB),
			candidates: residents.filter((r) => ids.has(r.id)),
		};
	}
	// force skips only the eviction offer, never an infeasible plan.
	if (!chosen.feasible) {
		throw new Error(chosen.warnings.at(-1) ?? "This model doesn't fit with these settings.");
	}
	if (chosen.engine_kind === 'llama_server' && !mockEngineSetup().llama) {
		throw new Error('Engine not installed — run engine_install first');
	}
	return { outcome: 'ready', plan: chosen };
}

/** Apply a successful text load: ledger resident, engine state, live plan. */
function commitTextLoad(
	modelId: string,
	label: string,
	ctx: number,
	estMb: number,
	plan: LoadPlan | null,
) {
	residents = residents.filter((r) => r.id !== 'text');
	residents.push({
		id: 'text',
		kind: 'text_engine',
		label,
		model_id: modelId,
		est_mb: estMb,
		actual_mb: null,
	});
	textPlan = plan ? { modelId, plan } : null;
	tokensGenerated = 0;
	setEngineState({ phase: 'ready', model_id: modelId, port: 8732, ctx, launcher: plan?.engine_kind ?? 'llama_server' });
	emitLedger();
}

function planVramMb(p: LoadPlan): number {
	const t = p.tiers;
	return Math.round((t.vram_weights + t.vram_expert_cache + t.vram_kv + t.vram_overhead) / MIB);
}

// ---------------------------------------------------------------------------
// Qwen3-TTS sidecar (qwentts.cpp tts-server) — mirrors tts::TtsStatus
// ---------------------------------------------------------------------------

const QWEN_TTS_VOICES = [
	'serena', 'vivian', 'uncle_fu', 'ryan', 'aiden', 'ono_anna', 'sohee', 'eric', 'dylan',
];

let ttsState: TtsStatus = { phase: 'idle' };

function setTtsState(s: TtsStatus) {
	ttsState = s;
	emit('subscribe_tts_status', s);
}

/** Simulate a qwentts pin-and-fetch install: installing ticks, then idle. */
async function simulateTtsInstall(): Promise<void> {
	const total = 480_000_000;
	const stages: [string, number][] = [
		['starting', 2],
		['downloading (1/2)', 30],
		['downloading (1/2)', 60],
		['extracting (1/2)', 72],
		['downloading (2/2)', 92],
		['installing (2/2)', 100],
	];
	for (const [stage, pct] of stages) {
		setTtsState({
			phase: 'installing',
			stage,
			pct,
			bytes: Math.round((total * pct) / 100),
			total,
		});
		await sleep(350);
	}
	ttsInstalled = true;
	setTtsState({ phase: 'idle' });
}

/** Whether the mock qwentts runtime is provisioned. Starts true so the
 * Studio Speech demo works out of the box; the install action flips it via
 * a fresh "not installed → installing → idle" walk when reset to false. */
let ttsInstalled = true;

const TTS_LOG_LINES = [
	'tts-server: build qwentts.cpp (mock) for x64 + CUDA',
	'loading talker model: qwen3-tts-1.7b-customvoice-q8_0.gguf',
	'loading codec: qwen-tokenizer-12hz-Q8_0.gguf',
	'server listening on http://127.0.0.1:18181',
	'GET /v1/voices -> 9 speakers',
];

/** Bring the mock sidecar to Ready with the given talker, ledger included. */
function mockLoadTts(modelId: string) {
	ttsState = {
		phase: 'ready',
		model_id: modelId,
		port: 18181,
		voices: QWEN_TTS_VOICES,
	};
	const model = MODELS.find((m) => m.id === modelId);
	residents = residents.filter((r) => r.id !== 'tts');
	residents.push({
		id: 'tts',
		kind: 'tts_engine',
		label: model?.display_name ?? modelId,
		model_id: modelId || null,
		est_mb: 3100,
		actual_mb: null,
	});
	emitLedger();
}

// Generation job history — one finished image job so the list isn't empty.
let generationJobs: GenerationJob[] = [
	{
		id: 'mock-job-history-1', task: 'text_to_image', status: 'done',
		params: { model: 'C:\\mock\\models\\sdxl-base', prompt: 'a lighthouse at dawn, oil painting', steps: 30, width: 1024, height: 1024,
		},
		outputs: ['C:\\mock\\OpenHarness\\assets\\img-20260610-0001.png'],
		meta: { vram_mb: 9870, seconds: 14.2 },
		error: null, created_at: now() - 7200, finished_at: now() - 7185,
	},
];
let jobCounter = 0;

/**
 * Test hook: append job rows as if an earlier session wrote them (e.g. an old
 * job still running under a long history). Not used by the UI.
 */
export function mockSeedGenerationJobs(rows: GenerationJob[]): void {
	generationJobs.push(...rows.map((r) => ({ ...r })));
}

const TASK_VRAM_MB: Record<string, number> = {
	text_to_image: 8200, text_to_video: 13500, text_to_speech: 1200, vibevoice: 6900, transcribe: 3000,
};

// ---------------------------------------------------------------------------
// Fixtures — asset library (M10)
// ---------------------------------------------------------------------------

function mockKindForPath(path: string): AssetKind {
	const ext = (path.split('.').pop() ?? '').toLowerCase();
	if (['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif'].includes(ext)) return 'image';
	if (['mp4', 'webm', 'mkv', 'mov', 'avi'].includes(ext)) return 'video';
	if (['wav', 'mp3', 'flac', 'ogg', 'm4a', 'opus'].includes(ext)) return 'audio';
	return 'text';
}

let assetCounter = 0;
let assets: Asset[] = [
	{
		id: 'mock-asset-1', job_id: 'mock-job-history-1', kind: 'image',
		path: 'C:\\mock\\OpenHarness\\assets\\images\\image-lighthouse-00.png',
		task: 'text_to_image', prompt: 'a lighthouse at dawn, oil painting',
		// sdxl-base is not installed in the mock catalog: remix keeps the current model.
		model: 'sdxl-base', params: { model: 'C:\\mock\\models\\sdxl-base', arch: 'sdxl', steps: 30, width: 1024, height: 1024,
		},
		meta: { seed: 7741, vram_mb: 9870 }, size: 1_830_000, created_at: now() - 7185,
	},
	{
		id: 'mock-asset-2', job_id: null, kind: 'image',
		path: 'C:\\mock\\OpenHarness\\assets\\images\\image-citynight-00.png',
		task: 'text_to_image', prompt: 'cyberpunk city street at night, rain, neon reflections',
		model: 'sdxl-turbo', params: { model: 'C:\\mock\\models\\sdxl-turbo', steps: 4, width: 1024, height: 1024,
		},
		meta: { seed: 91, vram_mb: 8120 }, size: 2_140_000, created_at: now() - 86_400,
	},
	{
		id: 'mock-asset-3', job_id: null, kind: 'audio',
		path: 'C:\\mock\\OpenHarness\\assets\\audio\\speech-welcome.wav',
		task: 'text_to_speech', prompt: 'Welcome to OpenHarness — everything runs on your own machine.',
		model: 'Kokoro-82M (af_heart)', params: { voice: 'af_heart', speed: 1 },
		meta: { voice: 'af_heart', duration: 4.6, sample_rate: 24000 }, size: 442_000, created_at: now() - 172_800,
	},
	{
		id: 'mock-asset-4', job_id: null, kind: 'text',
		path: 'C:\\mock\\OpenHarness\\assets\\transcripts\\transcript-standup.txt',
		task: 'transcribe', prompt: null,
		model: 'whisper-large-v3', params: { model: 'large-v3' },
		meta: { language: 'en', duration: 312.4, segment_count: 41 }, size: 18_400, created_at: now() - 259_200,
	},
	{
		id: 'mock-asset-5', job_id: null, kind: 'video',
		path: 'C:\\mock\\OpenHarness\\assets\\video\\video-drone.mp4',
		task: 'text_to_video', prompt: 'aerial drone shot over a misty pine forest at sunrise',
		model: 'LTX-Video', params: { num_frames: 97, fps: 24 },
		meta: { seed: 512, vram_mb: 13400 }, size: 6_280_000, created_at: now() - 345_600,
	},
];

/** Register a finished mock job's outputs as library assets (mirrors promote_job_outputs). */
function registerMockAssets(job: GenerationJob) {
	const params = (job.params ?? {}) as Record<string, unknown>;
	const prompt = (params['prompt'] ?? params['text']) as string | undefined;
	const model = String(params['model'] ?? job.task).split(/[\\/]/).pop() ?? job.task;
	for (const out of job.outputs) {
		assetCounter += 1;
		assets.unshift({
			id: `mock-asset-new-${assetCounter}`, job_id: job.id, kind: mockKindForPath(out),
			path: out, task: job.task, prompt: prompt ?? null, model,
			params: job.params, meta: job.meta, size: 1_500_000, created_at: now(),
		});
	}
}

/** Library subdir + extension(s) per task, mirroring assets.rs promotion. */
function mockOutputsFor(job: GenerationJob): string[] {
	const base = 'C:\\mock\\OpenHarness\\assets';
	switch (job.task) {
		case 'text_to_speech':
		case 'vibevoice':
			return [`${base}\\audio\\speech-${job.id}.wav`];
		case 'text_to_video':
			return [`${base}\\video\\video-${job.id}.mp4`];
		case 'transcribe':
			return [
				`${base}\\transcripts\\transcript-${job.id}.json`,
				`${base}\\transcripts\\transcript-${job.id}.txt`,
			];
		default:
			return [`${base}\\images\\image-${job.id}-00.png`];
	}
}

/** Scripted job: progress ticks then done, mirroring the worker's event shapes. */
async function simulateJob(job: GenerationJob) {
	const ev = (e: GenerationEvent) => emit('subscribe_generation_progress', e);
	const params = (job.params ?? {}) as Record<string, unknown>;
	const isQwenClone =
		job.task === 'text_to_speech' &&
		params['engine'] === 'qwen3-tts' &&
		params['reference_audio'] !== undefined;
	const isQwenTts =
		job.task === 'text_to_speech' && params['engine'] === 'qwen3-tts' && !isQwenClone;
	const residentId = isQwenClone
		? 'pipeline:qwen_tts_clone'
		: isQwenTts
			? 'tts'
			: `pipeline:${job.task}`;
	const stages = isQwenClone
		? ['loading model', 'encoding reference voice', 'generating', 'generating', 'saving']
		: ['loading model', 'generating', 'generating', 'generating', 'saving'];
	for (let i = 0; i < stages.length; i++) {
		await sleep(450);
		// Worker exit already reported this job's terminal error.
		if (job.status === 'error') return;
		if (job.status === 'cancelled') {
			ev({ event: 'job_error', job_id: job.id, error: 'cancelled' });
			removeResident(residentId);
			return;
		}
		ev({
			event: 'job_progress', job_id: job.id, pct: ((i + 1) / stages.length) * 100,
			stage: stages[i], step: i + 1, total_steps: stages.length,
		});
	}
	job.status = 'done';
	job.outputs = mockOutputsFor(job);
	job.meta =
		job.task === 'transcribe'
			? {
					language: 'en',
					language_probability: 0.97,
					duration: 42.5,
					segment_count: 3,
				}
			: isQwenClone
				? {
						engine: 'qwen3-tts',
						mode: 'clone',
						sample_rate: 24000,
						duration: 4.4,
					}
				: isQwenTts
					? { engine: 'qwen3-tts', voice: params['voice'] ?? 'serena', sample_rate: 24000, duration: 4.4,
						}
					: job.task === 'vibevoice'
						? {
								engine: 'vibevoice', sample_rate: 24000, duration: 5.2,
								cfg_scale: params['cfg_scale'] ?? 1.3, num_speakers: 1,
								vram_mb: TASK_VRAM_MB['vibevoice'],
							}
						: job.task === 'text_to_speech'
							? { voice: params['voice'] ?? 'af_heart', speed: 1, sample_rate: 24000, duration: 3.1,
								}
							: {
									seed: 1234,
									vram_mb: TASK_VRAM_MB[job.task] ?? 2000,
									seconds: 2.3,
								};
	job.finished_at = now();
	registerMockAssets(job);
	if (isQwenClone) {
		// The transient clone reservation is released once the CLI exits.
		removeResident(residentId);
	} else {
		// Swap the est for the NVML actual, like the real ledger does on job_done.
		const res = residents.find((r) => r.id === residentId);
		if (res) res.actual_mb = isQwenTts ? 3060 : (TASK_VRAM_MB[job.task] ?? 2000);
		emitLedger();
	}
	// The real Qwen paths send task: None so the orchestrator's pipeline-resident
	// bookkeeping ignores them; the job-sync + asset promotion key off job_id alone.
	ev({
		event: 'job_done', job_id: job.id, outputs: job.outputs, meta: job.meta,
		task: isQwenTts || isQwenClone ? null : job.task,
	});
}

/** Qwen3-TTS runs in its own sidecar or CLI; every other job is the worker's. */
function mockIsWorkerJob(job: GenerationJob): boolean {
	const p = (job.params ?? {}) as Record<string, unknown>;
	return !(job.task === 'text_to_speech' && p['engine'] === 'qwen3-tts');
}

/** Native chat turns in flight (mirrors the chat activity gate). */
let mockChatTurns = 0;
const MOCK_SPEECH_RESIDENTS = ['tts', 'pipeline:text_to_speech', 'pipeline:qwen_tts_clone', 'pipeline:vibevoice',
];

/**
 * Mirrors orchestrator::in_use_reason: a user unload refuses while a chat
 * turn holds the text engine or a running job may be using the resident.
 */
function mockAssertIdle(residentId: string): void {
	if (residentId === 'text' && mockChatTurns > 0) {
		throw new Error('A chat reply is streaming from this model. Stop it in Chat before unloading.');
	}
	const busy = generationJobs.some(
		(j) =>
			j.status === 'running' &&
			(j.task === 'text_to_speech' || j.task === 'vibevoice'
				? MOCK_SPEECH_RESIDENTS.includes(residentId)
				: residentId === `pipeline:${j.task}`),
	);
	if (busy) {
		throw new Error('A generation job is using this model. Let it finish or cancel it before unloading.',
		);
	}
}

/**
 * Leases whose job rows aren't written yet (orchestrator::leases): served by
 * active_generation_jobs so a reloaded page adopts them.
 */
const mockUnwrittenLeases = new Map<string, string>();

/** Test hook: a submit holds its lease but hasn't inserted its row yet. */
export function mockGrantUnwrittenLease(jobId: string, task: string): void {
	mockUnwrittenLeases.set(jobId, task);
}

/**
 * Test hook mirroring OrchestratorHandle::abort_lease: the submit failed
 * after its lease was granted, so a held lease ends with a job_error.
 */
export function mockAbortLease(jobId: string, error: string): void {
	if (!mockUnwrittenLeases.delete(jobId)) return;
	emit('subscribe_generation_progress', {
		event: 'job_error', job_id: jobId, error,
	} satisfies GenerationEvent);
}

/** A Qwen3-TTS sidecar synthesis is running (or leased) on the "tts" resident. */
function mockTtsLeased(): boolean {
	return generationJobs.some((j) => {
		const p = (j.params ?? {}) as Record<string, unknown>;
		return j.status === 'running' && !mockIsWorkerJob(j) && p['reference_audio'] === undefined;
	});
}

function submitMockJob(task: string, params: unknown, force: boolean): unknown {
	const p = (params ?? {}) as Record<string, unknown>;
	const isQwenClone =
		task === 'text_to_speech' &&
		p['engine'] === 'qwen3-tts' &&
		p['reference_audio'] !== undefined;
	const isQwenTts = task === 'text_to_speech' && p['engine'] === 'qwen3-tts' && !isQwenClone;
	// Qwen3-TTS runs in its own sidecar (or the one-shot CLI for clones), not
	// the Python worker.
	if (!isQwenTts && !isQwenClone && workerState.phase !== 'ready') {
		throw new Error('worker is not running — start it from Settings → Engines');
	}
	// No model auto-download: transcribe + worker-engine speech jobs require an
	// installed model, mirroring the worker pipelines raising on a missing one.
	const modelStr = String(p['model'] ?? '').trim();
	if (task === 'transcribe' && !modelStr) {
		throw new Error('No transcription model selected — install a Whisper model first.');
	}
	if (task === 'text_to_speech' && !isQwenTts && !modelStr) {
		throw new Error('No speech model selected — install a Kokoro model first.');
	}
	const residentId = isQwenClone
		? 'pipeline:qwen_tts_clone'
		: isQwenTts
			? 'tts'
			: `pipeline:${task}`;
	const est = isQwenClone ? 4900 : isQwenTts ? 3100 : (TASK_VRAM_MB[task] ?? 2000);
	const snap = ledgerSnapshot();
	const already = residents.some((r) => r.id === residentId);
	if (!force && !already && est > snap.free_mb && residents.length > 0) {
		return {
			outcome: 'needs_eviction', required_mb: est, free_mb: snap.free_mb,
			candidates: snap.residents,
		};
	}
	if (isQwenClone) {
		// Transient reservation for the one-shot CLI run, released on job end.
		const catalog = MODELS.find((m) => m.id === p['model']);
		residents.push({
			id: residentId,
			kind: 'pipeline',
			label: `${catalog?.display_name ?? 'Qwen3-TTS talker'} (voice clone)`,
			model_id: catalog?.id ?? null,
			est_mb: est,
			actual_mb: null,
		});
		emitLedger();
	} else if (isQwenTts) {
		mockLoadTts(String(p['model'] ?? ''));
	} else if (!already) {
		// Correlate the resident back to its catalog row by model path, like
		// generation::submit does (M16 W1).
		const catalog = MODELS.find(
			(m) => m.local_path === p['model'] || m.primary_file_path === p['model'],
		);
		residents.push({
			id: residentId,
			kind: 'pipeline',
			label: catalog?.display_name ?? String(p['model'] ?? task),
			model_id: catalog?.id ?? null,
			est_mb: est,
			actual_mb: null,
		});
		emitLedger();
	}
	jobCounter += 1;
	const job: GenerationJob = {
		id: `mock-job-${jobCounter}`, task, status: 'running', params,
		outputs: [], meta: {}, error: null, created_at: now(), finished_at: null,
	};
	generationJobs.push(job);
	void simulateJob(job);
	return { outcome: 'submitted', job_id: job.id };
}

// ---------------------------------------------------------------------------
// Chat simulator (M5 + M11): streams a canned reply through the chat channel
// ---------------------------------------------------------------------------

interface MockConversation {
	summary: ConversationSummaryDto;
	messages: MessageDto[];
}

const mockConvs = new Map<string, MockConversation>();
let convCounter = 0;

async function simulateChatReply(convId: string, content: string) {
	if (!mockConvs.has(convId)) return;
	mockChatTurns += 1;
	try {
		await streamChatReply(convId, content);
	} finally {
		mockChatTurns -= 1;
	}
}

async function streamChatReply(convId: string, content: string) {
	const conv = mockConvs.get(convId);
	if (!conv) return;
	const chat = (e: ChatEvent) => emit('subscribe_chat_events', e);

	const userSeq = conv.messages.length;
	conv.messages.push({ seq: userSeq, role: 'user', content, ts: now() });
	conv.summary.message_count = conv.messages.length;
	conv.summary.updated_at = now();
	if (!conv.summary.title) {
		conv.summary.title = content.length > 48 ? content.slice(0, 48) + '…' : content;
	}
	chat({ type: 'user_message_stored', conv_id: convId, seq: userSeq });

	const seq = conv.messages.length;
	chat({ type: 'assistant_start', conv_id: convId, seq });

	// Reasoning-model variant: a prompt mentioning "think" streams a thinking
	// phase first (llama-server's `delta.reasoning_content` → ChatEvent
	// `thinking`), exercising the collapsible Thinking block in the UI.
	if (/think/i.test(content)) {
		const thinking =
			`The user asked: "${content.trim()}". Let me work through this step by step. ` +
			`First, I should consider what they actually need. Second, the mock harness ` +
			`streams this reasoning through the thinking side-channel, exactly like a ` +
			`Qwen3 thinking GGUF behind llama-server. Okay, I have a plan — answering now.`;
		for (const word of thinking.split(/(?<=\s)/)) {
			await sleep(22);
			chat({ type: 'thinking', conv_id: convId, seq, delta: word });
		}
		await sleep(200);
	}

	// Markdown-rich variant: a prompt mentioning "markdown" or "code" streams
	// headings/lists/fences/tables/links so StreamingMarkdown can be exercised
	// in the browser harness (the fence must format from its first token).
	const reply = /markdown|code/i.test(content)
		? `Here is a **markdown-rich** streamed reply.\n\n## Retry helper\n\n` +
			`- exponential backoff\n- jitter via \`Math.random()\`\n\n` +
			'```ts\nexport function backoff(attempt: number): number {\n' +
			'\treturn 100 * 2 ** (attempt - 1);\n}\n```\n\n' +
			`| attempt | delay |\n| --- | --- |\n| 1 | 100ms |\n| 2 | 200ms |\n\n` +
			`See [the docs](https://example.com/docs) for *more details*.`
			: `Here is a mock streamed reply to “${content.trim()}”. The real backend runs ` +
				`llama-server with your loaded model; this browser harness just exercises the UI.`;
	let acc = '';
	for (const word of reply.split(/(?<=\s)/)) {
		await sleep(mockTokenDelayMs());
		acc += word;
		chat({ type: 'token', conv_id: convId, seq, delta: word });
	}

	conv.messages.push({
		seq, role: 'assistant', content: acc, ts: now(),
		prompt_tokens: 64, completion_tokens: acc.split(/\s+/).length, finish_reason: 'stop',
	});
	conv.summary.message_count = conv.messages.length;
	conv.summary.updated_at = now();
	chat({
		type: 'assistant_done', conv_id: convId, seq,
		prompt_tokens: 64, completion_tokens: acc.split(/\s+/).length, finish_reason: 'stop',
	});
}

/** Prune a mock conversation from `fromSeq` and broadcast the truncated event. */
function mockTruncateFrom(convId: string, fromSeq: number) {
	const conv = mockConvs.get(convId);
	if (!conv) return;
	conv.messages = conv.messages.filter((m) => m.seq < fromSeq);
	conv.summary.message_count = conv.messages.length;
	conv.summary.updated_at = now();
	emit('subscribe_chat_events', {
		type: 'truncated', conv_id: convId, from_seq: fromSeq,
	} satisfies ChatEvent);
}

/** Regenerate: truncate from `seq`, then stream a fresh canned assistant reply. */
async function simulateRegenerate(convId: string, seq: number) {
	if (!mockConvs.has(convId)) return;
	mockChatTurns += 1;
	try {
		await streamRegenerate(convId, seq);
	} finally {
		mockChatTurns -= 1;
	}
}

async function streamRegenerate(convId: string, seq: number) {
	const conv = mockConvs.get(convId);
	if (!conv) return;
	mockTruncateFrom(convId, seq);
	const chat = (e: ChatEvent) => emit('subscribe_chat_events', e);
	const lastUser = [...conv.messages].reverse().find((m) => m.role === 'user');

	const newSeq = conv.messages.length === 0 ? 0 : conv.messages[conv.messages.length - 1].seq + 1;
	chat({ type: 'assistant_start', conv_id: convId, seq: newSeq });
	const reply =
		`Here is a regenerated mock reply${lastUser ? ` to “${lastUser.content.trim()}”` : ''}. ` +
		`A fresh sample from the same history — the real backend re-streams from llama-server.`;
	let acc = '';
	for (const word of reply.split(/(?<=\s)/)) {
		await sleep(mockTokenDelayMs());
		acc += word;
		chat({ type: 'token', conv_id: convId, seq: newSeq, delta: word });
	}
	conv.messages.push({
		seq: newSeq, role: 'assistant', content: acc, ts: now(),
		prompt_tokens: 64, completion_tokens: acc.split(/\s+/).length, finish_reason: 'stop',
	});
	conv.summary.message_count = conv.messages.length;
	conv.summary.updated_at = now();
	chat({
		type: 'assistant_done', conv_id: convId, seq: newSeq,
		prompt_tokens: 64, completion_tokens: acc.split(/\s+/).length, finish_reason: 'stop',
	});
}

// ---------------------------------------------------------------------------
// Fixtures — HF store (M9 §9.5): canned search results + details per task
// ---------------------------------------------------------------------------

function card(
	id: string,
	pipeline_tag: string | undefined,
	library_name: string | undefined,
	tags: string[],
	downloads: number,
	likes: number,
): HfModelCard {
	return {
		id, author: id.split('/')[0], pipeline_tag, library_name, tags,
		downloads, likes, gated: false, private: false,
		last_modified: '2026-05-20T12:00:00.000Z', created_at: '2025-11-02T08:00:00.000Z',
	};
}

const f = (path: string, size: number): HfFile => ({
	path,
	size,
	is_lfs: size > 1_000_000,
});

/** Per-repo canned siblings + layout (mirrors models/layout.rs classify()). */
const HF_REPOS: Array<{ card: HfModelCard; siblings: HfFile[]; layout: RepoLayout; readme: string;
}> = [
	{
		card: card('unsloth/GLM-5.3-Flash-GGUF', undefined, 'gguf', ['gguf', 'chat'], 12000, 100),
		siblings: [
			...[9_000_000, 49_000_000_000, 50_000_000_000, 9_000_000_000].map((size, i) =>
				f(`UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-${String(i + 1).padStart(5, '0')}-of-00004.gguf`, size)),
			f('mmproj-F16.gguf', 900_000_000),
			f('Shard_Rewrite/model.gguf_file', 500_000_000),
			f('README.md', 8000),
		],
		layout: { kind: 'gguf', download_files: [] },
		readme: '# GLM-5.3-Flash GGUF\n\nSplit GGUF Store fixture. Download all four parts as one model.',
	},
	{
		card: card('Qwen/Qwen2.5-7B-Instruct-GGUF', 'text-generation', 'gguf', ['gguf', 'chat', 'qwen2'], 2_412_000, 1843,
		),
		siblings: [
			f('qwen2.5-7b-instruct-q4_k_m.gguf', 4_680_000_000),
			f('qwen2.5-7b-instruct-q5_k_m.gguf', 5_440_000_000),
			f('qwen2.5-7b-instruct-q8_0.gguf', 8_100_000_000),
			f('README.md', 12_000),
		],
		layout: { kind: 'gguf', download_files: [] },
		readme: '# Qwen2.5 7B Instruct — GGUF\n\nQuantized GGUF builds of Qwen2.5-7B-Instruct for llama.cpp.\n\n| Quant | Size |\n|---|---|\n| Q4_K_M | 4.7 GB |\n| Q5_K_M | 5.4 GB |\n| Q8_0 | 8.1 GB |',
	},
	{
		// The installed Q4_K_M (its sha256 is the catalog id) sits beside a
		// second file with the same quant label that is NOT installed.
		card: card('Qwen/Qwen2.5-Coder-7B-Instruct-GGUF', 'text-generation', 'gguf', ['gguf', 'code', 'qwen2'], 1_106_000, 912,
		),
		siblings: [
			{ path: 'qwen2.5-coder-7b-instruct-q4_k_m.gguf', size: 4_680_000_000, sha256: MODEL_ID, is_lfs: true,
			},
			{ path: 'qwen2.5-coder-7b-instruct-q4_k_m-imat.gguf', size: 4_710_000_000, sha256: 'mock-sha-imat', is_lfs: true,
			},
			f('qwen2.5-coder-7b-instruct-q5_k_m.gguf', 5_440_000_000),
			f('qwen2.5-coder-7b-instruct-q8_0.gguf', 8_100_000_000),
			f('README.md', 9_000),
		],
		layout: { kind: 'gguf', download_files: [] },
		readme: '# Qwen2.5 Coder 7B Instruct — GGUF\n\nCode-tuned Qwen2.5 7B, quantized for llama.cpp.',
	},
	{
		card: card('meta-llama/Llama-3.1-8B-Instruct', 'text-generation', 'transformers', ['safetensors', 'llama'], 5_204_000, 4112,
		),
		siblings: [
			f('model-00001-of-00004.safetensors', 4_980_000_000),
			f('model-00002-of-00004.safetensors', 4_990_000_000),
			f('model-00003-of-00004.safetensors', 4_920_000_000),
			f('model-00004-of-00004.safetensors', 1_170_000_000),
			f('model.safetensors.index.json', 24_000),
			f('config.json', 800),
			f('tokenizer.json', 9_000_000),
			f('README.md', 38_000),
		],
		layout: {
			kind: 'safetensors-shards',
			download_files: [
				'model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors',
				'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors',
				'model.safetensors.index.json', 'config.json', 'tokenizer.json',
			],
		},
		readme: '# Llama 3.1 8B Instruct\n\nFull-precision transformers checkpoint — needs a GGUF conversion to run locally on llama.cpp.',
	},
	{
		card: card('stabilityai/sdxl-turbo', 'text-to-image', 'diffusers', ['diffusers', 'stable-diffusion'], 1_893_000, 2967,
		),
		siblings: [
			f('model_index.json', 600),
			f('unet/diffusion_pytorch_model.safetensors', 5_140_000_000),
			f('unet/config.json', 1_800),
			f('vae/diffusion_pytorch_model.safetensors', 334_000_000),
			f('vae/config.json', 700),
			f('text_encoder/model.safetensors', 246_000_000),
			f('text_encoder/config.json', 600),
			f('text_encoder_2/model.safetensors', 2_780_000_000),
			f('tokenizer/vocab.json', 1_060_000),
			f('tokenizer/merges.txt', 525_000),
			f('scheduler/scheduler_config.json', 500),
			f('README.md', 18_000),
			f('sd_xl_turbo_1.0.safetensors', 6_940_000_000),
		],
		layout: {
			kind: 'diffusers-dir',
			download_files: [
				'model_index.json', 'unet/diffusion_pytorch_model.safetensors', 'unet/config.json',
				'vae/diffusion_pytorch_model.safetensors', 'vae/config.json',
				'text_encoder/model.safetensors', 'text_encoder/config.json',
				'text_encoder_2/model.safetensors', 'tokenizer/vocab.json', 'tokenizer/merges.txt',
				'scheduler/scheduler_config.json', 'sd_xl_turbo_1.0.safetensors',
			],
		},
		readme: '# SDXL-Turbo\n\nA fast text-to-image model. Runs in OpenHarness via the diffusers worker pipeline.',
	},
	{
		card: card('black-forest-labs/FLUX.1-schnell', 'text-to-image', 'diffusers', ['diffusers', 'flux'], 1_240_000, 3866,
		),
		siblings: [
			f('model_index.json', 700),
			f('transformer/diffusion_pytorch_model-00001-of-00003.safetensors', 9_960_000_000),
			f('transformer/diffusion_pytorch_model-00002-of-00003.safetensors', 9_950_000_000),
			f('transformer/diffusion_pytorch_model-00003-of-00003.safetensors', 3_870_000_000),
			f('vae/diffusion_pytorch_model.safetensors', 168_000_000),
			f('README.md', 16_000),
		],
		layout: {
			kind: 'diffusers-dir',
			download_files: [
				'model_index.json',
				'transformer/diffusion_pytorch_model-00001-of-00003.safetensors',
				'transformer/diffusion_pytorch_model-00002-of-00003.safetensors',
				'transformer/diffusion_pytorch_model-00003-of-00003.safetensors',
				'vae/diffusion_pytorch_model.safetensors',
			],
		},
		readme: '# FLUX.1 [schnell]\n\n12B rectified-flow transformer for fast text-to-image generation.',
	},
	{
		card: card('Lightricks/LTX-Video', 'text-to-video', 'diffusers', ['diffusers', 'video'], 312_000, 1421,
		),
		siblings: [
			f('model_index.json', 650),
			f('transformer/diffusion_pytorch_model.safetensors', 7_690_000_000),
			f('vae/diffusion_pytorch_model.safetensors', 1_240_000_000),
			f('README.md', 22_000),
		],
		layout: {
			kind: 'diffusers-dir',
			download_files: [
				'model_index.json', 'transformer/diffusion_pytorch_model.safetensors',
				'vae/diffusion_pytorch_model.safetensors',
			],
		},
		readme: '# LTX-Video\n\nReal-time-class text-to-video DiT. Runs on the text_to_video worker pipeline.',
	},
	{
		card: card('hexgrad/Kokoro-82M', 'text-to-speech', undefined, ['kokoro', 'tts'], 689_000, 3219),
		siblings: [
			f('kokoro-v1_0.pth', 327_000_000),
			f('voices/af_heart.pt', 524_000),
			f('voices/am_adam.pt', 524_000),
			f('README.md', 9_000),
		],
		layout: { kind: 'unknown', download_files: [] },
		readme: '# Kokoro 82M\n\nCompact, natural TTS. Runs on the text_to_speech worker pipeline.',
	},
	{
		card: card('Systran/faster-whisper-large-v3', 'automatic-speech-recognition', 'ctranslate2', ['ctranslate2', 'whisper'], 1_504_000, 743,
		),
		siblings: [
			f('model.bin', 3_090_000_000),
			f('config.json', 2_300),
			f('tokenizer.json', 2_400_000),
			f('vocabulary.txt', 460_000),
			f('README.md', 5_000),
		],
		layout: { kind: 'unknown', download_files: [] },
		readme: '# faster-whisper large-v3\n\nCTranslate2 conversion of Whisper large-v3 for fast local transcription.',
	},
	{
		card: card('openai/whisper-large-v3', 'automatic-speech-recognition', 'transformers', ['safetensors', 'whisper'], 3_812_000, 2954,
		),
		siblings: [
			f('model.safetensors', 3_090_000_000),
			f('config.json', 1_900),
			f('README.md', 30_000),
		],
		layout: {
			kind: 'safetensors-single',
			download_files: ['model.safetensors', 'config.json'],
		},
		readme: '# Whisper large-v3\n\nTransformers checkpoint — OpenHarness transcribes with the CTranslate2 conversion instead (see Systran/faster-whisper-large-v3).',
	},
	{
		card: card('nomic-ai/nomic-embed-text-v1.5-GGUF', 'feature-extraction', 'gguf', ['gguf', 'embeddings'], 178_000, 311,
		),
		siblings: [
			f('nomic-embed-text-v1.5.Q8_0.gguf', 146_000_000),
			f('README.md', 7_000)],
		layout: { kind: 'gguf', download_files: [] },
		readme: '# nomic-embed-text v1.5 — GGUF\n\nLong-context embedding model for llama.cpp.',
	},
	{
		card: card('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', 'text-to-video', 'diffusers', ['diffusers', 'video'], 98_000, 502,
		),
		siblings: [
			f('model_index.json', 640),
			f('transformer/diffusion_pytorch_model.safetensors', 2_840_000_000),
			f('vae/diffusion_pytorch_model.safetensors', 507_000_000),
			f('README.md', 11_000),
		],
		layout: {
			kind: 'diffusers-dir',
			download_files: [
				'model_index.json', 'transformer/diffusion_pytorch_model.safetensors',
				'vae/diffusion_pytorch_model.safetensors',
			],
		},
		readme: '# Wan2.1 T2V 1.3B\n\nLightweight text-to-video model that fits consumer GPUs.',
	},
];

/** Mirror of store::resolve — same vocabulary, simplified reasons. */
function mockResolve(q: RunnabilityQuery): Runnability {
	const lib = q.library ?? '';
	const repo = (q.repo_id ?? '').toLowerCase();
	const tags = (q.tags ?? []).map((t) => t.toLowerCase());
	const task = q.task ?? '';

	// Format hint: exact from layout (unknown falls through), else card guess.
	let hint: string | null = null;
	if (q.layout && q.layout !== 'unknown') {
		hint =
			q.layout === 'gguf' ? 'gguf'
			: q.layout === 'diffusers-dir' ? 'diffusers-dir'
			: q.layout === 'safetensors-single' ? 'safetensors-single-file'
			: 'safetensors';
	} else if (lib === 'gguf' || tags.includes('gguf')) {
		hint = 'gguf';
	} else if (lib === 'diffusers') {
		hint = 'diffusers-dir';
	} else if (lib === 'ctranslate2' || tags.includes('ctranslate2') || repo.includes('faster-whisper')) {
		hint = 'ctranslate2-dir';
	} else if (repo.includes('kokoro') || tags.includes('kokoro')) {
		hint = 'kokoro-repo';
	} else if (tags.includes('safetensors') || lib === 'transformers' || lib === 'sentence-transformers') {
		hint = 'safetensors';
	}

	if (hint === 'gguf') {
		if (['text-generation', '', 'feature-extraction', 'sentence-similarity'].includes(task)) {
			return { runnable: true, runtime: 'llama-cpp', pipeline: null, available: true, reason: 'Runs on llama.cpp',
			};
		}
		return { runnable: false, runtime: null, pipeline: null, available: false, reason: `GGUF ${task} models aren't supported by the local runtime`,
		};
	}

	// Non-GGUF Qwen3-TTS = the original safetensors release (no local runtime).
	// GGUF repos took the branch above; this keys off the repo name only.
	// Mirrors is_qwen_tts_repo() in store.rs.
	if (repo.replace(/[^a-z0-9]/g, '').match(/qwen3tts|qwentts/)) {
		return { runnable: false, runtime: null, pipeline: null, available: false, reason: QWEN_TTS_SAFETENSORS_REASON,
		};
	}

	const pipeline = ({
		'text-to-image': 'text_to_image', 'image-to-image': 'text_to_image',
		'text-to-video': 'text_to_video', 'image-to-video': 'text_to_video',
		'automatic-speech-recognition': 'transcribe', 'text-to-speech': 'text_to_speech',
	} as Record<string, string>)[task];
	if (pipeline) {
		const cap = WORKER_PIPELINES.find((p) => p.task === pipeline);
		if (cap && hint && cap.formats.includes(hint)) {
			return {
				runnable: true, runtime: 'worker', pipeline, available: cap.available,
				reason: cap.available
					? `Runs on the ${pipeline.replace(/_/g, '-')} pipeline`
					: `Runs on the ${pipeline.replace(/_/g, '-')} pipeline (worker not running)`,
			};
		}
		if (cap) {
			return { runnable: false, runtime: null, pipeline, available: false, reason: 'Model format not supported by the local pipeline',
			};
		}
	}
	return {
		runnable: false, runtime: null, pipeline: null, available: false,
		reason: task ? `No local runtime for ${task}` : 'No local runtime for this model',
	};
}

/** Mirror of store::PICKS — fits resolved against the mock GPU (16 GB). */
const CURATED_PICKS: Array<Omit<CuratedPick, 'fits'>> = [
	{ repo_id: 'Qwen/Qwen2.5-7B-Instruct-GGUF', display_name: 'Qwen2.5 7B Instruct', task: 'text-generation', format: 'gguf', recommended_file: 'qwen2.5-7b-instruct-q4_k_m.gguf', min_vram_mb: 6144, blurb: 'Best all-round 7B chat model — strong reasoning and coding.',
	},
	{ repo_id: 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', display_name: 'Llama 3.1 8B Instruct', task: 'text-generation', format: 'gguf', recommended_file: 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', min_vram_mb: 6144, blurb: 'Meta’s open 8B flagship — great general assistant.',
	},
	{ repo_id: 'Qwen/Qwen2.5-1.5B-Instruct-GGUF', display_name: 'Qwen2.5 1.5B Instruct', task: 'text-generation', format: 'gguf', recommended_file: 'qwen2.5-1.5b-instruct-q4_k_m.gguf', min_vram_mb: 2048, blurb: 'Tiny but capable — runs on almost anything.',
	},
	{ repo_id: 'stabilityai/sdxl-turbo', display_name: 'SDXL-Turbo', task: 'text-to-image', format: 'diffusers-dir', recommended_file: null, min_vram_mb: 8192, blurb: 'One-step SDXL — near-instant 1024px images.',
	},
	{ repo_id: 'stabilityai/stable-diffusion-xl-base-1.0', display_name: 'SDXL Base 1.0', task: 'text-to-image', format: 'diffusers-dir', recommended_file: null, min_vram_mb: 10240, blurb: 'The classic high-quality SDXL base model.',
	},
	{ repo_id: 'black-forest-labs/FLUX.1-schnell', display_name: 'FLUX.1 schnell', task: 'text-to-image', format: 'diffusers-dir', recommended_file: null, min_vram_mb: 20480, blurb: 'State-of-the-art open image model — needs a big GPU.',
	},
	{ repo_id: 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers', display_name: 'Wan2.1 T2V 1.3B', task: 'text-to-video', format: 'diffusers-dir', recommended_file: null, min_vram_mb: 8192, blurb: 'Lightweight text-to-video that fits consumer GPUs.',
	},
	{ repo_id: 'Lightricks/LTX-Video', display_name: 'LTX-Video', task: 'text-to-video', format: 'diffusers-dir', recommended_file: null, min_vram_mb: 20480, blurb: 'Fast, high-quality video DiT for 24 GB cards.',
	},
	{ repo_id: 'hexgrad/Kokoro-82M', display_name: 'Kokoro 82M', task: 'text-to-speech', format: 'kokoro-repo', recommended_file: null, min_vram_mb: 1024, blurb: 'Compact, natural TTS with multiple voices.',
	},
	{ repo_id: 'Serveurperso/Qwen3-TTS-GGUF', display_name: 'Qwen3-TTS 1.7B (CustomVoice)', task: 'text-to-speech', format: 'gguf', recommended_file: 'qwen-talker-1.7b-customvoice-Q8_0.gguf', min_vram_mb: 4096, blurb: 'Expressive multi-voice TTS; the GGUF build OpenHarness runs.',
	},
	{ repo_id: 'Systran/faster-whisper-large-v3', display_name: 'faster-whisper large-v3', task: 'automatic-speech-recognition', format: 'ctranslate2-dir', recommended_file: null, min_vram_mb: 4096, blurb: 'Best local transcription quality.',
	},
	{ repo_id: 'Systran/faster-whisper-small', display_name: 'faster-whisper small', task: 'automatic-speech-recognition', format: 'ctranslate2-dir', recommended_file: null, min_vram_mb: 1024, blurb: 'Fast transcription for low-VRAM machines.',
	},
	{ repo_id: 'nomic-ai/nomic-embed-text-v1.5-GGUF', display_name: 'nomic-embed-text v1.5', task: 'feature-extraction', format: 'gguf', recommended_file: 'nomic-embed-text-v1.5.Q8_0.gguf', min_vram_mb: 1024, blurb: 'Long-context local embeddings for search.',
	},
];

/** Mirror of store::quant_fits (§9.6) against the mock GPU's free VRAM. */
function mockQuantFits(files: HfFile[], freeMb: number): QuantFit[] {
	const FIXED_OVERHEAD_MB = 1517 + 577;
	const KV_ALLOWANCE_MB = 768;

	const quantLabel = (path: string): string | null => {
		const name = path.split('/').pop() ?? path;
		const stem = name.endsWith('.gguf') ? name.slice(0, -5) : name;
		for (const token of stem.split(/[-. ]/)) {
			const upper = token.toUpperCase();
			if (/^I?Q\d[A-Z0-9_]*$/.test(upper) || ['F16', 'BF16', 'F32'].includes(upper)) {
				return upper;
			}
		}
		return null;
	};
	const splitStem = (path: string): string | null => splitGgufPart(path)?.key ?? null;

	const groupTotals = new Map<string, number>();
	for (const f of files) {
		const stem = splitStem(f.path);
		if (stem) groupTotals.set(stem, (groupTotals.get(stem) ?? 0) + f.size);
	}

	const fits: QuantFit[] = files
		.filter((f) => f.path.endsWith('.gguf'))
		.map((f) => {
			const stem = splitStem(f.path);
			const effective = stem ? (groupTotals.get(stem) ?? f.size) : f.size;
			const estimated_mb = Math.floor(effective / (1024 * 1024)) + KV_ALLOWANCE_MB + FIXED_OVERHEAD_MB;
			const verdict =
				estimated_mb > freeMb ? ('wont_fit' as const)
				: freeMb - estimated_mb < 1024 ? ('tight' as const)
				: ('fits' as const);
			return { path: f.path, label: quantLabel(f.path), size: f.size, estimated_mb, verdict, best: false, split: stem !== null,
				set_key: stem, set_size: stem ? effective : null, part_count: splitGgufPart(f.path)?.count ?? null,
			};
		});

	const largest = (want: 'fits' | 'tight') =>
		fits.filter((f) => !f.split && f.verdict === want).sort((a, b) => b.size - a.size)[0];
	const best = largest('fits') ?? largest('tight');
	if (best) best.best = true;
	return fits;
}

// ---------------------------------------------------------------------------
// Mock downloads (M3 single jobs + M9 repo groups) with simulated progress
// ---------------------------------------------------------------------------

const dlJobs = new Map<string, Job>();
const dlGroups = new Map<string, DownloadGroup>();
const closedGgufGroups = new Set<string>();
const modelFiles = new Map<string, string[]>();
const installedDownloadError = 'This model is already installed. Remove it from Models before downloading it again.';
const supersededDownloadError = 'This download was superseded. Start a new download from the Store.';
function destinationKey(path: string): string {
	return path.replaceAll('\\', '/').split('/').filter((s) => s && s !== '.')
		.map((s) => s.replace(/[. ]+$/, '').toLowerCase().toUpperCase()).join('/');
}
function repoDestination(repo: string, path: string): string {
	return destinationKey(`C:/mock/models/repos/${repo.replaceAll('/', '__')}/${path}`);
}
function ownedDestinations(): Set<string> {
	return new Set(MODELS.flatMap((m) => {
		if (modelFiles.has(m.id)) return modelFiles.get(m.id)!;
		if (!m.primary_file_path) return [];
		const part = splitGgufPart(m.primary_file_path);
		return part ? Array.from({ length: part.count }, (_, i) => m.primary_file_path!.replace(
			/-(\d+)-of-(\d+)\.gguf$/, (_all, number: string, count: string) =>
				`-${String(i + 1).padStart(number.length, '0')}-of-${count}.gguf`)) : [m.primary_file_path];
	}).map(destinationKey));
}
function refuseGgufWrite(job: Job): boolean {
	if (!job.group_id || dlGroups.get(job.group_id)?.format !== 'gguf') return false;
	const error = closedGgufGroups.has(job.group_id) ? supersededDownloadError
		: ownedDestinations().has(repoDestination(job.repo_id, job.file)) ? installedDownloadError : null;
	if (!error) return false;
	job.status = 'failed'; job.error = error;
	emitDlProgress(job);
	return true;
}
let dlCounter = 0;

function emitDlProgress(job: Job, phase?: DownloadProgress['phase']) {
	const remaining = job.total_bytes - job.downloaded;
	emit('subscribe_download_progress', {
		id: job.id,
		status: job.status,
		downloaded: job.downloaded,
		total: job.total_bytes,
		error: job.error ?? null,
		bytes_per_sec: job.status === 'running' ? 220_000_000 : 0,
		eta_secs: job.status === 'running' ? Math.ceil(remaining / 220_000_000) : undefined,
		phase,
	} satisfies DownloadProgress);
}

/** ~16 ticks to completion regardless of size, so demos finish in ~4s. */
async function simulateDownload(job: Job) {
	job.status = 'running';
	emitDlProgress(job);
	const step = Math.max(1, Math.ceil(job.total_bytes / 16));
	while (job.downloaded < job.total_bytes) {
		await sleep(250);
		// Pause/cancel flips status from outside the loop — just stop ticking.
		if (job.status !== 'running') return;
		job.downloaded = Math.min(job.total_bytes, job.downloaded + step);
		emitDlProgress(job);
	}
	// Verifying: the backend hashes during the transfer (frontier hashing), so
	// only a short tail remains — mirror that with a quick sweep where
	// `downloaded` carries hashed-bytes so the bar visibly moves.
	job.downloaded = Math.floor(job.total_bytes * 0.95);
	emitDlProgress(job, 'verifying');
	await sleep(200);
	if (job.status !== 'running') return;
	job.downloaded = job.total_bytes;
	emitDlProgress(job, 'verifying');
	await sleep(100);
	if (job.status !== 'running') return;
	job.status = 'done';
	if (job.group_id) registerMockGgufSet(job.group_id);
	emitDlProgress(job);
}

// --- Serial download queue (parity with the backend's max_concurrent = 1) ---
// One job transfers at a time; the rest sit Queued and start FIFO.

const dlQueue: string[] = [];
let dlActiveId: string | null = null;

function pumpDlQueue() {
	if (dlActiveId) return;
	while (dlQueue.length > 0) {
		const id = dlQueue.shift()!;
		const job = dlJobs.get(id);
		// Skip jobs paused/cancelled/deleted while waiting in the queue.
		if (!job || job.status !== 'queued' || refuseGgufWrite(job)) continue;
		dlActiveId = id;
		void simulateDownload(job).finally(() => {
			if (dlActiveId === id) dlActiveId = null;
			pumpDlQueue();
		});
		return;
	}
}

function enqueueDownload(job: Job) {
	if (refuseGgufWrite(job)) return;
	job.status = 'queued';
	job.error = undefined;
	dlQueue.push(job.id);
	emitDlProgress(job);
	pumpDlQueue();
}

function newMockJob(repoId: string, revision: string, file: HfFile, groupId: string | null): Job {
	dlCounter += 1;
	const job: Job = {
		id: `mock-dl-${dlCounter}`,
		repo_id: repoId,
		revision,
		file: file.path,
		dest_path: `C:\\mock\\OpenHarness\\models\\staging\\${file.path.replace(/\//g, '_')}`,
		total_bytes: file.size,
		downloaded: 0,
		chunks: [],
		status: 'queued',
		expected_sha256: file.sha256,
		group_id: groupId,
		created_at: now(),
		updated_at: now(),
	};
	dlJobs.set(job.id, job);
	return job;
}

function groupMembers(groupId: string): Job[] {
	return [...dlJobs.values()].filter((j) => j.group_id === groupId);
}

function registerMockGgufSet(groupId: string) {
	const group = dlGroups.get(groupId);
	const members = groupMembers(groupId);
	if (group?.format !== 'gguf' || closedGgufGroups.has(groupId) || !members.length || members.some((j) => j.status !== 'done')) return;
	const primary = members.find((j) => splitGgufPart(j.file)?.number === 1)!;
	const key = splitGgufPart(primary.file)!.key;
	const id = `mock-set:${group.repo_id}@${group.revision}#${key}`;
	const model = {
		id, repo_id: group.repo_id, revision: group.revision,
		display_name: key.split('/').pop()!, architecture: 'glm5-next',
		capability: capabilityForTask(group.task), format: 'gguf' as const,
		task: group.task, local_path: null,
		params_b: 368, quantization: 'Q2_K_XL', context_train: 1048576,
		installed_at: now(), pinned: false,
		primary_file_path: `C:\\mock\\models\\repos\\${group.repo_id.replace('/', '__')}\\${primary.file.replaceAll('/', '\\')}`,
		primary_rel_path: primary.file, primary_sha256: primary.expected_sha256 ?? '',
		primary_file_size: primary.total_bytes, file_count: members.length,
		total_size: members.reduce((sum, j) => sum + j.total_bytes, 0),
	};
	const row = { ...model, caps: resolveMockCaps(model) };
	const existing = MODELS.findIndex((m) => m.id === id || (m.repo_id === group.repo_id
		&& m.revision === group.revision && m.primary_rel_path === primary.file));
	if (existing >= 0) MODELS[existing] = row; else MODELS.push(row);
	modelFiles.set(id, members.map((j) => repoDestination(j.repo_id, j.file)));
	closedGgufGroups.add(groupId);
}

export async function mockInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
	if (cmd.startsWith('server_')) return mockServerInvoke<T>(cmd, args);
	const a = (args ?? {}) as Record<string, never>;
	switch (cmd) {
		// --- shell / settings ------------------------------------------------
		case 'get_setting':
			return (settings.get(a['key']) ?? null) as T;
		case 'set_setting':
			settings.set(a['key'], a['value']);
			return undefined as T;
		case 'consume_migration_notice':
			return false as T;
		case 'app_paths':
			return {
				app_data_dir: 'C:\\mock\\OpenHarness',
				db_path: 'C:\\mock\\OpenHarness\\data.db',
				models_dir: 'C:\\mock\\OpenHarness\\models',
				transcripts_dir: 'C:\\mock\\OpenHarness\\transcripts',
				engine_dir: 'C:\\mock\\OpenHarness\\engine',
			} as T;

		// --- engine / GPU / models --------------------------------------------
		case 'engine_status':
			return engineState as T;
		case 'engine_unload_model':
			mockAssertIdle('text');
			setEngineState({ phase: 'idle' });
			removeResident('text');
			return undefined as T;
		case 'native_engine_info':
			return mockNativeInfo() as T;
		case 'native_engine_install':
			return undefined as T;
		case 'engine_active':
			return (mockEngineSetup().llama ? mockEngineRow() : null) as T;
		case 'engine_install': {
			const variant =
				(a['variant'] as string | undefined) ?? (GPU_OK.kind === 'available' ? 'cuda-12.4' : 'cpu');
			await simulateEngineInstall(variant);
			mockLlamaInstalled = true;
			return mockEngineRow() as T;
		}
		case 'engine_switch_variant':
			await simulateEngineInstall(a['variant'] as string);
			return undefined as T;
		case 'gpu_status':
			return mockGpuFeed() as T;
		case 'system_memory':
			return mockSystemMemory() as T;
		case 'reveal_model':
			// Browser harness: no Explorer to open.
			return undefined as T;
		case 'list_local_models':
			return MODELS as T;
		case 'delete_local_model': {
			const index = MODELS.findIndex((m) => m.id === a['id']);
			if (index >= 0) MODELS.splice(index, 1);
			modelFiles.delete(a['id']);
			return undefined as T;
		}
		case 'estimate_model_fit': {
			// Set localStorage `oh.mock.fitFails` to exercise a failed engine check.
			if (globalThis.localStorage?.getItem('oh.mock.fitFails')) {
				throw new Error('Could not read the graphics card');
			}
			// Mirror the backend: the list badge (no ctx) shows the
			// weights+overhead floor; the load dialog (ctx slider) adds KV at the
			// chosen context so the estimate + verdict track the slider.
			// Floor = file size + fixed overhead; free comes from the mock GPU
			// feed and the loaded text engine counts as reclaimable (a chat
			// load is a same-slot swap), exactly like commands::engine.
			const ctxArg = a['ctx'] as number | undefined;
			const fitModel = MODELS.find((m) => m.id === a['modelId']);
			if (fitModel?.caps.runtime === 'llama-server') {
				const plan = mockPlan(fitModel.id, ctxArg === undefined ? {} : { ctx: ctxArg });
				const gpu = !mockNoGpu();
				return { plan, estimated_mb: gpu ? planVramMb(plan) : Math.ceil(plan.tiers.ram / MIB), kv_at_ctx_mb: Math.ceil((gpu ? plan.tiers.vram_kv : contextCacheBytes(plan.context)) / MIB), free_mb: gpu ? mockGpuStatus().vram_free_mb : 0, reclaimable_mb: gpu ? residents.find((r) => r.kind === 'text_engine')?.est_mb ?? 0 : 0, verdict: plan.feasible ? 'fits' : 'wont_fit', gpu } as T;
			}
			const FLOOR = fitModel
				? Math.round(fitModel.primary_file_size / (1024 * 1024)) + 1100
				: 7421;
			const FREE = mockGpuStatus().vram_free_mb;
			const engine = residents.find((r) => r.kind === 'text_engine');
			const RECLAIM = engine ? (engine.actual_mb ?? engine.est_mb) : 0;
			const kvMb = ctxArg ? Math.round(ctxArg * 0.09) : 0;
			const est = FLOOR + kvMb;
			if (mockNoGpu()) {
				// Mirrors commands::engine: no GPU = CPU inference, always "fits".
				return { estimated_mb: est, kv_at_ctx_mb: kvMb, free_mb: 0, reclaimable_mb: 0, verdict: 'fits', gpu: false,
				} as T;
			}
			const room = FREE + RECLAIM;
			const fitVerdict = est <= room - 1024 ? 'fits' : est <= room ? 'tight' : 'wont_fit';
			return {
				estimated_mb: est,
				kv_at_ctx_mb: kvMb,
				free_mb: FREE,
				reclaimable_mb: RECLAIM,
				verdict: fitVerdict,
				gpu: true,
			} as T;
		}
		case 'engine_logs':
			return ENGINE_LOG_LINES as T;

		// --- load planner -----------------------------------------------------
		case 'hardware_profile':
			return mockHardware() as T;
		case 'benchmark_ssd': {
			await sleep(1500);
			const seconds = 0.6 + Math.random() * 0.04;
			ssdMeasurement = {
				mount: 'C:\\',
				gbps: Number(((4 * GIB) / seconds / 1e9).toFixed(2)),
				bytes_read: 4 * GIB,
				seconds,
				measured_at: now(),
				file:
					MODELS.find((m) => m.id === a['modelId'])?.primary_file_path ??
					'C:\\mock\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
			};
			return ssdMeasurement as T;
		}
		case 'context_memory': {
			const acct = mockAccountingFor(a['modelId'] as string);
			const params: ContextParams = {
				ctx: 4096,
				kv_k: 'f16',
				kv_v: 'f16',
				n_parallel: 1,
				ubatch: 512,
				experts_on_host: false,
				...((a['params'] as Partial<ContextParams> | undefined) ?? {}),
			};
			const ctxs = (a['contexts'] as number[] | undefined)?.length
				? (a['contexts'] as number[])
				: [params.ctx];
			return ctxs.map((ctx) => mockContextMemory(acct, { ...params, ctx })) as T;
		}
		case 'plan_load':
			if (globalThis.localStorage?.getItem('oh.mock.fitFails')) {
				throw new Error('Could not read the graphics card');
			}
			return mockPlan(
				a['modelId'] as string,
				(a['overrides'] as PlanOverrides | undefined) ?? {},
			) as T;

		// --- HF browser (canned, offline) ---------------------------------------
		case 'hf_search': {
			const params = (a['params'] ?? {}) as {
				query?: string;
				task?: string;
				library?: string;
			};
			const q = (params.query ?? '').trim().toLowerCase();
			const results = HF_REPOS.map((r) => r.card).filter((c) => {
				if (q && !c.id.toLowerCase().includes(q)) return false;
				if (params.task && params.task !== 'any' && c.pipeline_tag !== params.task) return false;
				if (params.library && params.library !== 'any') {
					const lib = params.library;
					const tags = c.tags.map((t) => t.toLowerCase());
					if (lib === 'gguf' || lib === 'safetensors') {
						if (c.library_name !== lib && !tags.includes(lib)) return false;
					} else if (c.library_name !== lib) {
						return false;
					}
				}
				return true;
			});
			return { results, page: 1, has_next: false } as T;
		}
		case 'hf_model_details': {
			const entry = HF_REPOS.find((r) => r.card.id === a['repoId']);
			if (!entry) throw new Error(`mock backend: unknown repo ${a['repoId']}`);
			return {
				card: entry.card,
				readme: entry.readme,
				gguf: undefined,
				siblings: entry.siblings,
				layout: entry.layout,
			} satisfies HfModelDetails as T;
		}
		case 'hf_model_tree': {
			const entry = HF_REPOS.find((r) => r.card.id === a['repoId']);
			return (entry?.siblings ?? []) as T;
		}
		case 'get_hf_token':
			return null as T;

		// --- store intelligence (M9 §9.4) ---------------------------------------
		case 'store_runnability':
			return (a['items'] as RunnabilityQuery[]).map(mockResolve) as T;
		case 'store_curated_picks': {
			const task = a['task'] as string | undefined;
			return CURATED_PICKS.filter((p) => !task || p.task === task).map((p) => ({
				...p,
				fits: GPU_OK.vram_total_mb >= p.min_vram_mb,
			})) as T;
		}
		case 'store_quant_fits':
			return mockQuantFits(a['files'] as HfFile[], GPU_OK.vram_free_mb) as T;

		// --- downloads (simulated, M3 jobs + M9 groups) --------------------------
		case 'start_download': {
			if (splitGgufPart((a['file'] as HfFile).path)) throw new Error('This file is part of a split GGUF. Download the complete set instead.');
			const job = newMockJob(a['repoId'], a['revision'], a['file'] as HfFile, null);
			enqueueDownload(job);
			return job.id as T;
		}
		case 'start_repo_download': {
			const files = a['files'] as HfFile[];
			const requested = new Set(files.map((f) => repoDestination(a['repoId'], f.path)));
			if (a['format'] === 'gguf') {
				if (requested.size !== files.length) throw new Error('This download contains parts that point to the same file.');
				if (!files.length || !splitGgufPart(files[0].path)
					|| splitGgufFiles(files[0], files).length !== files.length) throw new Error('Download one complete split GGUF set.');
				if ([...ownedDestinations()].some((p) => requested.has(p))) throw new Error(installedDownloadError);
			}
			if ([...dlJobs.values()].some((j) => !['done', 'cancelled'].includes(j.status)
				&& !(a['format'] === 'gguf' && j.group_id && closedGgufGroups.has(j.group_id))
				&& (a['format'] === 'gguf' ? requested.has(repoDestination(j.repo_id, j.file))
					: j.repo_id === a['repoId'] && files.some((f) => f.path === j.file)))) throw new Error('These files already have a download. Resume or remove it first.');
			if (a['format'] === 'gguf') {
				for (const j of dlJobs.values()) {
					if (j.group_id && dlGroups.get(j.group_id)?.format === 'gguf'
						&& requested.has(repoDestination(j.repo_id, j.file))) closedGgufGroups.add(j.group_id);
				}
			}
			dlCounter += 1;
			const groupId = `mock-grp-${dlCounter}`;
			dlGroups.set(groupId, {
				id: groupId,
				repo_id: a['repoId'],
				revision: a['revision'],
				format: a['format'],
				task: (a['task'] as string | null) ?? null,
				created_at: now(),
			});
			for (const file of a['files'] as HfFile[]) {
				const job = newMockJob(a['repoId'], a['revision'], file, groupId);
				enqueueDownload(job);
			}
			return groupId as T;
		}
		case 'pause_download': {
			const job = dlJobs.get(a['id']);
			if (job && (job.status === 'running' || job.status === 'queued')) {
				job.status = 'paused';
				emitDlProgress(job);
			}
			return undefined as T;
		}
		case 'resume_download': {
			const job = dlJobs.get(a['id']);
			if (job && (job.status === 'paused' || job.status === 'failed')) {
				enqueueDownload(job);
			}
			return undefined as T;
		}
		case 'cancel_download': {
			const job = dlJobs.get(a['id']);
			if (job && job.status !== 'done') {
				job.status = 'cancelled';
				emitDlProgress(job);
			}
			return undefined as T;
		}
		case 'delete_download':
			dlJobs.delete(a['id']);
			return undefined as T;
		case 'pause_download_group': {
			for (const job of groupMembers(a['id'])) {
				if (job.status === 'running' || job.status === 'queued') {
					job.status = 'paused';
					emitDlProgress(job);
				}
			}
			return undefined as T;
		}
		case 'resume_download_group': {
			for (const job of groupMembers(a['id'])) {
				if (job.status === 'paused' || job.status === 'failed') enqueueDownload(job);
			}
			return undefined as T;
		}
		case 'cancel_download_group': {
			for (const job of groupMembers(a['id'])) {
				if (job.status !== 'done') {
					job.status = 'cancelled';
					emitDlProgress(job);
				}
			}
			return undefined as T;
		}
		case 'delete_download_group': {
			for (const job of groupMembers(a['id'])) dlJobs.delete(job.id);
			dlGroups.delete(a['id']);
			return undefined as T;
		}
		case 'list_downloads':
			return [...dlJobs.values()].sort((x, y) => y.created_at - x.created_at) as T;
		case 'list_download_groups':
			return [...dlGroups.values()] as T;

		// --- chat (M5) -------------------------------------------
		case 'list_conversations':
			return [...mockConvs.values()]
				.map((c) => ({ ...c.summary }))
				.sort((x, y) => y.updated_at - x.updated_at) as T;
		case 'create_conversation': {
			convCounter += 1;
			const id = `mock-chat-${convCounter}`;
			const conv: MockConversation = {
				summary: {
					id, title: null, model_id: MODEL_ID,
					created_at: now(), updated_at: now(), message_count: 0,
				},
				messages: [],
			};
			mockConvs.set(id, conv);
			return { ...conv.summary } as T;
		}
		case 'get_conversation':
			return (mockConvs.get(a['id'])?.messages.map((m) => ({ ...m })) ?? []) as T;
		case 'delete_conversation':
			mockConvs.delete(a['id']);
			return undefined as T;
		case 'send_message':
			void simulateChatReply(a['convId'], a['content']);
			return undefined as T;
		case 'cancel_stream':
			return undefined as T;
		case 'chat_truncate_from':
			mockTruncateFrom(a['convId'], Number(a['fromSeq']));
			return undefined as T;
		case 'chat_regenerate':
			void simulateRegenerate(a['convId'], Number(a['seq']));
			return undefined as T;
		case 'get_conversation_settings':
			return {
				system_prompt: null,
				sampling: {
					temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0.05,
					repeat_penalty: 1.1, presence_penalty: 0, frequency_penalty: 0, stop: [],
				},
				load_config: null,
			} as T;
		case 'set_conversation_settings':
			return undefined as T;
		case 'search_messages':
			return [] as T;
		case 'save_voice_capture':
			return 'C:\\mock\\OpenHarness\\assets\\captures\\voice-mock.webm' as T;

		// --- Python worker (M8) --------------------------------------------------
		case 'worker_status':
			startWorkerAutoFlow();
			return workerState as T;
		case 'worker_install': {
			const variant = a['variant'] as string;
			const stages: [string, number][] = [
				['resolving python', 8],
				['downloading wheels', 35],
				['downloading wheels', 72],
				['installing packages', 90],
				['verifying venv', 100],
			];
			const total = variant === 'cuda' ? 6_400_000_000 : 900_000_000;
			for (const [stage, pct] of stages) {
				setWorkerState({
					phase: 'installing', variant, stage, pct,
					bytes: Math.round((total * pct) / 100), total,
				});
				await sleep(450);
			}
			setWorkerState({ phase: 'stopped', variant });
			return undefined as T;
		}
		case 'worker_start': {
			if (workerState.phase === 'not_installed') {
				throw new Error('worker venv is not installed — run worker_install first');
			}
			const variant = 'variant' in workerState ? workerState.variant : 'cuda';
			setWorkerState({ phase: 'starting' });
			await sleep(900);
			setWorkerState({
				phase: 'ready',
				variant,
				version: '0.1.0',
				pipelines: WORKER_PIPELINES,
			});
			return WORKER_PIPELINES as T;
		}
		case 'worker_stop': {
			const variant = 'variant' in workerState ? workerState.variant : 'cuda';
			setWorkerState({ phase: 'stopped', variant });
			// Worker exit reports job_error for every job it still held
			// (sidecar::read_stdout); Qwen3-TTS sidecar and CLI jobs aren't its.
			for (const job of generationJobs) {
				if (job.status !== 'running' || !mockIsWorkerJob(job)) continue;
				job.status = 'error';
				job.error = 'worker process exited';
				job.finished_at = now();
				emit('subscribe_generation_progress', {
					event: 'job_error', job_id: job.id, error: 'worker process exited',
				} satisfies GenerationEvent);
			}
			// Worker death clears every pipeline resident from the ledger.
			residents = residents.filter((r) => r.kind !== 'pipeline');
			emitLedger();
			return undefined as T;
		}
		case 'worker_logs':
			return (workerState.phase === 'ready' ? WORKER_LOG_LINES : []) as T;

		// --- orchestrator (M8 §8.4) ----------------------------------------------
		case 'orchestrator_ledger':
			return ledgerSnapshot() as T;
		case 'orchestrator_unload':
			mockAssertIdle(a['residentId'] as string);
			if (a['residentId'] === 'tts') ttsState = { phase: 'idle' };
			if (a['residentId'] === 'text') setEngineState({ phase: 'idle' });
			removeResident(a['residentId'] as string);
			return undefined as T;
		case 'orchestrator_unload_if': {
			// Mirrors orchestrator::unload_guard: refuse when the slot now holds
			// a different model, no-op when it is already empty.
			const rid = a['residentId'] as string;
			const expected = (a['expectedModelId'] as string | null | undefined) ?? null;
			const r = residents.find((x) => x.id === rid);
			if (!r) return undefined as T;
			if ((r.model_id ?? null) !== expected) {
				throw new Error(`Nothing was unloaded: that slot now holds ${r.label}.`);
			}
			mockAssertIdle(rid);
			if (rid === 'tts') ttsState = { phase: 'idle' };
			if (rid === 'text') setEngineState({ phase: 'idle' });
			removeResident(rid);
			return undefined as T;
		}
		case 'orchestrator_load_model': {
			const req = a['req'] as TextLoadRequest;
			const catalog = MODELS.find((m) => m.id === req.model_id);
			const label = catalog?.display_name ?? req.model_id;
			if (req.ngl === undefined) {
				// No explicit -ngl: the planner chooses (ctx / KV type pin it).
				const overrides: PlanOverrides = { ...(req.overrides ?? {}) };
				if (req.ctx !== undefined) overrides.ctx = req.ctx;
				if (req.kv_cache_type) overrides.kv_type = req.kv_cache_type as KvType;
				const r = mockPlannedTextLoad(req.model_id, Boolean(req.force), req.plan, overrides);
				if (r.outcome === 'needs_eviction') return r as T;
				await sleep(600);
				commitTextLoad(req.model_id, label, r.plan.ctx, planVramMb(r.plan), r.plan);
				return {
					outcome: 'loaded', port: 8732, model_id: req.model_id,
					ctx: r.plan.ctx, ngl: r.plan.gpu_layers, plan: r.plan,
				} as T;
			}
			// Legacy tuned load with an explicit -ngl.
			if (!mockEngineSetup().llama) throw new Error('Engine not installed — run engine_install first');
			const est = 7421;
			const gate = mockGate('text', est, Boolean(req.force));
			if (gate) return gate as T;
			// The engine reports Loading while the ledger still lists the old
			// text resident, which is the window a stale dock × can hit.
			setEngineState({ phase: 'loading', model_id: req.model_id, launcher: 'llama_server' });
			await sleep(600);
			commitTextLoad(req.model_id, label, req.ctx ?? 8192, est, null);
			return { outcome: 'loaded', port: 8732, model_id: req.model_id, ctx: req.ctx ?? 8192, ngl: req.ngl,
			} as T;
		}

		// --- unified load-first flow (M16 W1) --------------------------------------
		case 'load_model': {
			const id = a['modelId'] as string;
			const force = Boolean(a['force']);
			const model = MODELS.find((m) => m.id === id);
			if (!model) throw new Error(`unknown model: ${id}`);
			const caps = model.caps;
			if (!caps.runtime) {
				throw new Error(caps.reason ?? 'No local runtime recognizes this model');
			}
			if (caps.runtime === 'llama-server') {
				// Mirrors orchestrator::text_swap: a load restarts llama-server,
				// so it waits while a reply streams. Only a plain same-model
				// request (no force, plan or overrides) reuses the live model.
				if (mockChatTurns > 0 && engineState.phase === 'ready') {
					const plain = !force && a['plan'] == null && a['overrides'] == null;
					if (engineState.model_id === id && plain) {
						const live = textPlan?.modelId === id ? textPlan.plan : undefined;
						return { outcome: 'loaded', model_id: id, resident_id: 'text', plan: live,
						} as T;
					}
					throw new Error(
						'A chat reply is streaming from the loaded model. Stop it in Chat before loading another model.',
					);
				}
				const r = mockPlannedTextLoad(
					id,
					force,
					a['plan'] as LoadPlan | undefined,
					a['overrides'] as PlanOverrides | undefined,
				);
				if (r.outcome === 'needs_eviction') return r as T;
				setEngineState({ phase: 'loading', model_id: id, launcher: r.plan.engine_kind });
				await sleep(900);
				if (globalThis.localStorage?.getItem('oh.mock.loadFails')) {
					const error = 'Engine process exited during startup';
					setEngineState({ phase: 'failed', error, launcher: r.plan.engine_kind });
					throw new Error(error);
				}
				commitTextLoad(id, model.display_name, r.plan.ctx, planVramMb(r.plan), r.plan);
				return { outcome: 'loaded', model_id: id, resident_id: 'text', plan: r.plan,
				} as T;
			}
			if (caps.runtime === 'qwen-tts') {
				// Mirrors orchestrator::tts_swap.
				if (ttsState.phase === 'ready' && ttsState.model_id === id) {
					return { outcome: 'loaded', model_id: id, resident_id: 'tts' } as T;
				}
				if (ttsState.phase === 'ready' && mockTtsLeased()) {
					throw new Error(
						'A speech job is using the loaded voice model. Let it finish or cancel it before loading another one.',
					);
				}
				const gate = mockGate('tts', 3100, force);
				if (gate) return gate as T;
				await sleep(700);
				mockLoadTts(id);
				return { outcome: 'loaded', model_id: id, resident_id: 'tts' } as T;
			}
			// Worker pipeline — explicit preload. Mirrors preload_pipeline's
			// worker-state preconditions (honest errors; auto-start from stopped).
			const task = caps.pipeline ?? 'echo';
			if (workerState.phase === 'not_installed') {
				throw new Error(
					"The Python runtime isn't set up yet — it installs automatically on first run; check Settings → Engines for progress",
				);
			}
			if (workerState.phase === 'installing') {
				throw new Error('The Python runtime is still installing — try again when it finishes');
			}
			if (workerState.phase === 'failed') {
				throw new Error(`The Python runtime failed to start: ${workerState.error}`);
			}
			if (workerState.phase === 'stopped' || workerState.phase === 'starting') {
				const variant = 'variant' in workerState ? workerState.variant : 'cuda';
				setWorkerState({ phase: 'starting' });
				await sleep(600);
				setWorkerState({
					phase: 'ready',
					variant,
					version: '0.1.0',
					pipelines: WORKER_PIPELINES,
				});
			}
			const residentId = `pipeline:${task}`;
			const est = TASK_VRAM_MB[task] ?? 2000;
			const gate = mockGate(residentId, est, force);
			if (gate) return gate as T;
			residents = residents.filter((r) => r.id !== residentId);
			residents.push({
				id: residentId, kind: 'pipeline', label: model.display_name,
				model_id: id, est_mb: est, actual_mb: null,
			});
			emitLedger();
			await sleep(1100);
			const res = residents.find((r) => r.id === residentId);
			if (res) res.actual_mb = est;
			emitLedger();
			return { outcome: 'loaded', model_id: id, resident_id: residentId } as T;
		}
		case 'unload_model': {
			const id = a['modelId'] as string;
			const r = residents.find((x) => x.model_id === id);
			if (r) {
				mockAssertIdle(r.id);
				if (r.id === 'text') setEngineState({ phase: 'idle' });
				if (r.id === 'tts') ttsState = { phase: 'idle' };
				removeResident(r.id);
			}
			return undefined as T;
		}

		// --- generation (M8 §8.5) --------------------------------------------------
		case 'generate_image':
			return submitMockJob('text_to_image', a['params'], Boolean(a['force'])) as T;
		case 'generate_video':
			return submitMockJob('text_to_video', a['params'], Boolean(a['force'])) as T;
		case 'synthesize_speech': {
			// Mirrors the Rust command's routing: qwen3-tts (incl. CLI voice
			// clones with reference_audio) stays text_to_speech; engine
			// "vibevoice" (or any other reference_audio request) runs the
			// voice-cloning worker pipeline.
			const sp = (a['params'] ?? {}) as Record<string, unknown>;
			const cloning =
				sp['engine'] !== 'qwen3-tts' &&
				(sp['engine'] === 'vibevoice' || sp['reference_audio'] !== undefined);
			return submitMockJob(
				cloning ? 'vibevoice' : 'text_to_speech',
				a['params'],
				Boolean(a['force']),
			) as T;
		}
		case 'tts_status':
			return ttsState as T;
		case 'tts_installed':
			return ttsInstalled as T;
		case 'tts_cli_installed':
			return ttsInstalled as T;
		case 'tts_install':
			// Mirrors the actor's lazy install: a no-op when already
			// provisioned, otherwise a streamed install ending Idle.
			if (!ttsInstalled) await simulateTtsInstall();
			return undefined as T;
		case 'tts_unload':
			mockAssertIdle('tts');
			setTtsState({ phase: 'idle' });
			removeResident('tts');
			return undefined as T;
		case 'tts_logs':
			return (ttsState.phase === 'idle' ? [] : TTS_LOG_LINES) as T;
		case 'transcribe_audio':
			return submitMockJob('transcribe', a['params'], Boolean(a['force'])) as T;
		case 'active_generation_jobs':
			return [
				...generationJobs
					.filter((j) => j.status === 'running')
					.map((j) => ({ id: j.id, task: j.task })),
				...[...mockUnwrittenLeases].map(([id, task]) => ({ id, task })),
			] as T;
		case 'list_generation_jobs': {
			const task = a['task'] as string | undefined;
			const limit = (a['limit'] as number | undefined) ?? 100;
			return generationJobs
				.filter((j) => !task || j.task === task)
				.sort((x, y) => y.created_at - x.created_at)
				.slice(0, limit)
				.map((j) => ({ ...j })) as T;
		}
		case 'cancel_generation_job': {
			const job = generationJobs.find((j) => j.id === a['jobId']);
			if (job && job.status === 'running') {
				job.status = 'cancelled';
				job.error = 'cancelled';
				job.finished_at = now();
			}
			return undefined as T;
		}
		case 'delete_generation_job':
			generationJobs = generationJobs.filter((j) => j.id !== a['jobId']);
			return undefined as T;

		// --- asset library (M10) ---------------------------------------------------
		case 'list_assets': {
			const kind = a['kind'] as AssetKind | undefined;
			const limit = (a['limit'] as number | undefined) ?? 100;
			const offset = (a['offset'] as number | undefined) ?? 0;
			return assets
				.filter((x) => !kind || x.kind === kind)
				.sort((x, y) => y.created_at - x.created_at)
				.slice(offset, offset + limit)
				.map((x) => ({ ...x })) as T;
		}
		case 'search_assets': {
			const q = String(a['query'] ?? '').toLowerCase();
			const limit = (a['limit'] as number | undefined) ?? 100;
			const tokens = q.split(/[^a-z0-9]+/).filter(Boolean);
			return assets
				.filter((x) => {
					const hay = `${x.prompt ?? ''} ${x.model ?? ''} ${x.task}`.toLowerCase();
					return tokens.every((t) => hay.includes(t));
				})
				.slice(0, limit)
				.map((x) => ({ ...x })) as T;
		}
		case 'delete_asset':
			assets = assets.filter((x) => x.id !== a['assetId']);
			return undefined as T;
		case 'reveal_asset':
			console.info('[mock] reveal_asset', a['assetId']);
			return undefined as T;


		default:
			console.warn(`[mock] unhandled command: ${cmd}`, args);
			return undefined as T;
	}
}
