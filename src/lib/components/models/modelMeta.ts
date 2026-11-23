/**
 * Display helpers for installed models, shared by the Models table and the
 * inspector. Plain-language labels only; no fetching.
 */
import type { InstalledModel, LoadPlan, ModelFormat, RuntimeKind, Workflow } from '$lib/types';

export const WORKFLOW_LABELS: Record<Workflow, string> = {
	chat: 'Text',
	image: 'Image',
	speech: 'Speech',
	transcribe: 'Transcribe',
	video: 'Video',
	embeddings: 'Embeddings',
};

/** Header filter tabs, in display order. */
export const TASK_TABS: { id: 'all' | Workflow; label: string }[] = [
	{ id: 'all', label: 'All' },
	{ id: 'chat', label: 'Text' },
	{ id: 'image', label: 'Image' },
	{ id: 'video', label: 'Video' },
	{ id: 'speech', label: 'Speech' },
	{ id: 'transcribe', label: 'Transcribe' },
	{ id: 'embeddings', label: 'Embeddings' },
];

export const FORMAT_LABELS: Record<ModelFormat, string> = {
	gguf: 'GGUF',
	safetensors: 'Safetensors',
	'diffusers-dir': 'Diffusers',
};

export const RUNTIME_LABELS: Record<RuntimeKind, string> = {
	'llama-server': 'llama.cpp',
	worker: 'Python worker',
	'qwen-tts': 'Qwen3-TTS',
};

/** Which text engines can launch a model right now (see engineStore.textEngines). */
export interface TextEngines {
	/** llama.cpp is installed (an active row exists). */
	llama: boolean;
	/** The native engine probe has answered at least once. */
	nativeChecked: boolean;
	/** The native engine is installed and reports ssd_expert_streaming. */
	nativeStreaming: boolean;
}

export const LLAMA_NEEDED = 'Install the llama.cpp engine first (Settings, Engines)';
export const NATIVE_NEEDED =
	'The native engine is needed for this model. See Settings, Engines for where to place it.';
export const PLAN_PENDING = 'Checking which engine can run this model…';
export const CHECK_FAILED = "Couldn't check which engine can run this model";

/**
 * Why a text model can't start with the engines on this machine, or null when
 * it can. The planner decides the launcher (`plan.engine_kind`), never the
 * model's size or name, so this is the only rule the load buttons follow.
 *
 * `fit` is the installed-text estimate: null while it is in flight, and a fit
 * without a plan means the model has no plan and takes the llama-only path. No
 * answer yet never counts as allowed: an engine can't be assumed for a model
 * the planner may still send elsewhere, and neither does a failed check: it
 * blocks with its error until a retry answers. Only a model with no estimate
 * at all is known to take the llama-only path.
 */
export function textEngineBlock(
	m: InstalledModel,
	fit: { plan?: LoadPlan | null } | null,
	fitError: string | null,
	engines: TextEngines,
): string | null {
	if (m.caps.runtime !== 'llama-server') return null;
	if (!hasFitEstimate(m)) return engines.llama ? null : LLAMA_NEEDED;
	if (fitError) return `${CHECK_FAILED}: ${fitError}`;
	if (!fit) {
		return engines.llama || engines.nativeStreaming || !engines.nativeChecked
			? PLAN_PENDING
			: LLAMA_NEEDED;
	}
	if (fit.plan?.engine_kind === 'ssd_stream') {
		return engines.nativeStreaming ? null : (fit.plan.infeasible_reason ?? NATIVE_NEEDED);
	}
	return engines.llama ? null : LLAMA_NEEDED;
}

/**
 * Whether "Load with options" is worth opening: the dialog can switch a model
 * to the native engine, so it stays open when either engine can run.
 */
export function canOpenLoadOptions(m: InstalledModel, engines: TextEngines): boolean {
	return m.caps.runtime !== 'llama-server' || engines.llama || engines.nativeStreaming;
}

/** Studio workflows a model can be opened in (everything except chat/embeddings). */
export const STUDIO_WORKFLOWS: Workflow[] = ['image', 'video', 'speech', 'transcribe'];

export function sizeOnDisk(m: InstalledModel): number {
	return m.file_count > 1 ? m.total_size : m.primary_file_size;
}

/** Embedding-only and unrecognised models have nothing to load into. */
export function isLoadable(m: InstalledModel): boolean {
	if (m.caps.runtime === null) return false;
	return m.caps.workflows.some((wf) => wf !== 'embeddings');
}

/** The first Studio tab this model belongs to, if any. */
export function studioTask(m: InstalledModel): Workflow | null {
	return m.caps.workflows.find((wf) => STUDIO_WORKFLOWS.includes(wf)) ?? null;
}

/** Fit estimates only exist for llama.cpp text models (GGUF weights). */
export function hasFitEstimate(m: InstalledModel): boolean {
	return m.format === 'gguf' && m.caps.runtime === 'llama-server';
}

/** Context a plain Load uses when none is given. */
export const MAX_DEFAULT_CTX = 8192;

/**
 * The context length a plain Load launches llama-server with: the trained
 * context, capped at 8K. Mirrors the orchestrator's `min(context_train, 8192)`
 * so the fit estimate and the actual load agree.
 */
export function defaultCtx(m: Pick<InstalledModel, 'context_train'>): number {
	const t = m.context_train;
	if (t == null || !(t > 0)) return MAX_DEFAULT_CTX;
	return Math.min(Math.floor(t), MAX_DEFAULT_CTX);
}

/** 8192 → "8K", 2048 → "2K", 6000 → "5.9K", 512 → "512". */
export function ctxLabel(ctx: number): string {
	if (ctx < 1024) return String(ctx);
	const k = ctx / 1024;
	return `${Number.isInteger(k) ? k : k.toFixed(1)}K`;
}

/**
 * The orchestrator slot a load of this model fills: "text" for llama.cpp,
 * "tts" for the Qwen3-TTS sidecar, "pipeline:<task>" for worker pipelines.
 * Loading into an occupied slot replaces whatever is there.
 */
export function residentSlot(m: InstalledModel): string | null {
	switch (m.caps.runtime) {
		case 'llama-server':
			return 'text';
		case 'qwen-tts':
			return 'tts';
		case 'worker':
			return m.caps.pipeline ? `pipeline:${m.caps.pipeline}` : null;
		default:
			return null;
	}
}

/** "Qwen2.5 Coder" → family from the GGUF architecture, title-cased. */
export function family(m: InstalledModel): string {
	const a = m.architecture?.trim();
	if (!a) return '—';
	return a.charAt(0).toUpperCase() + a.slice(1);
}

export function params(m: InstalledModel): string {
	const b = m.params_b;
	if (b == null || !(b > 0)) return '—';
	if (b < 1) return `${Math.round(b * 1000)}M`;
	return `${b >= 10 ? Math.round(b) : b.toFixed(1)}B`;
}
