import type { Asset, InstalledModel } from '$lib/types';

/**
 * Restoring a past generation into a Studio form ("reuse settings").
 *
 * Each builder returns a patch the task page applies to its own state. The
 * rule they share: resolve the installed model first. Settings that depend
 * on the model (image architecture, size and steps, the video family's fps
 * and frame rules, speech voices and cloning) come back only when the model
 * that made the asset is still installed. Otherwise the form keeps its
 * current model and presets, restores the model-independent fields (prompt,
 * seed, …) and says why in `note`, leaving every model-dependent setting
 * exactly as it was.
 *
 * The installed list a page passes in must be its ModelPicker's own list
 * (`refresh()`), so the remix and the picker never disagree about which
 * models exist.
 */

/** An installed model as a remix sees it: catalog id plus the path the worker loads. */
export interface RemixModel {
	id: string;
	path: string;
}

export function remixModelOf(m: InstalledModel): RemixModel {
	return { id: m.id, path: m.local_path ?? m.primary_file_path };
}

/** Normalised for comparison: Windows paths are case-insensitive and take either slash. */
function norm(s: string): string {
	return s.trim().replace(/\\/g, '/').toLowerCase();
}

/**
 * The installed model a saved generation used, or null when it is gone.
 * Saved params carry either the load path (worker pipelines) or the catalog
 * id (Qwen3-TTS), so both are matched.
 */
export function findRemixModel(saved: unknown, installed: RemixModel[]): RemixModel | null {
	if (typeof saved !== 'string' || !saved.trim()) return null;
	const key = norm(saved);
	return installed.find((m) => norm(m.path) === key || m.id === saved) ?? null;
}

function paramsOf(a: Asset): Record<string, unknown> {
	return (a.params && typeof a.params === 'object' ? a.params : {}) as Record<string, unknown>;
}

const num = (p: Record<string, unknown>, k: string) =>
	typeof p[k] === 'number' && Number.isFinite(p[k]) ? (p[k] as number) : undefined;
const str = (p: Record<string, unknown>, k: string) => (typeof p[k] === 'string' ? (p[k] as string) : undefined);

function missingNote(what: string): string {
	return `The model that made this ${what} is no longer installed. The current model and its settings stay as they are.`;
}

// ---- Image -----------------------------------------------------------------

export const IMAGE_ARCHS = ['', 'sdxl', 'sd', 'flux'] as const;

export interface ImageRemix {
	prompt?: string;
	negative: string;
	seed: string;
	numImages?: number;
	/** Path to select, or null to keep the current model. */
	model: string | null;
	/** Present only when the source model is installed. */
	arch?: string;
	width?: number;
	height?: number;
	steps?: number;
	guidance?: number;
	/** True when exact numbers were restored, so no preset pill should light. */
	custom: boolean;
	note: string | null;
}

export function imageRemix(a: Asset, installed: RemixModel[]): ImageRemix {
	const p = paramsOf(a);
	const out: ImageRemix = {
		// Older/imported assets may only carry the prompt on the asset row.
		prompt: str(p, 'prompt') ?? a.prompt ?? undefined,
		negative: str(p, 'negative_prompt') ?? '',
		seed: p['seed'] != null ? String(p['seed']) : '',
		numImages: num(p, 'num_images'),
		model: null,
		custom: false,
		note: null,
	};
	const source = findRemixModel(p['model'], installed);
	if (!source) {
		if (p['model'] !== undefined) out.note = missingNote('image');
		return out;
	}
	out.model = source.path;
	const arch = str(p, 'arch');
	if (arch !== undefined) out.arch = (IMAGE_ARCHS as readonly string[]).includes(arch) ? arch : '';
	out.width = num(p, 'width');
	out.height = num(p, 'height');
	out.steps = num(p, 'steps');
	out.guidance = num(p, 'guidance');
	out.custom = true;
	return out;
}

// ---- Video -----------------------------------------------------------------

export interface VideoRemix {
	prompt?: string;
	negative: string;
	seed: string;
	model: string | null;
	resolution?: string;
	fps?: number;
	numFrames?: number;
	steps?: number;
	guidance?: number;
	custom: boolean;
	note: string | null;
}

export function videoRemix(a: Asset, installed: RemixModel[]): VideoRemix {
	const p = paramsOf(a);
	const out: VideoRemix = {
		prompt: str(p, 'prompt') ?? a.prompt ?? undefined,
		negative: str(p, 'negative_prompt') ?? '',
		seed: p['seed'] != null ? String(p['seed']) : '',
		model: null,
		custom: false,
		note: null,
	};
	const source = findRemixModel(p['model'], installed);
	if (!source) {
		if (p['model'] !== undefined) out.note = missingNote('video');
		return out;
	}
	out.model = source.path;
	const w = num(p, 'width');
	const h = num(p, 'height');
	if (w !== undefined && h !== undefined) out.resolution = `${w}x${h}`;
	out.fps = num(p, 'fps');
	out.numFrames = num(p, 'num_frames');
	out.steps = num(p, 'steps');
	out.guidance = num(p, 'guidance');
	out.custom = true;
	return out;
}

// ---- Speech ----------------------------------------------------------------

export interface SpeechRemix {
	text?: string;
	/** Catalog id to select, or null to keep the current model. */
	modelId: string | null;
	voice?: string;
	speed?: number;
	instructions?: string;
	cfgScale?: number;
	refText?: string;
	referenceAudios?: string[];
	cloneVoice?: boolean;
	note: string | null;
}

export function speechRemix(a: Asset, installed: RemixModel[]): SpeechRemix {
	const p = paramsOf(a);
	const out: SpeechRemix = {
		// Older/imported assets may only carry the spoken text on the asset row.
		text: str(p, 'text') ?? a.prompt ?? undefined,
		modelId: null,
		note: null,
	};
	const source = findRemixModel(p['model'], installed);
	if (!source) {
		if (p['model'] !== undefined) out.note = missingNote('clip');
		return out;
	}
	out.modelId = source.id;
	out.voice = str(p, 'voice');
	out.speed = num(p, 'speed');
	out.instructions = str(p, 'instructions') ?? '';
	out.cfgScale = num(p, 'cfg_scale');
	out.refText = str(p, 'ref_text') ?? '';
	const refs = p['reference_audio'];
	if (typeof refs === 'string') out.referenceAudios = [refs];
	else if (Array.isArray(refs) && refs.length > 0) out.referenceAudios = refs.map(String);
	// A cloned generation re-arms the clone toggle on models that have one.
	out.cloneVoice = refs !== undefined;
	return out;
}

// ---- Transcribe ------------------------------------------------------------

export interface TranscribeRemix {
	audioPath?: string;
	model: string | null;
	language: string;
	mode: 'transcribe' | 'translate';
	wordTimestamps: boolean;
	note: string | null;
}

/** Language, mode and timestamps work with any Whisper model, so they always come back. */
export function transcribeRemix(a: Asset, installed: RemixModel[]): TranscribeRemix {
	const p = paramsOf(a);
	const source = findRemixModel(p['model'], installed);
	return {
		audioPath: str(p, 'audio'),
		model: source?.path ?? null,
		language: str(p, 'language') ?? '',
		mode: p['task'] === 'translate' ? 'translate' : 'transcribe',
		wordTimestamps: p['word_timestamps'] === true,
		note: !source && p['model'] !== undefined ? missingNote('transcript') : null,
	};
}
