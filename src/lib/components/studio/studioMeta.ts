import type { GenerationTask } from '$lib/types';

/**
 * Pure helpers for the Studio surfaces: job-stage wording, recent-strip
 * kind labels and the library's asset captions.
 */

/** Worker stage names → sentence-case labels for the running job card. */
const STAGES: Record<string, string> = {
	submitting: 'Starting',
	queued: 'Waiting to start',
	'loading model': 'Loading the model',
	loaded: 'Model loaded',
	denoising: 'Generating',
	'encoding video': 'Encoding the video',
	'muxing audio': 'Adding the audio',
	'preparing voices': 'Preparing voices',
	synthesizing: 'Speaking',
	transcribing: 'Transcribing',
	saving: 'Saving',
};

/** Sentence-case label for a worker job stage. Unknown stages pass through capitalised. */
export function stageLabel(stage: string | null | undefined): string {
	const s = (stage ?? '').trim().toLowerCase();
	if (!s) return 'Working';
	if (STAGES[s]) return STAGES[s];
	return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Short label for an asset kind badge (Image, Video, Audio, Text). */
export function kindLabel(kind: string): string {
	switch (kind) {
		case 'image':
			return 'Image';
		case 'video':
			return 'Video';
		case 'audio':
			return 'Audio';
		case 'transcript':
		case 'text':
			return 'Text';
		default:
			return kind ? kind.charAt(0).toUpperCase() + kind.slice(1) : 'File';
	}
}

/** File name from a Windows or POSIX path. */
export function baseName(path: string): string {
	return path.split(/[\\/]/).pop() ?? path;
}

/**
 * Seconds of video for a frame count at a frame rate, to one decimal place
 * (e.g. 121 frames at 24 fps → "5.0").
 */
export function videoSeconds(frames: number, fps: number): string {
	if (!Number.isFinite(frames) || !Number.isFinite(fps) || fps <= 0) return '0.0';
	return (frames / fps).toFixed(1);
}

/**
 * Which Studio tab and task page can restore an asset from the library,
 * keyed by the asset's source task. Null when no task page produced it.
 */
export function remixTarget(assetTask: string): { tab: string; task: GenerationTask } | null {
	switch (assetTask) {
		case 'text_to_image':
			return { tab: 'image', task: 'text_to_image' };
		case 'text_to_video':
			return { tab: 'video', task: 'text_to_video' };
		case 'text_to_speech':
		case 'vibevoice':
			return { tab: 'speech', task: 'text_to_speech' };
		case 'transcribe':
			return { tab: 'transcribe', task: 'transcribe' };
		default:
			return null;
	}
}

/**
 * One-line settings summary for the library inspector, built from a job's
 * params ("1024 × 1024, 30 steps, guidance 5, seed 42"). Unknown keys are
 * skipped; returns '' when nothing worth showing is present.
 */
export function paramSummary(params: unknown): string {
	if (!params || typeof params !== 'object') return '';
	const p = params as Record<string, unknown>;
	const num = (k: string) => (typeof p[k] === 'number' && Number.isFinite(p[k]) ? (p[k] as number) : null);
	const parts: string[] = [];
	const w = num('width');
	const h = num('height');
	if (w !== null && h !== null) parts.push(`${w} × ${h}`);
	const frames = num('num_frames');
	const fps = num('fps');
	if (frames !== null && fps !== null) parts.push(`${frames} frames at ${fps} fps`);
	const steps = num('steps');
	if (steps !== null) parts.push(`${steps} steps`);
	const guidance = num('guidance');
	if (guidance !== null) parts.push(`guidance ${guidance}`);
	if (typeof p['voice'] === 'string' && p['voice']) parts.push(`voice ${p['voice']}`);
	const speed = num('speed');
	if (speed !== null && speed !== 1) parts.push(`speed ${speed}×`);
	if (p['task'] === 'translate') parts.push('translated to English');
	if (typeof p['language'] === 'string' && p['language']) parts.push(`language ${p['language']}`);
	const seed = num('seed');
	if (seed !== null) parts.push(`seed ${seed}`);
	return parts.join(', ');
}
