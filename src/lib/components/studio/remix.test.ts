import { describe, expect, it } from 'vitest';
import { findRemixModel, imageRemix, speechRemix, transcribeRemix, videoRemix, type RemixModel } from './remix';
import type { Asset } from '$lib/types';

function asset(task: string, params: Record<string, unknown>, prompt: string | null = null): Asset {
	return {
		id: 'a1',
		job_id: 'j1',
		kind: 'image',
		path: 'C:/assets/a1.png',
		task,
		prompt,
		model: null,
		params,
		meta: null,
		size: 1,
		created_at: 0,
	} as Asset;
}

const SDXL = 'C:\\OH\\models\\sdxl-base';
const SD15: RemixModel = { id: 'sd15', path: 'C:\\OH\\models\\sd15' };

const sdxlImage = asset('text_to_image', {
	model: SDXL,
	prompt: 'a lighthouse at dawn',
	negative_prompt: 'blurry',
	arch: 'sdxl',
	width: 1216,
	height: 832,
	steps: 40,
	guidance: 6.5,
	num_images: 2,
	seed: 7,
});

describe('image remix', () => {
	it('keeps the current model and its architecture when the source model is gone', () => {
		// Review repro: SDXL uninstalled, only SD 1.5 left.
		const r = imageRemix(sdxlImage, [SD15]);
		expect(r.model).toBeNull();
		expect(r.arch).toBeUndefined();
		expect(r.width).toBeUndefined();
		expect(r.height).toBeUndefined();
		expect(r.steps).toBeUndefined();
		expect(r.guidance).toBeUndefined();
		expect(r.custom).toBe(false);
		expect(r.note).toMatch(/no longer installed/);
		// Model-independent fields still come back.
		expect(r.prompt).toBe('a lighthouse at dawn');
		expect(r.negative).toBe('blurry');
		expect(r.seed).toBe('7');
		expect(r.numImages).toBe(2);
	});

	it('restores everything when the source model is installed', () => {
		const r = imageRemix(sdxlImage, [SD15, { id: 'sdxl', path: 'c:/oh/models/SDXL-base' }]);
		expect(r.model).toBe('c:/oh/models/SDXL-base');
		expect(r).toMatchObject({ arch: 'sdxl', width: 1216, height: 832, steps: 40, guidance: 6.5, custom: true });
		expect(r.note).toBeNull();
	});

	it('drops an unknown architecture to auto-detect', () => {
		const a = asset('text_to_image', { model: SD15.path, arch: 'weird' });
		expect(imageRemix(a, [SD15]).arch).toBe('');
	});

	it('says nothing when the asset never recorded a model', () => {
		const r = imageRemix(asset('text_to_image', {}, 'from the row'), [SD15]);
		expect(r.note).toBeNull();
		expect(r.prompt).toBe('from the row');
	});
});

describe('findRemixModel', () => {
	it('matches paths regardless of slash direction and case', () => {
		expect(findRemixModel('c:/oh/models/SD15', [SD15])).toBe(SD15);
	});
	it('matches a catalog id', () => {
		expect(findRemixModel('sd15', [SD15])).toBe(SD15);
	});
	it('ignores empty and non-string values', () => {
		expect(findRemixModel('', [SD15])).toBeNull();
		expect(findRemixModel(42, [SD15])).toBeNull();
		expect(findRemixModel(undefined, [SD15])).toBeNull();
	});
});

describe('video remix', () => {
	const wan = asset('text_to_video', {
		model: 'C:/OH/models/wan2.1-t2v',
		prompt: 'drone over a forest',
		width: 832,
		height: 480,
		fps: 16,
		num_frames: 65,
		steps: 30,
		guidance: 5,
	});

	it('does not carry the family frame rules onto another model', () => {
		const r = videoRemix(wan, [{ id: 'ltx', path: 'C:/OH/models/ltx-video' }]);
		expect(r.model).toBeNull();
		expect(r.fps).toBeUndefined();
		expect(r.numFrames).toBeUndefined();
		expect(r.resolution).toBeUndefined();
		expect(r.prompt).toBe('drone over a forest');
		expect(r.note).toMatch(/video/);
	});

	it('restores the exact clip settings with its model', () => {
		const r = videoRemix(wan, [{ id: 'wan', path: 'C:/OH/models/wan2.1-t2v' }]);
		expect(r).toMatchObject({ model: 'C:/OH/models/wan2.1-t2v', resolution: '832x480', fps: 16, numFrames: 65 });
	});
});

describe('speech remix', () => {
	const cloned = asset('text_to_speech', {
		model: 'C:/OH/models/vibevoice',
		text: 'Hello there',
		voice: 'Alice',
		reference_audio: ['C:/ref/a.wav', 'C:/ref/b.wav'],
		cfg_scale: 1.3,
	});

	it('keeps voices and cloning off a model that cannot use them', () => {
		const r = speechRemix(cloned, [{ id: 'kokoro', path: 'C:/OH/models/kokoro' }]);
		expect(r.modelId).toBeNull();
		expect(r.voice).toBeUndefined();
		expect(r.referenceAudios).toBeUndefined();
		expect(r.cloneVoice).toBeUndefined();
		expect(r.text).toBe('Hello there');
		expect(r.note).toMatch(/clip/);
	});

	it('selects the source model by catalog id and re-arms cloning', () => {
		const r = speechRemix(cloned, [{ id: 'vv', path: 'C:/OH/models/vibevoice' }]);
		expect(r.modelId).toBe('vv');
		expect(r.voice).toBe('Alice');
		expect(r.referenceAudios).toEqual(['C:/ref/a.wav', 'C:/ref/b.wav']);
		expect(r.cloneVoice).toBe(true);
		expect(r.cfgScale).toBe(1.3);
	});

	it('matches a Qwen3-TTS asset saved with the catalog id', () => {
		const q = asset('text_to_speech', { model: 'qwen-talker', voice: 'Vivian', engine: 'qwen3-tts' });
		const r = speechRemix(q, [{ id: 'qwen-talker', path: 'C:/OH/models/abc.gguf' }]);
		expect(r.modelId).toBe('qwen-talker');
		expect(r.voice).toBe('Vivian');
		expect(r.cloneVoice).toBe(false);
	});
});

describe('transcribe remix', () => {
	const a = asset('transcribe', {
		audio: 'C:/rec.mp3',
		model: 'C:/OH/models/whisper-large',
		task: 'translate',
		language: 'de',
		word_timestamps: true,
	});

	it('restores model-independent options even without the model', () => {
		const r = transcribeRemix(a, []);
		expect(r).toMatchObject({ audioPath: 'C:/rec.mp3', model: null, language: 'de', mode: 'translate', wordTimestamps: true });
		expect(r.note).toMatch(/transcript/);
	});

	it('selects the source model when installed', () => {
		const r = transcribeRemix(a, [{ id: 'w', path: 'C:/OH/models/whisper-large' }]);
		expect(r.model).toBe('C:/OH/models/whisper-large');
		expect(r.note).toBeNull();
	});
});
