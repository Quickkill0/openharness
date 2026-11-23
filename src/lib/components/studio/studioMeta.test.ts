import { describe, expect, it } from 'vitest';
import { baseName, kindLabel, paramSummary, remixTarget, stageLabel, videoSeconds } from './studioMeta';

describe('stageLabel', () => {
	it('maps known worker stages to plain words', () => {
		expect(stageLabel('loading model')).toBe('Loading the model');
		expect(stageLabel('denoising')).toBe('Generating');
		expect(stageLabel('submitting')).toBe('Starting');
		expect(stageLabel('Transcribing')).toBe('Transcribing');
	});
	it('capitalises unknown stages and handles empty input', () => {
		expect(stageLabel('upscaling')).toBe('Upscaling');
		expect(stageLabel('')).toBe('Working');
		expect(stageLabel(null)).toBe('Working');
	});
});

describe('kindLabel', () => {
	it('labels asset kinds', () => {
		expect(kindLabel('image')).toBe('Image');
		expect(kindLabel('audio')).toBe('Audio');
		expect(kindLabel('transcript')).toBe('Text');
		expect(kindLabel('model3d')).toBe('Model3d');
		expect(kindLabel('')).toBe('File');
	});
});

describe('baseName', () => {
	it('takes the last path segment on either separator', () => {
		expect(baseName('C:\\Users\\me\\assets\\out.png')).toBe('out.png');
		expect(baseName('/tmp/a/b.wav')).toBe('b.wav');
		expect(baseName('plain.txt')).toBe('plain.txt');
	});
});

describe('videoSeconds', () => {
	it('converts frames to seconds', () => {
		expect(videoSeconds(121, 24)).toBe('5.0');
		expect(videoSeconds(97, 24)).toBe('4.0');
	});
	it('guards against a zero or bad frame rate', () => {
		expect(videoSeconds(97, 0)).toBe('0.0');
		expect(videoSeconds(Number.NaN, 24)).toBe('0.0');
	});
});

describe('remixTarget', () => {
	it('routes each source task to the tab that can restore it', () => {
		expect(remixTarget('text_to_image')).toEqual({ tab: 'image', task: 'text_to_image' });
		expect(remixTarget('text_to_video')).toEqual({ tab: 'video', task: 'text_to_video' });
		expect(remixTarget('transcribe')).toEqual({ tab: 'transcribe', task: 'transcribe' });
	});
	it('sends VibeVoice clones to the speech page', () => {
		expect(remixTarget('vibevoice')).toEqual({ tab: 'speech', task: 'text_to_speech' });
		expect(remixTarget('text_to_speech')).toEqual({ tab: 'speech', task: 'text_to_speech' });
	});
	it('returns null for assets no task page produced', () => {
		expect(remixTarget('imported')).toBeNull();
		expect(remixTarget('')).toBeNull();
	});
});

describe('paramSummary', () => {
	it('summarises image settings in order', () => {
		expect(paramSummary({ prompt: 'x', width: 1024, height: 768, steps: 30, guidance: 5, seed: 42 })).toBe(
			'1024 × 768, 30 steps, guidance 5, seed 42',
		);
	});
	it('summarises video, speech and transcription settings', () => {
		expect(paramSummary({ num_frames: 97, fps: 24, steps: 20 })).toBe('97 frames at 24 fps, 20 steps');
		expect(paramSummary({ voice: 'af_heart', speed: 1.2 })).toBe('voice af_heart, speed 1.2×');
		expect(paramSummary({ voice: 'af_heart', speed: 1 })).toBe('voice af_heart');
		expect(paramSummary({ task: 'translate', language: 'de' })).toBe('translated to English, language de');
	});
	it('returns an empty string for missing or odd params', () => {
		expect(paramSummary(null)).toBe('');
		expect(paramSummary('nope')).toBe('');
		expect(paramSummary({ width: 512 })).toBe('');
		expect(paramSummary({ steps: Number.NaN })).toBe('');
	});
});
