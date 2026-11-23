import { describe, expect, it } from 'vitest';
import {
	ctxLabel,
	defaultCtx,
	family,
	hasFitEstimate,
	isLoadable,
	params,
	residentSlot,
	sizeOnDisk,
	studioTask,
} from './modelMeta';
import type { InstalledModel, ModelCaps } from '$lib/types';

type Overrides = { [K in keyof InstalledModel]?: unknown } & { caps?: { [K in keyof ModelCaps]?: unknown } };

function model(over: Overrides = {}): InstalledModel {
	const { caps, ...rest } = over;
	return {
		id: 'm1',
		repo_id: 'org/model',
		display_name: 'Model',
		architecture: 'qwen2',
		format: 'gguf',
		task: null,
		local_path: null,
		params_b: 7.6,
		quantization: 'Q4_K_M',
		context_train: 32768,
		installed_at: 0,
		primary_file_path: 'C:/m.gguf',
		primary_file_size: 4_000,
		file_count: 1,
		total_size: 9_000,
		...rest,
		caps: {
			runtime: 'llama-server',
			pipeline: null,
			workflows: ['chat'],
			reason: null,
			...caps,
		} as ModelCaps,
	} as InstalledModel;
}

describe('params', () => {
	it('formats billions and millions from params_b', () => {
		expect(params(model({ params_b: 7.6 }))).toBe('7.6B');
		expect(params(model({ params_b: 32 }))).toBe('32B');
		expect(params(model({ params_b: 0.082 }))).toBe('82M');
		expect(params(model({ params_b: null }))).toBe('—');
	});
});

describe('family', () => {
	it('title-cases the architecture or shows a dash', () => {
		expect(family(model({ architecture: 'llama' }))).toBe('Llama');
		expect(family(model({ architecture: null }))).toBe('—');
	});
});

describe('sizeOnDisk', () => {
	it('uses the folder total for multi-file models', () => {
		expect(sizeOnDisk(model())).toBe(4_000);
		expect(sizeOnDisk(model({ file_count: 3 }))).toBe(9_000);
	});
});

describe('loadability', () => {
	it('treats embedding-only and unrunnable models as not loadable', () => {
		expect(isLoadable(model())).toBe(true);
		expect(isLoadable(model({ caps: { workflows: ['embeddings'] } }))).toBe(false);
		expect(isLoadable(model({ caps: { runtime: null, workflows: [] } }))).toBe(false);
	});
	it('only estimates fit for llama.cpp GGUF models', () => {
		expect(hasFitEstimate(model())).toBe(true);
		expect(hasFitEstimate(model({ format: 'diffusers-dir', caps: { runtime: 'worker', workflows: ['image'] } }))).toBe(false);
	});
	it('maps Studio workflows to a Studio tab', () => {
		expect(studioTask(model())).toBeNull();
		expect(studioTask(model({ caps: { runtime: 'worker', workflows: ['speech'] } }))).toBe('speech');
	});
});

describe('defaultCtx', () => {
	it('matches the orchestrator: trained context capped at 8K', () => {
		expect(defaultCtx(model({ context_train: 32768 }))).toBe(8192);
		expect(defaultCtx(model({ context_train: 4096 }))).toBe(4096);
		expect(defaultCtx(model({ context_train: 2048 }))).toBe(2048);
	});
	it('falls back to 8K when the trained context is unknown', () => {
		expect(defaultCtx(model({ context_train: null }))).toBe(8192);
		expect(defaultCtx(model({ context_train: 0 }))).toBe(8192);
	});
	it('labels context lengths in K', () => {
		expect(ctxLabel(8192)).toBe('8K');
		expect(ctxLabel(2048)).toBe('2K');
		expect(ctxLabel(6000)).toBe('5.9K');
		expect(ctxLabel(512)).toBe('512');
	});
});

describe('residentSlot', () => {
	it('names the orchestrator slot a load fills', () => {
		expect(residentSlot(model())).toBe('text');
		expect(residentSlot(model({ caps: { runtime: 'qwen-tts', workflows: ['speech'] } }))).toBe('tts');
		expect(residentSlot(model({ caps: { runtime: 'worker', pipeline: 'text_to_image', workflows: ['image'] } }))).toBe(
			'pipeline:text_to_image',
		);
		expect(residentSlot(model({ caps: { runtime: null, workflows: [] } }))).toBeNull();
	});
});
