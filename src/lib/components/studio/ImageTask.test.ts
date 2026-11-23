// @vitest-environment jsdom
/**
 * Component-level remix tests for the Image form: the real ImageTask with its
 * ModelPicker, TaskShell and recent-generations strip, against a controlled
 * catalog. They pin the two review findings that helper tests can't see:
 *
 *  - S1: remix and the picker decide "still installed" from the same list, so
 *    a model installed after the form mounted is accepted by both (the picker
 *    used to keep its mount-time list, reject the model and fall back to
 *    another checkpoint while the remixed SDXL settings stayed).
 *  - S5: when the source model is gone, every model-dependent setting stays
 *    exactly as it was (it used to snap back to the Square/Balanced presets).
 *  - Out-of-order catalog reads: only the picker's newest refresh may write
 *    its list, so an older response landing late can't undo a newer remix.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import type { Asset, InstalledModel } from '$lib/types';

const catalog = vi.hoisted(() => ({ models: [] as unknown[], assets: [] as unknown[] }));

vi.mock('$lib/ipc/commands', async (importOriginal) => {
	const actual = await importOriginal<typeof import('$lib/ipc/commands')>();
	return {
		...actual,
		listLocalModels: vi.fn(async () => structuredClone(catalog.models)),
		listAssets: vi.fn(async () => structuredClone(catalog.assets)),
	};
});

import { listLocalModels } from '$lib/ipc/commands';
import ImageTask from './ImageTask.svelte';

function deferred<T>() {
	let resolve!: (v: T) => void;
	const promise = new Promise<T>((r) => (resolve = r));
	return { promise, resolve };
}

function imageModel(id: string, path: string, arch: string, name: string): InstalledModel {
	return {
		id,
		repo_id: `test/${id}`,
		revision: 'main',
		display_name: name,
		capability: 'image-gen',
		format: 'diffusers-dir',
		task: 'text-to-image',
		local_path: path,
		installed_at: 0,
		primary_file_path: `${path}\\model_index.json`,
		primary_file_size: 1,
		primary_rel_path: 'model_index.json',
		primary_sha256: null,
		file_count: 12,
		total_size: 1,
		pinned: false,
		caps: {
			runtime: 'worker',
			pipeline: 'text_to_image',
			workflows: ['image'],
			speech: null,
			image_arch: arch,
			gguf_diffusion: false,
			companion_arch: null,
			reason: null,
		},
	};
}

const SD15 = imageModel('sd15', 'C:\\models\\sd15', 'sd', 'Stable Diffusion 1.5');
const SDXL = imageModel('sdxl', 'C:\\models\\sdxl-base', 'sdxl', 'SDXL base');

function imageAsset(id: string, prompt: string, params: Record<string, unknown>): Asset {
	return {
		id,
		job_id: null,
		kind: 'image',
		path: `C:\\assets\\${id}.png`,
		task: 'text_to_image',
		prompt,
		model: null,
		params: { prompt, ...params },
		meta: null,
		size: 1,
		created_at: 1,
	};
}

const SDXL_ASSET = imageAsset('a-sdxl', 'castle on a cliff', {
	model: SDXL.local_path,
	arch: 'sdxl',
	width: 1024,
	height: 1024,
	steps: 40,
	guidance: 6,
});
const SD15_CUSTOM_ASSET = imageAsset('a-sd15', 'foggy harbour', {
	model: SD15.local_path,
	arch: 'sd',
	width: 640,
	height: 512,
	steps: 22,
	guidance: 6.5,
});
const MISSING_ASSET = imageAsset('a-gone', 'neon alley', {
	model: 'C:\\models\\flux-dev',
	arch: 'flux',
	width: 1344,
	height: 768,
	steps: 50,
	guidance: 3.5,
});

const modelSelect = () => screen.getByLabelText(/^Model/) as HTMLSelectElement;
const archSelect = () => document.getElementById('img-arch') as HTMLSelectElement;
const widthSelect = () => document.getElementById('img-w') as HTMLSelectElement;
const heightSelect = () => document.getElementById('img-h') as HTMLSelectElement;
const stepsInput = () => screen.getByLabelText('Steps value') as HTMLInputElement;
const checkedPill = (group: string) =>
	screen.getByRole('radiogroup', { name: group }).querySelector('[aria-checked="true"]')?.textContent?.trim() ?? null;

async function mountWith(models: InstalledModel[], assets: Asset[]) {
	catalog.models = models;
	catalog.assets = assets;
	render(ImageTask);
	// Picker resolved its initial selection and the strip has loaded.
	await waitFor(() => expect(modelSelect().value).toBe(models[0].local_path));
	await screen.findByRole('button', { name: `Reuse settings from ${assets[0].prompt}` });
	await fireEvent.click(screen.getByRole('button', { name: /^Advanced/ }));
}

async function reuse(asset: Asset) {
	await fireEvent.click(screen.getByRole('button', { name: `Reuse settings from ${asset.prompt}` }));
}

beforeEach(() => {
	localStorage.clear();
});

describe('ImageTask remix', () => {
	it('selects a model installed after the form mounted, with its own settings (S1)', async () => {
		// Studio opens with only SD 1.5 installed...
		await mountWith([SD15], [SDXL_ASSET]);
		expect(modelSelect().querySelector(`option[value="${CSS.escape(SDXL.local_path!)}"]`)).toBeNull();

		// ...then SDXL finishes installing while the page stays open.
		catalog.models = [SD15, SDXL];
		await reuse(SDXL_ASSET);

		// The picker accepts SDXL instead of falling back to SD 1.5, and the
		// SDXL settings land on the SDXL model.
		await waitFor(() => expect(modelSelect().value).toBe(SDXL.local_path));
		await waitFor(() => expect(archSelect().value).toBe('sdxl'));
		expect(widthSelect().value).toBe('1024');
		expect(heightSelect().value).toBe('1024');
		expect(stepsInput().value).toBe('40');
		expect(screen.queryByRole('status', { name: /no longer installed/i })).toBeNull();
		expect(document.querySelector('.wb-banner.warn')).toBeNull();
	});

	it('leaves SDXL settings off when the remixed model is uninstalled (S1)', async () => {
		// The review repro the other way round: SDXL asset, only SD 1.5 left.
		await mountWith([SD15, SDXL], [SDXL_ASSET]);
		catalog.models = [SD15];
		await reuse(SDXL_ASSET);

		await waitFor(() => expect(document.querySelector('.wb-banner.warn')).not.toBeNull());
		expect(modelSelect().value).toBe(SD15.local_path);
		expect(archSelect().value).toBe(''); // auto-detect, not a stale 'sdxl'
		expect(widthSelect().value).toBe('512'); // SD 1.5 square preset
	});

	it('keeps the current model and all its settings when the source model is gone (S5)', async () => {
		await mountWith([SD15], [SD15_CUSTOM_ASSET, MISSING_ASSET]);

		// Put custom numbers on the current model through an installed remix.
		await reuse(SD15_CUSTOM_ASSET);
		await waitFor(() => expect(widthSelect().value).toBe('640'));
		expect(heightSelect().value).toBe('512');
		expect(stepsInput().value).toBe('22');
		expect(checkedPill('Size')).toBeNull();
		expect(checkedPill('Quality')).toBeNull();

		// Remixing a generation whose model is gone restores its prompt only.
		await reuse(MISSING_ASSET);
		await waitFor(() => expect(document.querySelector('.wb-banner.warn')).not.toBeNull());
		expect((document.getElementById('img-prompt') as HTMLTextAreaElement).value).toBe('neon alley');
		expect(modelSelect().value).toBe(SD15.local_path);
		expect(archSelect().value).toBe('sd');
		expect(widthSelect().value).toBe('640');
		expect(heightSelect().value).toBe('512');
		expect(stepsInput().value).toBe('22');
		expect(checkedPill('Size')).toBeNull();
		expect(checkedPill('Quality')).toBeNull();
	});

	it('ignores an older catalog response that arrives after a newer one', async () => {
		await mountWith([SD15], [SDXL_ASSET, SD15_CUSTOM_ASSET]);
		const older = deferred<InstalledModel[]>();
		const newer = deferred<InstalledModel[]>();
		vi.mocked(listLocalModels)
			.mockImplementationOnce(() => older.promise)
			.mockImplementationOnce(() => newer.promise);

		// Remix A starts while only SD 1.5 is installed; its catalog read stalls.
		await reuse(SD15_CUSTOM_ASSET);
		// SDXL finishes installing, then remix B starts and its read returns first.
		await reuse(SDXL_ASSET);
		newer.resolve([SD15, SDXL]);
		await waitFor(() => expect(modelSelect().value).toBe(SDXL.local_path));
		await waitFor(() => expect(archSelect().value).toBe('sdxl'));

		// A's stale answer (SD 1.5 only) lands last and must not replace the list.
		older.resolve([SD15]);
		await new Promise((r) => setTimeout(r, 0));
		await waitFor(() => expect(modelSelect().value).toBe(SDXL.local_path));
		expect(modelSelect().querySelector(`option[value="${CSS.escape(SDXL.local_path!)}"]`)).not.toBeNull();
		expect(archSelect().value).toBe('sdxl');
		expect(widthSelect().value).toBe('1024');
		expect(stepsInput().value).toBe('40');
		expect(document.querySelector('.wb-banner.warn')).toBeNull();
	});
});
