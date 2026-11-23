// @vitest-environment jsdom
/**
 * The load gate on the real inspector against the mock backend's engine
 * setups: a model the planner sends to the native engine loads with no
 * llama.cpp install, and one that needs llama-server is still refused.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/svelte';
import { mockInvoke } from '$lib/ipc/mock';
import { refreshActiveEngine, refreshNativeEngine } from '$lib/stores/engine.svelte';
import { ensureFit, invalidateFits } from '$lib/stores/fits.svelte';
import type { InstalledModel, LoadPlan } from '$lib/types';
import ModelInspector from './ModelInspector.svelte';
import { CHECK_FAILED, LLAMA_NEEDED, PLAN_PENDING, defaultCtx, textEngineBlock } from './modelMeta';

const GLM = 'glm-5.3-flash-ud-q2_k_xl';

type Setup = 'both' | 'native-only' | 'llama-only' | 'neither' | 'native-no-capability';

async function show(setup: Setup, pick: (models: InstalledModel[]) => InstalledModel) {
	localStorage.setItem('oh.mock.engines', setup);
	invalidateFits([]);
	await refreshActiveEngine();
	await refreshNativeEngine();
	const model = pick(await mockInvoke<InstalledModel[]>('list_local_models'));
	await ensureFit(model.id, defaultCtx(model));
	const onload = vi.fn();
	const noop = () => {};
	render(ModelInspector, {
		model,
		busy: false,
		onload,
		onunload: noop,
		onoptions: noop,
		onchat: noop,
		onstudio: noop,
		onreveal: noop,
		ondelete: noop,
	});
	const load = (await screen.findByRole('button', { name: 'Load' })) as HTMLButtonElement;
	return { load, onload, model };
}

const glm = (models: InstalledModel[]) => models.find((m) => m.id === GLM)!;
/** A small text model every plan keeps on llama-server. */
const small = (models: InstalledModel[]) =>
	[...models]
		.filter((m) => m.caps.runtime === 'llama-server' && m.id !== GLM)
		.sort((a, b) => a.total_size - b.total_size)[0];

afterEach(() => localStorage.removeItem('oh.mock.engines'));

describe('ModelInspector engine gate', () => {
	it('loads a native-plan model with only the native engine installed', async () => {
		const { load, onload, model } = await show('native-only', glm);
		await waitFor(() => expect(load.disabled).toBe(false));
		load.click();
		expect(onload).toHaveBeenCalledWith(model);
	});

	it('refuses a llama-server model without llama.cpp, even with the native engine', async () => {
		const { load } = await show('native-only', small);
		await waitFor(() => expect(load.disabled).toBe(true));
		expect(load.title).toBe(LLAMA_NEEDED);
	});

	it('loads a llama-server model with only llama.cpp installed', async () => {
		const { load } = await show('llama-only', small);
		await waitFor(() => expect(load.disabled).toBe(false));
	});

	it('shows the planner reason when neither engine can run the model', async () => {
		const { load } = await show('neither', glm);
		await waitFor(() => expect(load.disabled).toBe(true));
		expect(load.title).toContain("native engine isn't installed");
	});

	it('refuses a native plan when the engine lacks SSD streaming', async () => {
		const { load } = await show('native-no-capability', glm);
		await waitFor(() => expect(load.disabled).toBe(true));
		expect(load.title).toContain('ssd_expert_streaming');
	});
});

describe('textEngineBlock', () => {
	const m = { format: 'gguf', caps: { runtime: 'llama-server' } } as InstalledModel;
	const nativeOnly = { llama: false, nativeChecked: true, nativeStreaming: true };
	const llamaOnly = { llama: true, nativeChecked: true, nativeStreaming: false };
	const nativePlan = { plan: { engine_kind: 'ssd_stream', infeasible_reason: 'needs native' } as LoadPlan };

	it('waits for the plan rather than guessing while only the native engine could run it', () => {
		expect(textEngineBlock(m, null, null, nativeOnly)).toMatch(/Checking/);
	});

	it('does not treat an unknown plan as allowed when llama.cpp is installed', () => {
		expect(textEngineBlock(m, null, null, llamaOnly)).toBe(PLAN_PENDING);
	});

	it('refuses a native plan on a llama-only machine with the plan reason', () => {
		expect(textEngineBlock(m, nativePlan, null, llamaOnly)).toBe('needs native');
	});

	it('blocks a failed check with its own error whichever engines are installed', () => {
		const both = { llama: true, nativeChecked: true, nativeStreaming: true };
		const neither = { llama: false, nativeChecked: true, nativeStreaming: false };
		for (const engines of [both, llamaOnly, nativeOnly, neither]) {
			expect(textEngineBlock(m, null, 'boom', engines)).toBe(`${CHECK_FAILED}: boom`);
		}
	});

	it('does not wait forever for a model that never gets an estimate', () => {
		const noFit = { format: 'safetensors', caps: { runtime: 'llama-server' } } as InstalledModel;
		expect(textEngineBlock(noFit, null, null, llamaOnly)).toBeNull();
	});

	it('never blocks a model that runs on another runtime', () => {
		const worker = { caps: { runtime: 'worker' } } as InstalledModel;
		const none = { llama: false, nativeChecked: true, nativeStreaming: false };
		expect(textEngineBlock(worker, null, null, none)).toBeNull();
	});
});

describe('engine status ownership', () => {
	async function storeFor(status: object) {
		localStorage.setItem('oh.mock.status', JSON.stringify(status));
		vi.resetModules();
		const { engineStore, initEngineStore } = await import('$lib/stores/engine.svelte');
		await initEngineStore();
		localStorage.removeItem('oh.mock.status');
		return engineStore;
	}

	it('gives the native engine a load the UI never started, once Ready', async () => {
		const s = await storeFor({ phase: 'ready', model_id: GLM, port: 1, ctx: 8192, launcher: 'ssd_stream' });
		expect(s.nativeOwnsStatus).toBe(true);
		expect(s.runningKind).toBe('ssd_stream');
	});

	it('follows the launcher through loading and failure with llama.cpp installed', async () => {
		expect((await storeFor({ phase: 'loading', model_id: GLM, launcher: 'ssd_stream' })).nativeOwnsStatus).toBe(true);
		expect((await storeFor({ phase: 'failed', error: 'x', launcher: 'ssd_stream' })).nativeOwnsStatus).toBe(true);
		expect((await storeFor({ phase: 'failed', error: 'x', launcher: 'llama_server' })).nativeOwnsStatus).toBe(false);
	});

	it('shows a load failure only on the card of the engine that failed', async () => {
		await storeFor({ phase: 'failed', error: 'native boom', launcher: 'ssd_stream' });
		const { enginesStore } = await import('$lib/stores/engines.svelte');
		const card = (id: string) => enginesStore.engines.find((e) => e.id === id)!;
		expect(card('native').status.error).toBe('native boom');
		expect(card('llama-server').status.error).toBeUndefined();
	});

	it('leaves no load error behind for a later llama.cpp model to inherit', async () => {
		const store = await storeFor({ phase: 'idle' });
		const { loadModel } = await import('$lib/stores/engine.svelte');
		await expect(loadModel('no-such-model')).rejects.toThrow();
		expect(store.error).toBeNull();
	});

	it('leaves an install failure and idle with no launcher to llama.cpp', async () => {
		expect((await storeFor({ phase: 'failed', error: 'x' })).nativeOwnsStatus).toBe(false);
		expect((await storeFor({ phase: 'idle' })).nativeOwnsStatus).toBe(false);
	});
});
