// @vitest-environment jsdom
/** The dialog's Engine select against the mock planner. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { mockInvoke } from '$lib/ipc/mock';
import { refreshActiveEngine, refreshNativeEngine } from '$lib/stores/engine.svelte';
import type { InstalledModel, LoadPlan, PlanOverrides } from '$lib/types';
import ModelLoadDialog from './ModelLoadDialog.svelte';

const GLM = 'glm-5.3-flash-ud-q2_k_xl';

const spies = vi.hoisted(() => ({
	planHook: null as null | ((overrides: PlanOverrides | undefined) => Promise<LoadPlan> | null),
	loads: [] as Record<string, unknown>[],
	estimates: 0,
}));

vi.mock('$lib/ipc/commands', async (original) => {
	const actual = await original<typeof import('$lib/ipc/commands')>();
	return {
		...actual,
		planLoad: (id: string, overrides?: PlanOverrides) =>
			spies.planHook?.(overrides) ?? actual.planLoad(id, overrides),
		estimateModelFit: (id: string, ctx?: number) => {
			spies.estimates++;
			return actual.estimateModelFit(id, ctx);
		},
		orchestratorLoadModel: (req: Parameters<typeof actual.orchestratorLoadModel>[0]) => {
			spies.loads.push(req as unknown as Record<string, unknown>);
			return actual.orchestratorLoadModel(req);
		},
	};
});

async function open(setup: string, id = GLM) {
	localStorage.setItem('oh.mock.engines', setup);
	await refreshActiveEngine();
	await refreshNativeEngine();
	const models = await mockInvoke<InstalledModel[]>('list_local_models');
	render(ModelLoadDialog, { open: true, model: models.find((m) => m.id === id)! });
	const select = (await screen.findByLabelText('Engine')) as HTMLSelectElement;
	const option = (value: string) =>
		[...select.options].find((o) => o.value === value) as HTMLOptionElement;
	return { select, option };
}

afterEach(() => {
	localStorage.removeItem('oh.mock.engines');
	spies.planHook = null;
	spies.loads.length = 0;
	spies.estimates = 0;
	vi.useRealTimers();
});

const loadButton = () => screen.getByRole('button', { name: 'Load model' }) as HTMLButtonElement;
const enabled = () => waitFor(() => expect(loadButton().disabled).toBe(false));
const contextSlider = () => screen.getByRole('slider') as HTMLInputElement;
async function setContext(value: number) {
	const slider = contextSlider();
	slider.value = String(value);
	await fireEvent.input(slider);
}

function deferred() {
	let resolve!: (p: LoadPlan) => void;
	let reject!: (e: unknown) => void;
	const promise = new Promise<LoadPlan>((ok, no) => {
		resolve = ok;
		reject = no;
	});
	return { promise, resolve, reject };
}

describe('ModelLoadDialog admission', () => {
	it('drops the old plan while new settings are checked, then shows their refusal', async () => {
		await open('both');
		await enabled();
		const feasible = await mockInvoke<LoadPlan>('plan_load', { modelId: GLM, overrides: {} });
		const pending = deferred();
		spies.planHook = (o) => (o?.ctx === 16384 ? pending.promise : null);

		await setContext(16384);
		await waitFor(() => expect(screen.getByTestId('load-pending')).toBeTruthy());
		expect(loadButton().disabled).toBe(true);
		expect(screen.queryByTestId('load-plan')).toBeNull();

		const reason = 'Context 16K needs more memory than this machine has free.';
		pending.resolve({ ...feasible, feasible: false, evict_alternative: null, infeasible_reason: reason });
		await waitFor(() => expect(screen.getByTestId('load-plan').textContent).toContain(reason));
		expect(loadButton().disabled).toBe(true);
	});

	it('ignores an answer for settings that have since changed', async () => {
		await open('both');
		await enabled();
		const feasible = await mockInvoke<LoadPlan>('plan_load', { modelId: GLM, overrides: {} });
		const slow = deferred();
		spies.planHook = (o) => (o?.ctx === 16384 ? slow.promise : null);
		await setContext(16384);
		await setContext(12288);
		await enabled();

		slow.resolve({ ...feasible, feasible: false, evict_alternative: null, infeasible_reason: 'obsolete' });
		await new Promise((r) => setTimeout(r, 20));
		expect(screen.getByTestId('load-plan').textContent).not.toContain('obsolete');
		expect(loadButton().disabled).toBe(false);
	});

	it('recovers when the settings return after a failed check', async () => {
		await open('both');
		await enabled();
		const start = Number(contextSlider().value);
		spies.planHook = (o) => (o?.ctx === 16384 ? Promise.reject(new Error('no')) : null);
		await setContext(16384);
		await enabled(); // the fallback estimate answered, so the check isn't stuck
		await setContext(start);
		await enabled();
		expect(screen.getByTestId('load-plan').textContent).toContain('Native (SSD streaming)');
	});

	it('submits the engine and the context shown', async () => {
		await open('both');
		await enabled();
		await setContext(12288);
		await waitFor(() => expect(screen.getByTestId('load-plan')).toBeTruthy());
		await enabled();
		loadButton().click();
		await waitFor(() => expect(spies.loads).toHaveLength(1));
		expect(spies.loads[0]).toMatchObject({ model_id: GLM, ctx: 12288 });
		expect(spies.loads[0].ngl).toBeUndefined();
	});

	it('submits the picked engine as a planner override', async () => {
		const { select } = await open('both');
		await enabled();
		select.value = 'ssd_stream';
		await fireEvent.change(select);
		await enabled();
		loadButton().click();
		await waitFor(() => expect(spies.loads).toHaveLength(1));
		expect(spies.loads[0].overrides).toEqual({ engine: 'ssd_stream' });
	});
});

describe('ModelLoadDialog memory limits', () => {
	const limit = (name: string) => screen.getByLabelText(name) as HTMLInputElement;
	async function type(input: HTMLInputElement, value: string) {
		input.value = value;
		await fireEvent.input(input);
	}
	const checked = () =>
		waitFor(() => expect(loadButton().disabled).toBe(false), { timeout: 3000 });

	it('re-plans with the typed limits and submits them as planner overrides', async () => {
		await open('both');
		await enabled();
		const asked: (PlanOverrides | undefined)[] = [];
		spies.planHook = (overrides) => {
			asked.push(overrides);
			return null;
		};
		await type(limit('System memory limit (GB)'), '20');
		// Load waits for the planner's answer about the limit just typed.
		expect(loadButton().disabled).toBe(true);
		await checked();
		expect(asked.some((o) => o?.ram_limit_mib === 19073)).toBe(true);
		await type(limit('Graphics memory limit (GB)'), '12.5');
		expect(loadButton().disabled).toBe(true);
		await checked();
		loadButton().click();
		await waitFor(() => expect(spies.loads).toHaveLength(1));
		expect(spies.loads[0].overrides).toEqual({ ram_limit_mib: 19073, vram_limit_mib: 11921 });
	});

	it('locks the limits and loads by layer count once GPU layers is typed', async () => {
		await open('both', 'llama-3.1-8b-instruct-q5_k_m');
		await enabled();
		await type(limit('Graphics memory limit (GB)'), '4');
		await checked();
		// A number input binds a number, which used to throw and leave the dialog planned.
		await type(limit('GPU layers (-ngl)'), '12');
		await waitFor(() => expect(limit('System memory limit (GB)').disabled).toBe(true));
		expect(limit('Graphics memory limit (GB)').disabled).toBe(true);
		expect(screen.queryByTestId('load-plan')).toBeNull();
		await type(limit('Batch size'), '256');
		loadButton().click();
		await waitFor(() => expect(spies.loads).toHaveLength(1));
		expect(spies.loads[0]).toMatchObject({ ngl: 12, batch_size: 256 });
		expect(spies.loads[0].overrides).toBeUndefined();
	});

	it('shows a limit the planner refuses and clears it when the field is emptied', async () => {
		await open('both');
		await enabled();
		await type(limit('System memory limit (GB)'), '4');
		await waitFor(
			() => expect(screen.getByTestId('load-plan').textContent).toContain('system memory limit'),
			{ timeout: 3000 },
		);
		expect(loadButton().disabled).toBe(true);
		await type(limit('System memory limit (GB)'), '');
		await checked();
		loadButton().click();
		await waitFor(() => expect(spies.loads).toHaveLength(1));
		expect(spies.loads[0].overrides).toBeUndefined();
	});
});

describe('ModelLoadDialog engine select', () => {
	it('re-plans when the engine changes and shows the refusal', async () => {
		const { select } = await open('both');
		const plan = () => screen.getByTestId('load-plan').textContent ?? '';
		await waitFor(() => expect(plan()).toContain('Native (SSD streaming)'));
		expect(plan()).toContain('prompt speed unknown');

		select.value = 'llama_server';
		await fireEvent.change(select);
		await waitFor(() => expect(plan()).toContain('overcommits RAM'));
		expect((screen.getByRole('button', { name: 'Load model' }) as HTMLButtonElement).disabled).toBe(true);
	});

	it('does not offer Native when it is not installed', async () => {
		const { option } = await open('llama-only');
		expect(option('ssd_stream').disabled).toBe(true);
		expect(option('ssd_stream').label).toBe('Native (not installed)');
		expect(option('llama_server').disabled).toBe(false);
	});

	it('does not offer Native without SSD streaming', async () => {
		const { option } = await open('native-no-capability');
		expect(option('ssd_stream').disabled).toBe(true);
		expect(option('ssd_stream').label).toBe('Native (no SSD streaming)');
	});

	it('lets a native-only machine load the model and refuses llama.cpp', async () => {
		const { option } = await open('native-only');
		expect(option('llama_server').disabled).toBe(true);
		await waitFor(() =>
			expect((screen.getByRole('button', { name: 'Load model' }) as HTMLButtonElement).disabled).toBe(false),
		);
	});

	it('does not accept an estimate made without the chosen engine, and Retry asks the same thing', async () => {
		const { select } = await open('both');
		await enabled();
		select.value = 'ssd_stream';
		await fireEvent.change(select);
		await enabled();

		const before = spies.estimates;
		spies.planHook = () => Promise.reject(new Error('planner hiccup'));
		await setContext(4096);
		await waitFor(() => expect(screen.getByTestId('load-block').textContent).toContain('planner hiccup'));
		expect(loadButton().disabled).toBe(true);
		expect(spies.estimates).toBe(before);

		const asked: (PlanOverrides | undefined)[] = [];
		spies.planHook = (overrides) => {
			asked.push(overrides);
			return null;
		};
		await fireEvent.click(screen.getByTestId('load-retry'));
		await enabled();
		expect(asked).toEqual([{ engine: 'ssd_stream', ctx: 4096 }]);
	});

	it('keeps a context the user chose when the opening plan answers late', async () => {
		const opening = deferred();
		spies.planHook = () => opening.promise;
		await open('both');
		spies.planHook = null;
		await setContext(4096);
		const plan = await mockInvoke<LoadPlan>('plan_load', { modelId: GLM });
		opening.resolve({ ...plan, ctx: 2048 });
		await enabled();
		expect(contextSlider().value).toBe('4096');
	});

	it('ignores the opening plan once it has timed out', async () => {
		vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
		const opening = deferred();
		spies.planHook = () => opening.promise;
		await open('both');
		spies.planHook = null;
		const start = contextSlider().value;
		await vi.advanceTimersByTimeAsync(30_000);
		const plan = await mockInvoke<LoadPlan>('plan_load', { modelId: GLM });
		opening.resolve({ ...plan, ctx: 2048 });
		await vi.advanceTimersByTimeAsync(0);
		expect(contextSlider().value).toBe(start);
	});
});
