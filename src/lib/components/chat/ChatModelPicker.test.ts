// @vitest-environment jsdom
/** The picker's rows follow the plan, whichever engines are installed. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { refreshActiveEngine, refreshNativeEngine } from '$lib/stores/engine.svelte';
import { invalidateFits } from '$lib/stores/fits.svelte';
import type { FitEstimate } from '$lib/types';
import ChatModelPicker from './ChatModelPicker.svelte';

const hold = vi.hoisted(() => ({
	release: null as null | (() => void),
	calls: [] as string[],
	fail: null as null | string,
}));

vi.mock('$lib/ipc/commands', async (original) => {
	const actual = await original<typeof import('$lib/ipc/commands')>();
	return {
		...actual,
		estimateModelFit: async (id: string, ctx?: number): Promise<FitEstimate> => {
			hold.calls.push(id);
			if (id.startsWith('glm') && hold.fail) throw new Error(hold.fail);
			if (id.startsWith('glm')) await new Promise<void>((done) => (hold.release = done));
			return actual.estimateModelFit(id, ctx);
		},
	};
});

afterEach(() => localStorage.removeItem('oh.mock.engines'));

describe('ChatModelPicker with llama.cpp installed and no native engine', () => {
	it('holds a GLM row as checking, then refuses it, asking once per model', async () => {
		localStorage.setItem('oh.mock.engines', 'llama-only');
		await refreshActiveEngine();
		await refreshNativeEngine();
		render(ChatModelPicker);
		await fireEvent.click(screen.getByRole('button', { name: 'Choose chat model' }));

		const row = async () =>
			(await screen.findByText(/GLM 5\.3 Flash/)).closest('button') as HTMLButtonElement;
		await waitFor(async () => expect((await row()).textContent).toContain('Checking'));
		expect((await row()).disabled).toBe(true);

		await waitFor(() => expect(hold.release).not.toBeNull());
		hold.release!();
		await waitFor(async () => expect((await row()).textContent).not.toContain('Checking'));
		expect((await row()).disabled).toBe(true);
		expect((await row()).textContent).not.toMatch(/\bLoad\b/);
		expect(new Set(hold.calls).size).toBe(hold.calls.length);
	});

	it('blocks a failed check with its error, and retries from the row and on reopening', async () => {
		localStorage.setItem('oh.mock.engines', 'llama-only');
		await refreshActiveEngine();
		await refreshNativeEngine();
		invalidateFits([]);
		hold.fail = 'probe died';
		render(ChatModelPicker);
		const toggle = screen.getByRole('button', { name: 'Choose chat model' });
		await fireEvent.click(toggle);
		const row = async () =>
			(await screen.findByText(/GLM 5\.3 Flash/)).closest('button') as HTMLButtonElement;
		await waitFor(async () => expect((await row()).textContent).toContain('probe died'));
		expect((await row()).disabled).toBe(true);

		const settle = async () => {
			await waitFor(async () => expect((await row()).textContent).toContain('Checking'));
			await waitFor(() => expect(hold.release).not.toBeNull());
			hold.release!();
			hold.release = null;
			await waitFor(async () => expect((await row()).textContent).not.toContain('Checking'));
			expect((await row()).textContent).not.toContain('probe died');
		};

		hold.fail = null;
		hold.release = null;
		await fireEvent.click(screen.getByRole('button', { name: 'Retry check' }));
		await settle();

		// A cached error doesn't outlive the list being closed.
		hold.fail = 'probe died';
		invalidateFits([]);
		await fireEvent.click(toggle);
		await fireEvent.click(toggle);
		await waitFor(async () => expect((await row()).textContent).toContain('probe died'));
		await fireEvent.click(toggle);
		hold.fail = null;
		await fireEvent.click(toggle);
		await settle();
	});
});
