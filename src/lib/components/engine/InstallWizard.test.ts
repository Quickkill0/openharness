// @vitest-environment jsdom
/** The llama.cpp installer only speaks for install failures. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/svelte';
import InstallWizard from './InstallWizard.svelte';

const engine = vi.hoisted(() => ({ status: { phase: 'idle' } as Record<string, unknown> }));

vi.mock('$lib/stores/engine.svelte', () => ({
	engineStore: {
		get status() {
			return engine.status;
		},
		get phase() {
			return engine.status.phase;
		},
		error: null,
	},
	installEngine: vi.fn(),
}));

afterEach(cleanup);

describe('InstallWizard failure ownership', () => {
	it('does not report a native startup failure as an install failure', () => {
		engine.status = { phase: 'failed', error: 'native startup died', launcher: 'ssd_stream' };
		render(InstallWizard, { open: true });
		expect(screen.queryByText(/install failed/i)).toBeNull();
		expect(screen.queryByText(/native startup died/)).toBeNull();
	});

	it('still reports a real install failure', () => {
		engine.status = { phase: 'failed', error: 'download broke' };
		render(InstallWizard, { open: true });
		expect(screen.getByText(/install failed/i)).toBeTruthy();
		expect(screen.getByText(/download broke/)).toBeTruthy();
	});
});
