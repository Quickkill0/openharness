/**
 * Regression tests for the unload guards' activity signal (PR #60 review,
 * round 2). They drive the real chat and Studio stores through the mock IPC
 * backend, so the events arrive the same way they do in the app.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { chatStore, createConversation, initChatStore, sendMessage } from '$lib/stores/chat.svelte';
import { engineStore, initEngineStore } from '$lib/stores/engine.svelte';
import { studio, submitJobAndWait } from '$lib/stores/studio.svelte';
import { unloadLock } from '$lib/stores/chipLocks.svelte';
import type { ResidentInfo } from '$lib/types';

const resident = (id: string, model_id: string | null, kind = 'pipeline'): ResidentInfo => ({
	id,
	kind: kind as ResidentInfo['kind'],
	label: model_id ?? 'Unknown',
	model_id,
	est_mb: 3000,
	actual_mb: null,
});

beforeEach(() => {
	vi.useFakeTimers();
});

afterEach(async () => {
	await vi.runAllTimersAsync();
	vi.useRealTimers();
});

describe('unload guard activity', () => {
	it('keeps the chat model locked after switching away from a conversation that is still replying', async () => {
		await initEngineStore();
		await initChatStore();
		await vi.advanceTimersByTimeAsync(0);
		const status = engineStore.status;
		expect(status.phase).toBe('ready');
		const model = status.phase === 'ready' ? status.model_id : '';
		const text = resident('text', model, 'text_engine');

		const a = await createConversation();
		expect(a).not.toBeNull();
		await sendMessage('Write a long answer');
		await vi.advanceTimersByTimeAsync(100);
		expect(chatStore.streaming).not.toBeNull();
		expect(unloadLock(text)?.kind).toBe('in_use');

		// Selecting another conversation clears what Chat shows, not the work.
		await createConversation();
		expect(chatStore.activeId).not.toBe(a);
		expect(chatStore.streaming).toBeNull();
		expect(chatStore.inflight.get(a!)).toBe(model);
		expect(unloadLock(text)?.kind).toBe('in_use');
		await vi.advanceTimersByTimeAsync(500);
		expect(unloadLock(text)?.kind).toBe('in_use');

		// A's real end event releases the lock even though A isn't selected.
		await vi.runAllTimersAsync();
		expect(chatStore.inflight.has(a!)).toBe(false);
		expect(unloadLock(text)).toBeNull();
	});

	it('keeps the Qwen3-TTS chip locked while chat voice synthesis runs headless', async () => {
		const tts = resident('tts', 'qwen3-tts-12hz-0.6b-customvoice-q8_0', 'tts_engine');
		expect(unloadLock(tts)).toBeNull();

		const done = submitJobAndWait('text_to_speech', {
			engine: 'qwen3-tts',
			text: 'Spoken reply',
			voice: 'serena',
			model: 'C:\\models\\qwen3-tts.gguf',
		});
		// Busy from the moment the request leaves.
		await vi.advanceTimersByTimeAsync(0);
		expect(studio.active.text_to_speech).toBeUndefined();
		expect(studio.busyTasks.has('text_to_speech')).toBe(true);
		expect(unloadLock(tts)?.kind).toBe('in_use');

		// Mid-job (the mock runs 5 stages of 450 ms).
		await vi.advanceTimersByTimeAsync(1200);
		expect(unloadLock(tts)?.kind).toBe('in_use');

		await vi.runAllTimersAsync();
		const result = await done;
		expect(result.outputs.length).toBeGreaterThan(0);
		expect(studio.busyTasks.has('text_to_speech')).toBe(false);
		expect(unloadLock(tts)).toBeNull();
	});

	it('keeps a headless job locked past its waiter timeout until the job really ends', async () => {
		const tts = resident('tts', 'qwen3-tts-12hz-0.6b-customvoice-q8_0', 'tts_engine');
		const done = submitJobAndWait(
			'text_to_speech',
			{ engine: 'qwen3-tts', text: 'Slow reply', voice: 'serena', model: 'C:\\models\\qwen3-tts.gguf' },
			500,
		);
		const settled = done.catch((e: Error) => e);
		await vi.advanceTimersByTimeAsync(600);
		expect(await settled).toBeInstanceOf(Error);
		// The waiter gave up, but the sidecar is still synthesizing.
		expect(unloadLock(tts)?.kind).toBe('in_use');
		await vi.runAllTimersAsync();
		expect(unloadLock(tts)).toBeNull();
	});
});

/**
 * PR #60 review round 3: the backend is the authority. These load fresh
 * modules so the stores start empty, the way a reloaded webview does, while
 * the mock backend keeps its own record of running jobs.
 */
describe('backend authority', () => {
	const ttsResident = resident('tts', 'qwen3-tts-12hz-0.6b-customvoice-q8_0', 'tts_engine');
	const qwenParams = (text: string) => ({
		engine: 'qwen3-tts',
		text,
		voice: 'serena',
		model: 'C:\models\qwen3-tts.gguf',
	});

	it('a reloaded page adopts a job the backend is still running and keeps it locked', async () => {
		vi.resetModules();
		const cmds = await import('$lib/ipc/commands');
		// Submitted before the reload: the backend has it, the new page doesn't.
		const out = await cmds.synthesizeSpeech(qwenParams('Started before the reload'));
		expect(out.outcome).toBe('submitted');

		const fresh = await import('$lib/stores/studio.svelte');
		const locks = await import('$lib/stores/chipLocks.svelte');
		expect(locks.unloadLock(ttsResident)).toBeNull();

		await fresh.initStudio();
		expect(fresh.studio.busyTasks.has('text_to_speech')).toBe(true);
		expect(locks.unloadLock(ttsResident)?.kind).toBe('in_use');
		// Even a caller that skips the UI guard is refused by the backend.
		await expect(cmds.orchestratorUnload('tts')).rejects.toThrow(/generation job is using/);

		await vi.runAllTimersAsync();
		expect(fresh.studio.busyTasks.has('text_to_speech')).toBe(false);
		expect(locks.unloadLock(ttsResident)).toBeNull();
		await expect(cmds.orchestratorUnload('tts')).resolves.toBeUndefined();
	});

	it('a worker exit ends its accepted jobs with an error and unlocks them, leaving sidecar jobs alone', async () => {
		vi.resetModules();
		const cmds = await import('$lib/ipc/commands');
		const fresh = await import('$lib/stores/studio.svelte');
		const locks = await import('$lib/stores/chipLocks.svelte');
		await fresh.initStudio();
		await vi.advanceTimersByTimeAsync(5000); // mock worker auto-install reaches ready
		const whisper = resident('pipeline:transcribe', 'faster-whisper-small');

		const transcript = fresh
			.submitJobAndWait('transcribe', { model: 'C:\models\whisper-small', audio: 'C:\tmp\a.wav' })
			.catch((e: Error) => e);
		const speech = fresh.submitJobAndWait('text_to_speech', qwenParams('Sidecar keeps going'));
		await vi.advanceTimersByTimeAsync(600);
		expect(locks.unloadLock(whisper)?.kind).toBe('in_use');
		expect(locks.unloadLock(ttsResident)?.kind).toBe('in_use');

		await cmds.workerStop();
		await vi.advanceTimersByTimeAsync(0);
		const err = await transcript;
		expect(err).toBeInstanceOf(Error);
		expect((err as Error).message).toBe('worker process exited');
		expect(locks.unloadLock(whisper)).toBeNull();
		// The Qwen3-TTS sidecar isn't the worker: its job and lock carry on.
		expect(locks.unloadLock(ttsResident)?.kind).toBe('in_use');
		const jobs = await cmds.listGenerationJobs();
		expect(jobs.find((j) => j.task === 'transcribe')?.status).toBe('error');

		await vi.runAllTimersAsync();
		expect((await speech).outputs.length).toBeGreaterThan(0);
		expect(locks.unloadLock(ttsResident)).toBeNull();
	});

	it('an adopted lease whose submit then fails ends with an error and unlocks', async () => {
		vi.resetModules();
		const mock = await import('$lib/ipc/mock');
		// The reload lands after the reservation but before the row insert.
		mock.mockGrantUnwrittenLease('unwritten-1', 'text_to_speech');
		const fresh = await import('$lib/stores/studio.svelte');
		const locks = await import('$lib/stores/chipLocks.svelte');
		await fresh.initStudio();
		expect(fresh.studio.busyTasks.has('text_to_speech')).toBe(true);
		expect(locks.unloadLock(ttsResident)?.kind).toBe('in_use');

		// The insert fails: the backend aborts the lease with a terminal error.
		mock.mockAbortLease('unwritten-1', 'database is locked');
		await vi.advanceTimersByTimeAsync(0);
		expect(fresh.studio.busyTasks.has('text_to_speech')).toBe(false);
		expect(locks.unloadLock(ttsResident)).toBeNull();
	});

	it('a synthesis on one talker refuses loading another until it ends', async () => {
		vi.resetModules();
		const cmds = await import('$lib/ipc/commands');
		const talkerA = 'qwen3-tts-1-7b-customvoice-q8_0';
		const talkerB = 'qwen3-tts-1-7b-base-q8_0';
		const out = await cmds.synthesizeSpeech({ ...qwenParams('Hold the talker'), model: talkerA });
		expect(out.outcome).toBe('submitted');

		await expect(cmds.loadModelById(talkerB)).rejects.toThrow(/speech job is using the loaded voice model/);
		// The same talker is reused, not restarted.
		await expect(cmds.loadModelById(talkerA)).resolves.toMatchObject({ outcome: 'loaded' });

		await vi.runAllTimersAsync();
		const swap = cmds.loadModelById(talkerB);
		await vi.advanceTimersByTimeAsync(1000); // the mock's load time
		await expect(swap).resolves.toMatchObject({ outcome: 'loaded' });
	});

	it('a reloaded page adopts an old running job however many newer rows exist', async () => {
		vi.resetModules();
		const mock = await import('$lib/ipc/mock');
		const cmds = await import('$lib/ipc/commands');
		const row = (id: string, task: string, status: 'running' | 'done', at: number) => ({
			id, task, status, params: {}, outputs: [], meta: {}, error: null,
			created_at: at, finished_at: status === 'done' ? at + 5 : null,
		});
		// A long video job, then 250 newer finished image rows: far past any
		// newest-N history window (the old scan read 50, the backend 200).
		mock.mockSeedGenerationJobs([
			row('old-video', 'text_to_video', 'running', 1_000),
			...Array.from({ length: 250 }, (_, i) => row(`img-${i}`, 'text_to_image', 'done', 2_000 + i)),
		]);
		const recent = await cmds.listGenerationJobs(undefined, 50);
		expect(recent.some((j) => j.id === 'old-video')).toBe(false);

		const fresh = await import('$lib/stores/studio.svelte');
		const locks = await import('$lib/stores/chipLocks.svelte');
		await fresh.initStudio();
		expect(fresh.studio.busyTasks.has('text_to_video')).toBe(true);
		expect(locks.unloadLock(resident('pipeline:text_to_video', 'wan2.1-t2v-1.3b'))?.kind).toBe('in_use');
	});
});
