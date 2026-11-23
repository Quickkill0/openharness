/**
 * Svelte 5 rune-based Studio store (M10).
 *
 * Tracks the active generation job per task page, the live progress stream,
 * worker status, and the eviction prompt. One generation-progress channel is
 * shared by all task pages; results land here so switching tabs mid-job
 * doesn't lose anything.
 */
import { subscribeGenerationProgress } from '$lib/ipc/channels';
import {
	cancelGenerationJob,
	generateImage,
	generateVideo,
	activeGenerationJobs,
	type ActiveGenerationJob,
	synthesizeSpeech,
	transcribeAudio,
} from '$lib/ipc/commands';
import { initWorkerStore, workerStore } from '$lib/stores/worker.svelte';
import { settleHeadlessJob, withJob, withoutJob, type HeadlessJobs } from '$lib/activity';
import type {
	Asset,
	GenerationEvent,
	GenerationTask,
	ResidentInfo,
	SubmitOutcome,
} from '$lib/types';

export interface ActiveJob {
	jobId: string;
	task: GenerationTask;
	pct: number;
	stage: string;
	step: number | null;
	totalSteps: number | null;
}

export interface TaskResult {
	jobId: string;
	outputs: string[];
	meta: unknown;
	/** Params the job was submitted with (for the result caption). */
	params: Record<string, unknown>;
	finishedAt: number;
}

/** Pending VRAM-eviction prompt for a blocked submit (or blocked model load). */
export interface EvictionPrompt {
	task: GenerationTask | 'load_model';
	params: Record<string, unknown>;
	requiredMb: number;
	freeMb: number;
	candidates: ResidentInfo[];
	/**
	 * Headless jobs (chat voice mode) carry their own retry/cancel hooks so the
	 * waiter promise resolves through the dialog. Task-page submits leave these
	 * unset and go back through `submitJob`.
	 */
	resubmit?: (force: boolean) => void;
	cancel?: () => void;
}

/** One active job per task — a second submit on the same page is blocked by the UI. */
let _active = $state<Partial<Record<GenerationTask, ActiveJob>>>({});
let _results = $state<Partial<Record<GenerationTask, TaskResult>>>({});
let _errors = $state<Partial<Record<GenerationTask, string>>>({});
let _eviction = $state<EvictionPrompt | null>(null);
/** A library asset waiting to be restored into its task page's form. */
let _remix = $state<{ task: GenerationTask; asset: Asset } | null>(null);
/** Params snapshot per running job so results can show what produced them. */
const jobParams = new Map<string, { task: GenerationTask; params: Record<string, unknown> }>();
/**
 * One-shot waiters for headless jobs (chat voice mode, M11). These jobs do NOT
 * touch the task pages' active/results state — they resolve a promise instead.
 * The generation channel is single-subscriber on the Rust side, so everything
 * funnels through this store's applyEvent.
 */
const jobWaiters = new Map<
	string,
	{ task: GenerationTask; resolve: (r: TaskResult) => void; reject: (e: Error) => void }
>();
/**
 * Headless jobs still running, by job id (or a `pending:` token while the
 * submit is out). Unlike `jobWaiters` an entry outlives a waiter timeout and
 * only clears on the job's real `job_done`/`job_error`, so the unload guards
 * keep the model locked for as long as it is actually working.
 */
let _headless = $state.raw<HeadlessJobs>(new Map());
let pendingCounter = 0;
/**
 * Job ids whose terminal event arrived before their submit call returned.
 * Without this an entry registered after its own `job_done` would never
 * clear. Bounded; only the most recent ids matter.
 */
const settledEarly = new Set<string>();
const SETTLED_EARLY_MAX = 64;

function rememberSettled(jobId: string): void {
	settledEarly.add(jobId);
	if (settledEarly.size > SETTLED_EARLY_MAX) {
		const oldest = settledEarly.values().next().value;
		if (oldest !== undefined) settledEarly.delete(oldest);
	}
}

export const studio = {
	get worker() {
		return workerStore.status;
	},
	get active() {
		return _active;
	},
	get results() {
		return _results;
	},
	get errors() {
		return _errors;
	},
	get eviction() {
		return _eviction;
	},
	get remix() {
		return _remix;
	},
	/**
	 * Every task with a job in flight: Studio task pages plus headless jobs
	 * (chat voice transcription and spoken replies). Read by the unload guards.
	 */
	get busyTasks(): ReadonlySet<GenerationTask> {
		return new Set<GenerationTask>([
			...(Object.keys(_active) as GenerationTask[]),
			..._headless.values(),
		]);
	},
};

const SUBMITTERS: Record<
	GenerationTask,
	(params: Record<string, unknown>, force?: boolean) => Promise<SubmitOutcome>
> = {
	text_to_image: generateImage,
	text_to_video: generateVideo,
	text_to_speech: synthesizeSpeech,
	transcribe: transcribeAudio,
};

let wired = false;

/**
 * Idempotent: wire the worker-status + generation-progress channels, then
 * adopt the jobs the backend is already running. Called from the app shell so
 * the unload guards are right on every page, including after a reload.
 */
export async function initStudio(): Promise<void> {
	if (wired) return;
	wired = true;
	await initWorkerStore();
	await subscribeGenerationProgress(applyEvent);
	await adoptRunningJobs();
}

/**
 * A reloaded webview starts with no record of jobs a previous page submitted,
 * and the progress channel sends no snapshot. Read the backend's active-job
 * snapshot (subscribed first, so a terminal event can't fall in the gap) and
 * track every job this page doesn't already own until its real terminal
 * event. The snapshot has no history cutoff, and the backend heals crash
 * orphans before serving it, so every entry it returns does end with one.
 */
async function adoptRunningJobs(): Promise<void> {
	let jobs: ActiveGenerationJob[];
	try {
		jobs = await activeGenerationJobs();
	} catch {
		return; // The backend's own unload guard still protects these jobs.
	}
	const owned = new Set(Object.values(_active).map((a) => a?.jobId));
	let next = _headless;
	for (const job of jobs) {
		if (owned.has(job.id) || settledEarly.has(job.id)) continue;
		const task = rowTask(job.task);
		if (task) next = withJob(next, job.id, task);
	}
	_headless = next;
}

/** Job rows store the routed task; VibeVoice clone rows count as speech. */
function rowTask(task: string): GenerationTask | null {
	if (task === 'vibevoice') return 'text_to_speech';
	return task in SUBMITTERS ? (task as GenerationTask) : null;
}

function applyEvent(ev: GenerationEvent): void {
	if (ev.event === 'job_done' || ev.event === 'job_error') {
		if (_headless.has(ev.job_id)) _headless = settleHeadlessJob(_headless, ev);
		else rememberSettled(ev.job_id);
	}
	// Headless waiter jobs (voice mode) resolve their promise and stop here.
	const waiter = 'job_id' in ev ? jobWaiters.get(ev.job_id) : undefined;
	if (waiter) {
		if (ev.event === 'job_done') {
			jobWaiters.delete(ev.job_id);
			waiter.resolve({
				jobId: ev.job_id,
				outputs: ev.outputs,
				meta: ev.meta,
				params: {},
				finishedAt: Math.floor(Date.now() / 1000),
			});
		} else if (ev.event === 'job_error') {
			jobWaiters.delete(ev.job_id);
			waiter.reject(new Error(ev.error));
		}
		return;
	}
	if (ev.event === 'job_progress') {
		const known = jobParams.get(ev.job_id);
		if (!known) return;
		_active = {
			..._active,
			[known.task]: {
				jobId: ev.job_id,
				task: known.task,
				pct: ev.pct,
				stage: ev.stage,
				step: ev.step ?? null,
				totalSteps: ev.total_steps ?? null,
			},
		};
	} else if (ev.event === 'job_done') {
		const known = jobParams.get(ev.job_id);
		if (!known) return;
		jobParams.delete(ev.job_id);
		const next = { ..._active };
		delete next[known.task];
		_active = next;
		_results = {
			..._results,
			[known.task]: {
				jobId: ev.job_id,
				outputs: ev.outputs,
				meta: ev.meta,
				params: known.params,
				finishedAt: Math.floor(Date.now() / 1000),
			},
		};
	} else if (ev.event === 'job_error') {
		const known = jobParams.get(ev.job_id);
		if (!known) return;
		jobParams.delete(ev.job_id);
		const next = { ..._active };
		delete next[known.task];
		_active = next;
		if (ev.error !== 'cancelled') {
			_errors = { ..._errors, [known.task]: ev.error };
		}
	}
}

/**
 * Submit a job for a task page. Returns true when the job started. A
 * needs_eviction outcome opens the shared eviction prompt instead; errors
 * land in `studio.errors[task]`.
 */
export async function submitJob(
	task: GenerationTask,
	params: Record<string, unknown>,
	force?: boolean,
): Promise<boolean> {
	const cleared = { ..._errors };
	delete cleared[task];
	_errors = cleared;
	try {
		const outcome = await SUBMITTERS[task](params, force);
		if (outcome.outcome === 'needs_eviction') {
			setEviction({
				task,
				params,
				requiredMb: outcome.required_mb,
				freeMb: outcome.free_mb,
				candidates: outcome.candidates,
			});
			return false;
		}
		jobParams.set(outcome.job_id, { task, params });
		_active = {
			..._active,
			[task]: { jobId: outcome.job_id, task, pct: 0, stage: 'submitting', step: null, totalSteps: null },
		};
		return true;
	} catch (e: unknown) {
		_errors = { ..._errors, [task]: String(e) };
		return false;
	}
}

/**
 * Submit a job and await its result without touching the task pages' state
 * (chat voice mode, M11). Initializes the progress channel if needed. A
 * needs_eviction outcome opens the shared eviction dialog; the returned
 * promise settles when the user retries/forces (job result) or cancels
 * (rejection).
 */
export async function submitJobAndWait(
	task: GenerationTask,
	params: Record<string, unknown>,
	timeoutMs = 5 * 60_000,
): Promise<TaskResult> {
	await initStudio();
	const attempt = async (force: boolean): Promise<TaskResult> => {
		// Count the job as busy from the moment the request leaves, not only
		// once its id comes back: the backend may already be using the model.
		const token = `pending:${++pendingCounter}`;
		_headless = withJob(_headless, token, task);
		let outcome: SubmitOutcome;
		try {
			outcome = await SUBMITTERS[task](params, force);
		} finally {
			_headless = withoutJob(_headless, token);
		}
		if (outcome.outcome === 'submitted' && !settledEarly.delete(outcome.job_id)) {
			_headless = withJob(_headless, outcome.job_id, task);
		}
		if (outcome.outcome === 'needs_eviction') {
			return new Promise<TaskResult>((resolve, reject) => {
				setEviction({
					task,
					params,
					requiredMb: outcome.required_mb,
					freeMb: outcome.free_mb,
					candidates: outcome.candidates,
					resubmit: (f) => void attempt(f).then(resolve, reject),
					cancel: () =>
						reject(
							new Error(
								`Not enough free VRAM (${outcome.required_mb} MB needed, ${outcome.free_mb} MB free).`,
							),
						),
				});
			});
		}
		const jobId = outcome.job_id;
		return new Promise<TaskResult>((resolve, reject) => {
			const timer = setTimeout(() => {
				jobWaiters.delete(jobId);
				reject(new Error(`${task} job timed out`));
			}, timeoutMs);
			jobWaiters.set(jobId, {
				task,
				resolve: (r) => {
					clearTimeout(timer);
					resolve(r);
				},
				reject: (e) => {
					clearTimeout(timer);
					reject(e);
				},
			});
		});
	};
	return attempt(false);
}

/**
 * Replace the single eviction-prompt slot. A displaced prompt's caller is
 * still parked on its hooks (pending loadModel promise / voice-mode waiter),
 * so cancel it — otherwise the promise never settles and per-model loading
 * flags leak forever.
 */
function setEviction(prompt: EvictionPrompt): void {
	const prev = _eviction;
	_eviction = prompt;
	prev?.cancel?.();
}

/**
 * Open the shared eviction dialog for a non-studio flow (model loads, M16 W1).
 * Callers must provide `resubmit`/`cancel` hooks — there is no submitJob
 * fallback for these prompts.
 */
export function requestEviction(prompt: EvictionPrompt): void {
	setEviction(prompt);
}

/** Re-submit the eviction prompt's job with `force: true`. */
export async function forceSubmit(): Promise<void> {
	const p = _eviction;
	if (!p) return;
	_eviction = null;
	if (p.resubmit) {
		p.resubmit(true);
		return;
	}
	if (p.task !== 'load_model') await submitJob(p.task, p.params, true);
}

/** Retry the eviction prompt's job after the user unloaded a resident. */
export async function retrySubmit(): Promise<void> {
	const p = _eviction;
	if (!p) return;
	_eviction = null;
	if (p.resubmit) {
		p.resubmit(false);
		return;
	}
	if (p.task !== 'load_model') await submitJob(p.task, p.params, false);
}

export function dismissEviction(): void {
	const p = _eviction;
	_eviction = null;
	p?.cancel?.();
}

export async function cancelJob(task: GenerationTask): Promise<void> {
	const job = _active[task];
	if (!job) return;
	try {
		await cancelGenerationJob(job.jobId);
	} catch (e: unknown) {
		_errors = { ..._errors, [task]: String(e) };
	}
}

/** Ask a task page to restore a library asset's prompt and settings. */
export function requestRemix(task: GenerationTask, asset: Asset): void {
	_remix = { task, asset };
}

/** Claim the pending remix for a task (null when none is waiting for it). */
export function takeRemix(task: GenerationTask): Asset | null {
	if (!_remix || _remix.task !== task) return null;
	const asset = _remix.asset;
	_remix = null;
	return asset;
}

export function clearError(task: GenerationTask): void {
	const next = { ..._errors };
	delete next[task];
	_errors = next;
}
