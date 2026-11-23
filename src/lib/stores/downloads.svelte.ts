/**
 * Svelte 5 rune-based download store.
 *
 * Holds all known download jobs keyed by UUID. Updated by the DownloadDrawer
 * via the progress channel and initial list_downloads call.
 */
import type { DownloadProgress, Job, JobStatus } from '$lib/types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _jobs = $state<Map<string, Job>>(new Map());

export const downloadStore = {
	get jobs() {
		return _jobs;
	},
	get activeCount() {
		let n = 0;
		for (const j of _jobs.values()) {
			if (j.status === 'running' || j.status === 'queued') n++;
		}
		return n;
	},
};

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export function addJob(job: Job): void {
	const next = new Map(_jobs);
	next.set(job.id, job);
	_jobs = next;
}

export function updateJob(id: string, patch: Partial<Job>): void {
	const existing = _jobs.get(id);
	if (!existing) return;
	const next = new Map(_jobs);
	next.set(id, { ...existing, ...patch });
	_jobs = next;
}

export function removeJob(id: string): void {
	const next = new Map(_jobs);
	next.delete(id);
	_jobs = next;
}

export function clear(): void {
	_jobs = new Map();
}

/**
 * Apply a progress event. If the job doesn't exist in the store yet,
 * it is not automatically created — call addJob first via listDownloads.
 */
export function applyProgress(progress: DownloadProgress): void {
	const existing = _jobs.get(progress.id);
	if (!existing) return;

	const next = new Map(_jobs);
	next.set(progress.id, {
		...existing,
		downloaded: progress.downloaded,
		total_bytes: progress.total,
		status: progress.status as JobStatus,
		error: progress.error ?? undefined,
	});
	_jobs = next;
}

/**
 * Seed the store from the result of listDownloads().
 */
export function seedFromList(jobs: Job[]): void {
	const next = new Map<string, Job>();
	for (const j of jobs) {
		next.set(j.id, j);
	}
	_jobs = next;
}
