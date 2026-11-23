/**
 * Display helpers for the downloads panel. Pure functions so the wording and
 * status rules are unit-tested rather than living inside the row components.
 */
import type { Job } from '$lib/types';
import { formatBytes } from '$lib/format';

/** Bytes per second → "41 MB/s"; empty below 1 B/s (stalled or unknown). */
export function humanSpeed(bps: number | null | undefined): string {
	if (bps == null || !(bps >= 1)) return '';
	return `${formatBytes(bps)}/s`;
}

/** Seconds remaining → "40 s left" / "3 min left" / "1 h 5 min left"; empty when unknown. */
export function humanEta(secs: number | null | undefined): string {
	if (secs == null || !Number.isFinite(secs) || secs < 0) return '';
	if (secs < 60) return `${Math.max(1, Math.round(secs))} s left`;
	const mins = Math.ceil(secs / 60);
	if (mins < 60) return `${mins} min left`;
	const h = Math.floor(mins / 60);
	const m = mins % 60;
	return m ? `${h} h ${m} min left` : `${h} h left`;
}

/**
 * Files still to come: everything not finished or cancelled. Paused and
 * failed files count, since both resume from the drawer (the downloader
 * resumes any paused or failed job from its saved chunks). The dock's
 * activity indicator uses `downloadStore.activeCount`, running and queued only.
 */
export function unfinishedCount(jobs: Iterable<Pick<Job, 'status'>>): number {
	let n = 0;
	for (const j of jobs) {
		if (j.status === 'queued' || j.status === 'running' || j.status === 'paused' || j.status === 'failed') n++;
	}
	return n;
}

export interface StatusBadge {
	label: string;
	tone: '' | 'ok' | 'warn' | 'err' | 'teal';
}

/**
 * Badge for a job or group status. Running jobs show no badge: the progress
 * bar and speed already say it, so the row stays quiet.
 */
export function jobStatusBadge(status: Job['status']): StatusBadge | null {
	switch (status) {
		case 'done':
			return { label: 'Done', tone: 'ok' };
		case 'failed':
			return { label: 'Failed', tone: 'err' };
		case 'paused':
			return { label: 'Paused', tone: 'warn' };
		case 'queued':
			return { label: 'Queued', tone: '' };
		case 'cancelled':
			return { label: 'Cancelled', tone: '' };
		default:
			return null;
	}
}

/** Group status from member statuses: failed > running > paused > queued > cancelled > done. */
export function groupStatus(statuses: Job['status'][]): Job['status'] {
	if (statuses.some((s) => s === 'failed')) return 'failed';
	if (statuses.some((s) => s === 'running')) return 'running';
	if (statuses.some((s) => s === 'paused')) return 'paused';
	if (statuses.some((s) => s === 'queued')) return 'queued';
	if (statuses.some((s) => s === 'cancelled')) return 'cancelled';
	return 'done';
}
