<script lang="ts">
	/**
	 * GroupRow — one repo download group (M9 §9.5) in the downloads panel.
	 *
	 * Sums member-job progress into one bar and exposes group-level
	 * pause/resume/cancel/remove. Members expand into an indented file list.
	 */
	import Icon from '$lib/components/shell/Icon.svelte';
	import {
		pauseDownloadGroup,
		resumeDownloadGroup,
		cancelDownloadGroup,
		deleteDownloadGroup,
	} from '$lib/ipc/commands';
	import { removeJob } from '$lib/stores/downloads.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { formatBytes } from '$lib/format';
	import { groupStatus, humanEta, humanSpeed, jobStatusBadge } from './format';
	import type { DownloadProgress, Job } from '$lib/types';

	interface Props {
		groupId: string;
		jobs: Job[];
		/** Latest progress snapshots keyed by job id. */
		progressMap?: Map<string, DownloadProgress>;
	}

	let { groupId, jobs, progressMap }: Props = $props();

	let expanded = $state(false);

	const repoId = $derived(jobs[0]?.repo_id ?? '');

	function jobDownloaded(j: Job): number {
		return progressMap?.get(j.id)?.downloaded ?? j.downloaded;
	}
	function jobTotal(j: Job): number {
		return progressMap?.get(j.id)?.total ?? j.total_bytes;
	}
	function jobStatus(j: Job): Job['status'] {
		return (progressMap?.get(j.id)?.status as Job['status']) ?? j.status;
	}

	const downloaded = $derived(jobs.reduce((s, j) => s + jobDownloaded(j), 0));
	const total = $derived(jobs.reduce((s, j) => s + jobTotal(j), 0));
	const percent = $derived(total > 0 ? Math.min(100, (downloaded / total) * 100) : 0);
	const doneCount = $derived(jobs.filter((j) => jobStatus(j) === 'done').length);
	const status = $derived(groupStatus(jobs.map(jobStatus)));
	const badge = $derived(jobStatusBadge(status));
	const active = $derived(status === 'running' || status === 'queued' || status === 'paused');

	/** Speed and ETA from whichever member is transferring right now. */
	const live = $derived(
		jobs.map((j) => progressMap?.get(j.id)).find((p) => p && p.status === 'running' && p.bytes_per_sec > 0),
	);
	const etaSecs = $derived(
		live && live.bytes_per_sec > 0 ? Math.max(0, total - downloaded) / live.bytes_per_sec : undefined,
	);

	const failed = $derived(jobs.filter((j) => jobStatus(j) === 'failed' && j.error));

	async function run(action: () => Promise<unknown>, what: string) {
		try {
			await action();
		} catch (e) {
			addToast(`${what} failed: ${e}`, 'error');
		}
	}

	const handlePause = () => run(() => pauseDownloadGroup(groupId), 'Pause');
	const handleResume = () => run(() => resumeDownloadGroup(groupId), 'Resume');
	const handleCancel = () => run(() => cancelDownloadGroup(groupId), 'Cancel');
	const handleRemove = () =>
		run(async () => {
			await deleteDownloadGroup(groupId);
			for (const j of jobs) removeJob(j.id);
		}, 'Remove');
</script>

<div class="wb-dl">
	<div class="wb-dl-top">
		<button
			type="button"
			class="wb-icon-btn expander"
			aria-expanded={expanded}
			aria-label={expanded ? 'Hide files' : 'Show files'}
			title={expanded ? 'Hide files' : 'Show files'}
			onclick={() => (expanded = !expanded)}
		>
			<Icon name="chevron" size={12} class="chev {expanded ? 'open' : ''}" />
		</button>
		<div class="min-w-0 flex-1">
			<div class="wb-dl-name" title={repoId}>{repoId}</div>
			<div class="wb-dl-sub">
				Whole repo, <span class="num">{doneCount}</span> of <span class="num">{jobs.length}</span> files
			</div>
		</div>
		{#if badge}
			<span class="wb-badge {badge.tone}">{badge.label}</span>
		{/if}
		{#if status === 'running' || status === 'queued'}
			<button type="button" class="wb-icon-btn" onclick={handlePause} aria-label="Pause" title="Pause">
				<Icon name="pause" size={13} />
			</button>
		{:else if status === 'paused' || status === 'failed'}
			<button type="button" class="wb-icon-btn" onclick={handleResume} aria-label="Resume" title="Resume">
				<Icon name="play" size={13} />
			</button>
		{/if}
		{#if active || status === 'failed'}
			<button type="button" class="wb-icon-btn" onclick={handleCancel} aria-label="Cancel" title="Cancel">
				<Icon name="x" size={13} />
			</button>
		{:else}
			<button type="button" class="wb-icon-btn" onclick={handleRemove} aria-label="Remove from list" title="Remove from list">
				<Icon name="x" size={13} />
			</button>
		{/if}
	</div>

	{#if active}
		<div
			class="wb-pbar {status === 'paused' ? 'paused' : ''}"
			role="progressbar"
			aria-label="Repo download progress"
			aria-valuemin={0}
			aria-valuemax={100}
			aria-valuenow={Math.round(percent)}
		>
			<i style="width: {percent}%"></i>
		</div>
		<div class="wb-dl-meta">
			<span class="num">{formatBytes(downloaded)} / {formatBytes(total)}</span>
			{#if status === 'paused'}
				<span>Paused</span>
			{:else if live}
				<span class="num">{[humanSpeed(live.bytes_per_sec), humanEta(etaSecs)].filter(Boolean).join(', ')}</span>
			{:else}
				<span class="num">{percent.toFixed(0)}%</span>
			{/if}
		</div>
	{/if}

	{#if expanded}
		<div class="wb-dl-files">
			{#each jobs as j (j.id)}
				{@const s = jobStatus(j)}
				<div>
					<span class="file" title={j.file}>{j.file}</span>
					<span class="num shrink-0">
						{#if s === 'done'}
							<span class="text-success">Done</span>
						{:else if s === 'failed'}
							<span class="text-error">Failed</span>
						{:else if jobTotal(j) > 0}
							{Math.min(100, (jobDownloaded(j) / jobTotal(j)) * 100).toFixed(0)}%
						{:else}
							Waiting
						{/if}
					</span>
				</div>
			{/each}
		</div>
	{/if}

	{#each failed as j (j.id)}
		<p class="wb-dl-err" title={j.error}>{j.file}: {j.error}</p>
	{/each}
</div>

<style>
	.expander {
		margin-left: -6px;
	}
	.expander :global(.chev) {
		transition: transform 0.15s ease;
	}
	.expander :global(.chev.open) {
		transform: rotate(90deg);
	}
	.file {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-family: var(--font-mono);
	}
	@media (prefers-reduced-motion: reduce) {
		.expander :global(.chev) {
			transition: none;
		}
	}
</style>
