<script lang="ts">
	/** One single-file download in the downloads panel (Workbench ruled row). */
	import Icon from '$lib/components/shell/Icon.svelte';
	import { pauseDownload, resumeDownload, cancelDownload, deleteDownload } from '$lib/ipc/commands';
	import { removeJob } from '$lib/stores/downloads.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { formatBytes } from '$lib/format';
	import { humanEta, humanSpeed, jobStatusBadge } from './format';
	import type { DownloadProgress, Job } from '$lib/types';

	interface Props {
		job: Job;
		/** Latest progress snapshot — may lag behind the job store briefly. */
		progress?: DownloadProgress;
	}

	let { job, progress }: Props = $props();

	const filename = $derived(job.file.split('/').pop() ?? job.file);

	const downloaded = $derived(progress?.downloaded ?? job.downloaded);
	const total = $derived(progress?.total ?? job.total_bytes);
	const percent = $derived(total > 0 ? Math.min(100, (downloaded / total) * 100) : 0);

	const bps = $derived(progress?.bytes_per_sec ?? 0);
	const eta = $derived(progress?.eta_secs);
	const phase = $derived(progress?.phase ?? 'downloading');
	const phaseLabel = $derived(
		phase === 'verifying' ? 'Checking the file' : phase === 'finalizing' ? 'Finishing up' : null,
	);

	const badge = $derived(jobStatusBadge(job.status));
	const showBar = $derived(
		job.status === 'running' || job.status === 'paused' || (job.status === 'queued' && total > 0),
	);

	async function run(action: () => Promise<unknown>, what: string) {
		try {
			await action();
		} catch (e) {
			addToast(`${what} failed: ${e}`, 'error');
		}
	}

	const handlePause = () => run(() => pauseDownload(job.id), 'Pause');
	const handleResume = () => run(() => resumeDownload(job.id), 'Resume');
	const handleCancel = () => run(() => cancelDownload(job.id), 'Cancel');
	const handleDelete = () =>
		run(async () => {
			await deleteDownload(job.id);
			removeJob(job.id);
		}, 'Remove');
</script>

<div class="wb-dl">
	<div class="wb-dl-top">
		<div class="min-w-0 flex-1">
			<div class="wb-dl-name" title={job.file}>{filename}</div>
			<div class="wb-dl-sub" title={job.repo_id}>{job.repo_id}</div>
		</div>
		{#if badge}
			<span class="wb-badge {badge.tone}">{badge.label}</span>
		{/if}
		{#if job.status === 'running'}
			<button type="button" class="wb-icon-btn" onclick={handlePause} aria-label="Pause" title="Pause">
				<Icon name="pause" size={13} />
			</button>
		{:else if job.status === 'paused' || job.status === 'failed'}
			<button type="button" class="wb-icon-btn" onclick={handleResume} aria-label="Resume" title="Resume">
				<Icon name="play" size={13} />
			</button>
		{/if}
		{#if job.status === 'running' || job.status === 'paused' || job.status === 'failed' || job.status === 'queued'}
			<button type="button" class="wb-icon-btn" onclick={handleCancel} aria-label="Cancel" title="Cancel">
				<Icon name="x" size={13} />
			</button>
		{:else}
			<button type="button" class="wb-icon-btn" onclick={handleDelete} aria-label="Remove from list" title="Remove from list">
				<Icon name="x" size={13} />
			</button>
		{/if}
	</div>

	{#if showBar}
		<div
			class="wb-pbar {job.status === 'paused' ? 'paused' : ''} {phaseLabel ? 'indeterminate' : ''}"
			role="progressbar"
			aria-label="Download progress"
			aria-valuemin={0}
			aria-valuemax={100}
			aria-valuenow={Math.round(percent)}
		>
			<i style="width: {percent}%"></i>
		</div>
		<div class="wb-dl-meta">
			<span class="num">{formatBytes(downloaded)} / {formatBytes(total)}</span>
			{#if phaseLabel}
				<span class="text-primary">{phaseLabel}</span>
			{:else if job.status === 'queued'}
				<!-- Downloads run one at a time; queued jobs start automatically. -->
				<span>Starts when the current download finishes</span>
			{:else if job.status === 'paused'}
				<span>Paused</span>
			{:else}
				<span class="num">{[humanSpeed(bps), humanEta(eta)].filter(Boolean).join(', ')}</span>
			{/if}
		</div>
	{:else if job.status === 'queued'}
		<div class="wb-dl-meta"><span>Starts when the current download finishes</span></div>
	{/if}

	{#if job.status === 'failed' && job.error}
		<p class="wb-dl-err">{job.error}</p>
	{/if}
</div>
