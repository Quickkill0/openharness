<script lang="ts">
	/**
	 * Downloads panel — a floating card that rises above the memory dock
	 * (Workbench). Ruled rows: single files and whole-repo groups.
	 * Escape or the close button dismisses it; the page stays usable behind it.
	 */
	import Icon from '$lib/components/shell/Icon.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import JobRow from './JobRow.svelte';
	import GroupRow from './GroupRow.svelte';
	import { subscribeDownloadProgress, type ChannelLike } from '$lib/ipc/channels';
	import { listDownloads } from '$lib/ipc/commands';
	import { downloadStore, applyProgress, seedFromList } from '$lib/stores/downloads.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { unfinishedCount } from './format';
	import { onMount } from 'svelte';
	import type { DownloadProgress, Job } from '$lib/types';

	interface Props {
		open?: boolean;
		onclose?: () => void;
	}

	let { open = $bindable(false), onclose }: Props = $props();

	let loading = $state(true);
	let channel = $state<ChannelLike<DownloadProgress> | null>(null);

	// Latest progress snapshots keyed by job id — passed down to the rows.
	let progressMap = $state<Map<string, DownloadProgress>>(new Map());

	onMount(async () => {
		try {
			const jobs = await listDownloads();
			seedFromList(jobs);
		} catch (e) {
			addToast(`Failed to load downloads: ${e}`, 'error');
		} finally {
			loading = false;
		}

		try {
			channel = await subscribeDownloadProgress((progress) => {
				applyProgress(progress);
				const next = new Map(progressMap);
				next.set(progress.id, progress);
				progressMap = next;
			});
		} catch (e) {
			addToast(`Progress subscription failed: ${e}`, 'error');
		}
	});

	const jobs = $derived([...downloadStore.jobs.values()].sort((a, b) => b.created_at - a.created_at));
	/** Header count: every unfinished file, paused ones included. */
	const leftCount = $derived(unfinishedCount(jobs));

	/** One panel entry: an ungrouped job OR a whole repo download group (M9). */
	type Entry =
		| { kind: 'job'; key: string; job: Job; created_at: number }
		| { kind: 'group'; key: string; groupId: string; jobs: Job[]; created_at: number };

	const entries = $derived.by<Entry[]>(() => {
		const groups = new Map<string, Job[]>();
		const out: Entry[] = [];
		for (const job of jobs) {
			const gid = job.group_id;
			if (!gid) {
				out.push({ kind: 'job', key: job.id, job, created_at: job.created_at });
				continue;
			}
			let members = groups.get(gid);
			if (!members) {
				members = [];
				groups.set(gid, members);
				out.push({ kind: 'group', key: gid, groupId: gid, jobs: members, created_at: job.created_at });
			}
			members.push(job);
		}
		return out.sort((a, b) => b.created_at - a.created_at);
	});

	function close() {
		open = false;
		onclose?.();
	}

	function onWindowKey(e: KeyboardEvent) {
		if (open && e.key === 'Escape') close();
	}
</script>

<svelte:window onkeydown={onWindowKey} />

{#if open}
	<div class="dl-panel" role="dialog" aria-label="Downloads">
		<div class="dl-panel-h">
			<Icon name="download" size={15} />
			<span>Downloads</span>
			<span class="grow"></span>
			{#if leftCount > 0}
				<span class="active">
					<span class="num">{leftCount}</span>
					{leftCount === 1 ? 'file' : 'files'} left
				</span>
			{/if}
			<button type="button" class="wb-icon-btn" onclick={close} aria-label="Close downloads">
				<Icon name="x" size={14} />
			</button>
		</div>

		<div class="dl-panel-b">
			{#if loading}
				<div class="flex justify-center py-8"><Spinner size="md" /></div>
			{:else if entries.length === 0}
				<div class="wb-empty">
					<Icon name="download" size={24} />
					<b>No downloads</b>
					<span>Get models from the Store.</span>
				</div>
			{:else}
				{#each entries as e (e.key)}
					{#if e.kind === 'group'}
						<GroupRow groupId={e.groupId} jobs={e.jobs} {progressMap} />
					{:else}
						<JobRow job={e.job} progress={progressMap.get(e.job.id)} />
					{/if}
				{/each}
			{/if}
		</div>
	</div>
{/if}

<style>
	.dl-panel {
		position: fixed;
		right: 12px;
		bottom: 68px;
		z-index: 40;
		width: 440px;
		max-height: min(560px, calc(100vh - 120px));
		display: flex;
		flex-direction: column;
		border-radius: 14px;
		border: 1px solid var(--border);
		background: var(--surface);
		box-shadow:
			0 18px 48px -12px rgb(0 0 0 / 0.35),
			inset 0 1px 0 var(--inset-highlight);
		animation: dl-rise 0.2s cubic-bezier(0.2, 0.8, 0.2, 1);
	}
	@keyframes dl-rise {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
	}
	.dl-panel-h {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 10px 10px 14px;
		border-bottom: 1px solid var(--border);
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--foreground);
	}
	.active {
		font-size: var(--text-2xs);
		font-weight: 400;
		color: var(--muted-foreground);
	}
	.dl-panel-b {
		overflow-y: auto;
		min-height: 0;
	}
	@media (prefers-reduced-motion: reduce) {
		.dl-panel {
			animation: none;
		}
	}
</style>
