<script lang="ts">
	/**
	 * Recent-generations strip — this task's history from the asset library,
	 * under the canvas for quick compare and reuse. Clicking an item hands the
	 * asset back to the task page, which restores its prompt and settings.
	 */
	import Icon, { type IconName } from '$lib/components/shell/Icon.svelte';
	import { fileSrc } from '$lib/ipc/backend';
	import { listAssets } from '$lib/ipc/commands';
	import { baseName } from './studioMeta';
	import type { Asset } from '$lib/types';

	interface Props {
		/** Source-task ids to match in the library ('text_to_image', …). */
		tasks: string[];
		/** Any change reloads the strip — pass the latest result's job id. */
		refreshKey?: string | null;
		onremix?: (asset: Asset) => void;
	}

	let { tasks, refreshKey = null, onremix }: Props = $props();

	let items = $state<Asset[]>([]);

	async function load() {
		try {
			const all = await listAssets(undefined, 120);
			items = all.filter((a) => tasks.includes(a.task)).slice(0, 12);
		} catch {
			// The strip is an accelerator — failures just leave it empty.
		}
	}

	// Runs on mount and again whenever a new result lands for this task. The
	// post-result reload waits a beat so the backend's job-sync task has
	// committed the promoted asset row before we list.
	$effect(() => {
		if (refreshKey == null) {
			void load();
			return;
		}
		const t = setTimeout(() => void load(), 600);
		return () => clearTimeout(t);
	});

	function titleOf(a: Asset): string {
		const when = new Date(a.created_at * 1000).toLocaleString(undefined, {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit',
		});
		return `${a.prompt ?? baseName(a.path)}\n${when}. Click to reuse these settings.`;
	}

	function iconFor(kind: string): IconName {
		if (kind === 'video') return 'film';
		if (kind === 'audio') return 'speaker';
		return 'doc';
	}
</script>

<div class="recent">
	<div class="r-head">
		<span class="r-label">Recent</span>
		{#if items.length > 0}
			<span class="r-hint">Click one to reuse its settings</span>
		{/if}
	</div>
	{#if items.length === 0}
		<p class="r-none">Your generations from this tab show up here.</p>
	{:else}
		<div class="r-row">
			{#each items as a (a.id)}
				<button type="button" class="r-item" title={titleOf(a)} onclick={() => onremix?.(a)} aria-label="Reuse settings from {a.prompt ?? baseName(a.path)}">
					{#if a.kind === 'image'}
						<img src={fileSrc(a.path)} alt="" loading="lazy" />
					{:else}
						<span class="r-ic"><Icon name={iconFor(a.kind)} size={18} /></span>
					{/if}
					<span class="r-over" aria-hidden="true"><Icon name="remix" size={16} /></span>
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.recent {
		flex: none;
	}
	.r-head {
		display: flex;
		align-items: baseline;
		gap: 10px;
		margin-bottom: 8px;
	}
	.r-label {
		font-size: var(--text-xs);
		font-weight: 600;
		color: var(--foreground);
	}
	.r-hint,
	.r-none {
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.r-none {
		height: 64px;
		display: flex;
		align-items: center;
		padding: 0 14px;
		border: 1px dashed var(--border);
		border-radius: 10px;
	}
	.r-row {
		display: flex;
		gap: 8px;
		overflow-x: auto;
		padding-bottom: 2px;
	}
	.r-item {
		position: relative;
		flex: none;
		width: 64px;
		height: 64px;
		padding: 0;
		overflow: hidden;
		border: 1px solid var(--border);
		border-radius: 10px;
		background: var(--elevated);
		cursor: pointer;
		transition:
			border-color 0.15s ease,
			transform 0.15s ease;
	}
	.r-item:hover {
		border-color: var(--ring);
		transform: translateY(-1px);
	}
	.r-item:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
	}
	.r-item img {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}
	.r-ic {
		display: grid;
		place-items: center;
		width: 100%;
		height: 100%;
		color: var(--muted-foreground);
	}
	.r-over {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		background: rgb(0 0 0 / 0.5);
		color: white;
		opacity: 0;
		transition: opacity 0.15s ease;
	}
	.r-item:hover .r-over,
	.r-item:focus-visible .r-over {
		opacity: 1;
	}
	@media (prefers-reduced-motion: reduce) {
		.r-item {
			transition: none;
		}
		.r-item:hover {
			transform: none;
		}
	}
</style>
