<script lang="ts">
	/**
	 * Studio (Workbench) — one 48px header carrying the title, the task tabs
	 * and the worker status; the body is the active task page. Task pages stay
	 * mounted so switching tabs keeps form state and in-flight progress.
	 */
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import Icon from '$lib/components/shell/Icon.svelte';
	import ImageTask from '$lib/components/studio/ImageTask.svelte';
	import SpeechTask from '$lib/components/studio/SpeechTask.svelte';
	import TranscribeTask from '$lib/components/studio/TranscribeTask.svelte';
	import VideoTask from '$lib/components/studio/VideoTask.svelte';
	import AssetGallery from '$lib/components/studio/AssetGallery.svelte';
	import { initStudio, requestRemix, studio } from '$lib/stores/studio.svelte';
	import { remixTarget } from '$lib/components/studio/studioMeta';
	import type { Asset, GenerationTask } from '$lib/types';
	import type { IconName } from '$lib/components/shell/Icon.svelte';

	const TABS: { id: string; label: string; icon: IconName; task?: GenerationTask }[] = [
		{ id: 'image', label: 'Image', icon: 'image', task: 'text_to_image' },
		{ id: 'video', label: 'Video', icon: 'film', task: 'text_to_video' },
		{ id: 'speech', label: 'Speech', icon: 'speaker', task: 'text_to_speech' },
		{ id: 'transcribe', label: 'Transcribe', icon: 'mic', task: 'transcribe' },
		{ id: 'library', label: 'Library', icon: 'folder' },
	];

	let active = $state('image');

	const worker = $derived(studio.worker);
	const workerReady = $derived(worker.phase === 'ready');

	const workerText = $derived.by(() => {
		switch (worker.phase) {
			case 'ready':
				return 'Runtime ready';
			case 'not_installed':
				return 'Preparing the runtime';
			case 'installing':
				return `Installing the runtime, ${Math.round(worker.pct)}%`;
			case 'starting':
				return 'Starting the runtime';
			case 'failed':
				return 'Runtime failed to start';
			default:
				return 'Runtime starts when you generate';
		}
	});
	const workerTone = $derived(
		workerReady ? 'var(--success)' : worker.phase === 'failed' ? 'var(--error)' : 'var(--warning)',
	);

	onMount(() => {
		void initStudio();
		// `/studio?task=speech` opens on that tab.
		const task = page.url.searchParams.get('task');
		if (task && TABS.some((t) => t.id === task)) active = task;
	});

	/** Library "Remix": open the asset's task tab and restore its settings there. */
	function remixFromLibrary(asset: Asset) {
		const target = remixTarget(asset.task);
		if (!target) return;
		requestRemix(target.task, asset);
		active = target.tab;
	}

	function onTabKey(e: KeyboardEvent, i: number) {
		if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
		e.preventDefault();
		const next = (i + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length;
		active = TABS[next].id;
		document.getElementById(`studio-tab-${TABS[next].id}`)?.focus();
	}
</script>

<div class="studio">
	<header class="head">
		<h1>Studio</h1>
		<div class="tabs" role="tablist" aria-label="Studio task">
			{#each TABS as t, i (t.id)}
				<button
					type="button"
					role="tab"
					id="studio-tab-{t.id}"
					class="wb-tab"
					aria-selected={active === t.id}
					aria-controls="studio-panel-{t.id}"
					tabindex={active === t.id ? 0 : -1}
					onclick={() => (active = t.id)}
					onkeydown={(e) => onTabKey(e, i)}
				>
					<Icon name={t.icon} size={15} />
					{t.label}
					{#if t.task && studio.active[t.task]}
						<span class="run" title="Running"></span>
					{/if}
				</button>
			{/each}
		</div>
		<span class="flex-1"></span>
		<a
			href="/settings/engines"
			class="worker {workerReady ? 'ok' : ''}"
			title={workerReady ? 'The Python runtime is running' : 'Open engine settings'}
		>
			<span class="wb-dot" style="--tone: {workerTone}">{workerText}</span>
		</a>
	</header>

	<div class="body">
		<div id="studio-panel-image" role="tabpanel" aria-labelledby="studio-tab-image" class="panel" hidden={active !== 'image'}>
			<ImageTask />
		</div>
		<div id="studio-panel-video" role="tabpanel" aria-labelledby="studio-tab-video" class="panel" hidden={active !== 'video'}>
			<VideoTask />
		</div>
		<div id="studio-panel-speech" role="tabpanel" aria-labelledby="studio-tab-speech" class="panel" hidden={active !== 'speech'}>
			<SpeechTask />
		</div>
		<div id="studio-panel-transcribe" role="tabpanel" aria-labelledby="studio-tab-transcribe" class="panel" hidden={active !== 'transcribe'}>
			<TranscribeTask />
		</div>
		{#if active === 'library'}
			<div id="studio-panel-library" role="tabpanel" aria-labelledby="studio-tab-library" class="panel">
				<AssetGallery onremix={remixFromLibrary} />
			</div>
		{/if}
	</div>
</div>

<style>
	.studio {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}
	.head {
		display: flex;
		align-items: stretch;
		height: 48px;
		flex: none;
		padding: 0 16px 0 16px;
		border-bottom: 1px solid var(--border);
	}
	h1 {
		align-self: center;
		margin-right: 16px;
		font-size: var(--text-lg);
		font-weight: 600;
		letter-spacing: -0.01em;
		color: var(--foreground);
	}
	.tabs {
		display: flex;
		gap: 2px;
		overflow: hidden;
	}
	.tabs .wb-tab {
		height: 47px;
	}
	.run {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--primary);
		box-shadow: 0 0 8px var(--glow-color);
		animation: run-pulse 1.4s ease-in-out infinite;
	}
	@keyframes run-pulse {
		50% {
			opacity: 0.35;
		}
	}
	.worker {
		align-self: center;
		padding: 5px 10px;
		border-radius: 999px;
		border: 1px solid var(--border);
		font-size: var(--text-xs);
		color: var(--muted-foreground);
		white-space: nowrap;
		transition:
			color 0.15s ease,
			border-color 0.15s ease;
	}
	.worker:not(.ok) {
		border-color: color-mix(in oklch, var(--warning) 35%, transparent);
		color: var(--foreground);
	}
	.worker:hover {
		color: var(--foreground);
		border-color: var(--ring);
	}
	.worker:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
	}
	.body {
		flex: 1;
		min-height: 0;
	}
	.panel {
		height: 100%;
	}
	.panel[hidden] {
		display: none;
	}
	@media (prefers-reduced-motion: reduce) {
		.run {
			animation: none;
		}
	}
</style>
