<script lang="ts">
	/**
	 * Library (Workbench) — every saved generation. Kind filters and prompt
	 * search over a tile grid on the left; the inspector on the right previews
	 * the selected item with its prompt and settings, and offers Reveal,
	 * Delete and Remix (restores the settings into the task page that made it).
	 */
	import { onMount } from 'svelte';
	import Icon, { type IconName } from '$lib/components/shell/Icon.svelte';
	import Modal from '$lib/components/ui/Modal.svelte';
	import Toggle from '$lib/components/ui/Toggle.svelte';
	import { fileSrc, IS_TAURI } from '$lib/ipc/backend';
	import { deleteAsset, listAssets, revealAsset, searchAssets } from '$lib/ipc/commands';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { formatBytes } from '$lib/format';
	import { baseName, kindLabel, paramSummary, remixTarget } from './studioMeta';
	import type { Asset, AssetKind } from '$lib/types';

	interface Props {
		/** Restore an asset into its task page (the Studio page switches tabs). */
		onremix?: (asset: Asset) => void;
	}

	let { onremix }: Props = $props();

	let assets = $state<Asset[]>([]);
	let loading = $state(true);
	let failed = $state<string | null>(null);
	let kind = $state<AssetKind | ''>('');
	let query = $state('');
	let selectedId = $state<string | null>(null);
	let deleteTarget = $state<Asset | null>(null);
	let deleteFile = $state(true);
	let deleting = $state(false);
	let debounce: ReturnType<typeof setTimeout> | undefined;

	const FILTERS: Array<{ value: AssetKind | ''; label: string }> = [
		{ value: '', label: 'All' },
		{ value: 'image', label: 'Images' },
		{ value: 'video', label: 'Video' },
		{ value: 'audio', label: 'Audio' },
		{ value: 'text', label: 'Transcripts' },
	];

	const TASK_LABELS: Record<string, string> = {
		text_to_image: 'Image',
		text_to_video: 'Video',
		text_to_speech: 'Speech',
		vibevoice: 'Cloned speech',
		transcribe: 'Transcript',
	};

	const selected = $derived(assets.find((a) => a.id === selectedId) ?? null);
	const settings = $derived(selected ? paramSummary(selected.params) : '');
	const canRemix = $derived(!!selected && remixTarget(selected.task) !== null);

	async function load() {
		loading = true;
		failed = null;
		try {
			assets = query.trim()
				? await searchAssets(query.trim(), 200)
				: await listAssets(kind || undefined, 200);
			if (!assets.some((a) => a.id === selectedId)) selectedId = assets[0]?.id ?? null;
		} catch (e) {
			failed = String(e);
		} finally {
			loading = false;
		}
	}

	onMount(load);

	function setKind(k: AssetKind | '') {
		kind = k;
		query = '';
		void load();
	}

	function onQueryInput() {
		clearTimeout(debounce);
		debounce = setTimeout(load, 250);
	}

	async function reveal(a: Asset) {
		if (!IS_TAURI) {
			addToast('Reveal in Explorer needs the desktop app', 'info');
			return;
		}
		try {
			await revealAsset(a.id);
		} catch (e) {
			addToast(`Couldn't reveal the file: ${e}`, 'error');
		}
	}

	async function confirmDelete() {
		if (!deleteTarget) return;
		const target = deleteTarget;
		deleting = true;
		try {
			await deleteAsset(target.id, deleteFile);
			const i = assets.findIndex((a) => a.id === target.id);
			assets = assets.filter((a) => a.id !== target.id);
			if (selectedId === target.id) selectedId = assets[Math.min(i, assets.length - 1)]?.id ?? null;
			addToast('Deleted from the library', 'success');
		} catch (e) {
			addToast(`Couldn't delete: ${e}`, 'error');
		} finally {
			deleting = false;
			deleteTarget = null;
		}
	}

	function fmtDate(secs: number): string {
		return new Date(secs * 1000).toLocaleString(undefined, {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit',
		});
	}

	function iconFor(k: string): IconName {
		if (k === 'image') return 'image';
		if (k === 'video') return 'film';
		if (k === 'audio') return 'speaker';
		return 'doc';
	}

	/** Arrow keys move the selection through the grid in reading order. */
	function onGridKey(e: KeyboardEvent) {
		if (!['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp'].includes(e.key) || assets.length === 0) return;
		const grid = e.currentTarget as HTMLElement;
		const cols = getComputedStyle(grid).gridTemplateColumns.split(' ').length || 1;
		const i = Math.max(0, assets.findIndex((a) => a.id === selectedId));
		const delta = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols }[e.key] ?? 0;
		const next = Math.min(assets.length - 1, Math.max(0, i + delta));
		e.preventDefault();
		selectedId = assets[next].id;
		grid.querySelector<HTMLElement>(`[data-id="${assets[next].id}"]`)?.focus();
	}
</script>

<Modal open={deleteTarget !== null} title="Delete from the library" size="sm" onclose={() => (deleteTarget = null)}>
	{#if deleteTarget}
		<p class="del-lead">
			Delete <b>{baseName(deleteTarget.path)}</b> from the library?
		</p>
		<div class="del-row">
			<span>Also delete the file from disk</span>
			<Toggle bind:checked={deleteFile} label="Also delete the file from disk" />
		</div>
	{/if}
	{#snippet footer()}
		<button type="button" class="wb-btn ghost" onclick={() => (deleteTarget = null)}>Cancel</button>
		<button type="button" class="wb-btn danger" disabled={deleting} onclick={confirmDelete}>
			{deleting ? 'Deleting…' : 'Delete'}
		</button>
	{/snippet}
</Modal>

<div class="lib">
	<section class="browse" aria-label="Library items">
		<div class="filterbar">
			<div class="wb-seg" role="group" aria-label="Show">
				{#each FILTERS as f (f.value)}
					<button type="button" aria-pressed={kind === f.value && !query} onclick={() => setKind(f.value)}>
						{f.label}
					</button>
				{/each}
			</div>
			<span class="flex-1"></span>
			{#if !loading && !failed}
				<span class="count"><span class="num">{assets.length}</span> {assets.length === 1 ? 'item' : 'items'}</span>
			{/if}
			<label class="wb-search search">
				<Icon name="search" size={14} />
				<input bind:value={query} placeholder="Search prompts" oninput={onQueryInput} aria-label="Search the library" />
			</label>
		</div>

		<div class="grid-wrap">
			{#if loading && assets.length === 0}
				<div class="tiles" aria-hidden="true">
					{#each { length: 8 } as _, i (i)}<div class="tile skel"></div>{/each}
				</div>
			{:else if failed}
				<div class="wb-empty">
					<b>Couldn't load the library</b>
					<span>{failed}</span>
					<button type="button" class="wb-btn sm" onclick={load}><Icon name="refresh" size={13} /> Try again</button>
				</div>
			{:else if assets.length === 0}
				<div class="wb-empty">
					<span class="empty-ic"><Icon name="folder" size={22} /></span>
					<b>{query ? 'Nothing matches that search' : 'Nothing here yet'}</b>
					<span>{query ? 'Try other words from the prompt or file name.' : 'Images, clips, speech and transcripts you generate land here.'}</span>
				</div>
			{:else}
				<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
				<div class="tiles" role="listbox" tabindex="-1" aria-label="Library items" onkeydown={onGridKey}>
					{#each assets as a (a.id)}
						<button
							type="button"
							role="option"
							class="tile"
							class:on={a.id === selectedId}
							aria-selected={a.id === selectedId}
							tabindex={a.id === selectedId ? 0 : -1}
							data-id={a.id}
							title={a.prompt ?? baseName(a.path)}
							onclick={() => (selectedId = a.id)}
							ondblclick={() => canRemix && onremix?.(a)}
						>
							{#if a.kind === 'image'}
								<img src={fileSrc(a.path)} alt="" loading="lazy" />
							{:else}
								<span class="tile-ic"><Icon name={iconFor(a.kind)} size={26} /></span>
							{/if}
							<span class="meta" class:plain={a.kind !== 'image'}>
								<span>{TASK_LABELS[a.task] ?? kindLabel(a.kind)}</span>
								<span class="when">{fmtDate(a.created_at)}</span>
							</span>
						</button>
					{/each}
				</div>
			{/if}
		</div>
	</section>

	<aside class="ins" aria-label="Selected item">
		{#if selected}
			<div class="ins-body">
				<div class="preview">
					{#if selected.kind === 'image'}
						<img src={fileSrc(selected.path)} alt={selected.prompt ?? baseName(selected.path)} />
					{:else if selected.kind === 'video' && IS_TAURI}
						<!-- svelte-ignore a11y_media_has_caption -->
						<video src={fileSrc(selected.path)} controls loop preload="metadata"></video>
					{:else if selected.kind === 'audio' && IS_TAURI}
						<div class="pv-audio">
							<Icon name="speaker" size={26} />
							<!-- svelte-ignore a11y_media_has_caption -->
							<audio controls src={fileSrc(selected.path)} preload="none"></audio>
						</div>
					{:else}
						<div class="pv-icon"><Icon name={iconFor(selected.kind)} size={30} /></div>
					{/if}
				</div>

				<div class="wb-sec">
					<div class="wb-sec-h"><span class="wb-sec-label">{selected.prompt ? (selected.kind === 'audio' ? 'Text' : 'Prompt') : 'File'}</span></div>
					<p class="prompt">{selected.prompt ?? baseName(selected.path)}</p>
				</div>

				<div class="wb-sec">
					<dl class="wb-props">
						<dt>Made with</dt>
						<dd>{TASK_LABELS[selected.task] ?? selected.task}</dd>
						{#if selected.model}
							<dt>Model</dt>
							<dd>{selected.model}</dd>
						{/if}
						{#if settings}
							<dt>Settings</dt>
							<dd>{settings}</dd>
						{/if}
						<dt>Created</dt>
						<dd>{fmtDate(selected.created_at)}</dd>
						<dt>Size</dt>
						<dd class="num">{formatBytes(selected.size)}</dd>
						<dt>File</dt>
						<dd class="file">{baseName(selected.path)}</dd>
					</dl>
				</div>
			</div>
			<div class="wb-ins-foot">
				<button type="button" class="wb-btn" onclick={() => reveal(selected)} title="Show the file in Explorer">
					<Icon name="folder" size={14} /> Reveal
				</button>
				<button
					type="button"
					class="wb-btn danger"
					onclick={() => {
						deleteFile = true;
						deleteTarget = selected;
					}}
				>
					<Icon name="trash" size={14} /> Delete
				</button>
				<button
					type="button"
					class="wb-btn primary grow"
					disabled={!canRemix}
					onclick={() => onremix?.(selected)}
					title={canRemix ? 'Open in its Studio tab with these settings' : 'No Studio tab can reuse this item'}
				>
					<Icon name="remix" size={14} /> Remix
				</button>
			</div>
		{:else}
			<div class="wb-empty ins-empty">
				<b>Select an item</b>
				<span>Its preview, prompt and actions show here.</span>
			</div>
		{/if}
	</aside>
</div>

<style>
	.lib {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 360px;
		height: 100%;
		min-height: 0;
		overflow: hidden;
	}
	@media (max-width: 1180px) {
		.lib {
			grid-template-columns: minmax(0, 1fr) 300px;
		}
	}
	.browse {
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
	}
	.filterbar {
		display: flex;
		align-items: center;
		gap: 10px;
		flex: none;
		padding: 10px 16px;
		border-bottom: 1px solid var(--border);
	}
	.count {
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
		white-space: nowrap;
	}
	.search {
		width: 240px;
	}
	@media (max-width: 1180px) {
		.search {
			width: 180px;
		}
	}
	.grid-wrap {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.tiles {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(136px, 1fr));
		gap: 10px;
		padding: 16px;
		outline: none;
	}
	.tile {
		position: relative;
		aspect-ratio: 1;
		padding: 0;
		overflow: hidden;
		border: 1px solid var(--border);
		border-radius: 10px;
		background: var(--surface);
		cursor: pointer;
		transition:
			border-color 0.15s ease,
			box-shadow 0.15s ease;
	}
	.tile:hover {
		border-color: var(--ring);
	}
	.tile.on {
		border-color: var(--primary);
		box-shadow: 0 0 0 1px var(--primary);
	}
	.tile:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
	}
	.tile img {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}
	.tile-ic {
		display: grid;
		place-items: center;
		width: 100%;
		height: 100%;
		padding-bottom: 20px;
		color: var(--muted-foreground);
		background: radial-gradient(color-mix(in oklch, var(--muted-foreground) 18%, transparent) 1px, transparent 1px) 0 0 /
			12px 12px;
	}
	.meta {
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		display: flex;
		flex-direction: column;
		padding: 14px 8px 6px;
		background: linear-gradient(transparent, rgb(0 0 0 / 0.68));
		color: white;
		font-size: var(--text-2xs);
		text-align: left;
		line-height: 1.35;
	}
	/* Icon tiles have no picture to darken, so the label sits on the tile itself. */
	.meta.plain {
		background: none;
		color: var(--foreground);
	}
	.meta.plain .when {
		color: var(--muted-foreground);
		opacity: 1;
	}
	.when {
		opacity: 0.8;
	}
	.skel {
		cursor: default;
		animation: lib-pulse 1.4s ease-in-out infinite;
		background: var(--elevated);
	}
	@keyframes lib-pulse {
		50% {
			opacity: 0.5;
		}
	}
	.empty-ic {
		display: grid;
		place-items: center;
		width: 48px;
		height: 48px;
		margin-bottom: 4px;
		border: 1px solid var(--border);
		border-radius: 12px;
		background: var(--elevated);
	}
	.wb-empty span {
		max-width: 360px;
		font-size: var(--text-xs);
	}

	.ins {
		display: flex;
		flex-direction: column;
		min-height: 0;
		border-left: 1px solid var(--border);
		background: var(--surface);
	}
	.ins-body {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.ins-empty {
		margin: auto 0;
	}
	.preview {
		padding: 16px;
		border-bottom: 1px solid var(--border);
	}
	.preview img,
	.preview video {
		display: block;
		width: 100%;
		max-height: 320px;
		object-fit: contain;
		border-radius: 10px;
		border: 1px solid var(--border);
		background: var(--background);
	}
	.pv-icon,
	.pv-audio {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 12px;
		aspect-ratio: 4 / 3;
		padding: 16px;
		border-radius: 10px;
		border: 1px solid var(--border);
		background: radial-gradient(color-mix(in oklch, var(--muted-foreground) 18%, transparent) 1px, transparent 1px) 0
				0 / 14px 14px,
			var(--background);
		color: var(--muted-foreground);
	}
	.pv-audio audio {
		width: 100%;
	}
	.prompt {
		font-size: var(--text-sm);
		line-height: 1.55;
		color: var(--foreground);
		overflow-wrap: anywhere;
		white-space: pre-wrap;
	}
	.file {
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.ins :global(.wb-ins-foot .grow) {
		flex: 1;
	}
	.del-lead {
		font-size: var(--text-sm);
		color: var(--muted-foreground);
	}
	.del-lead b {
		font-weight: 500;
		color: var(--foreground);
		overflow-wrap: anywhere;
	}
	.del-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		margin-top: 16px;
		font-size: var(--text-sm);
		color: var(--foreground);
	}
	@media (prefers-reduced-motion: reduce) {
		.skel {
			animation: none;
		}
		.tile {
			transition: none;
		}
	}
</style>
