<script lang="ts">
	/**
	 * Chat model picker (Workbench) — header button plus a ruled list of every
	 * installed chat model, loaded first. Picking a loaded model just closes
	 * the list; picking one that isn't loaded loads it through the unified
	 * ledger flow (`loadModel`), so a needs-eviction outcome opens the global
	 * EvictionDialog and an in-use refusal from the backend surfaces as a
	 * toast. The list reacts live to ledger events.
	 */
	import { onMount, untrack } from 'svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import Icon from '$lib/components/shell/Icon.svelte';
	import { listLocalModels } from '$lib/ipc/commands';
	import { engineStore } from '$lib/stores/engine.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { initLedgerStore, ledgerStore, loadModel } from '$lib/stores/ledger.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { formatBytes } from '$lib/format';
	import { ensureFit, fitStore, invalidateFits, retryFit } from '$lib/stores/fits.svelte';
	import {
		PLAN_PENDING,
		defaultCtx,
		hasFitEstimate,
		textEngineBlock,
	} from '$lib/components/models/modelMeta';
	import type { InstalledModel } from '$lib/types';

	interface Props {
		/** Reports the current model's display name (for the composer placeholder). */
		oncurrent?: (name: string | null) => void;
	}

	let { oncurrent }: Props = $props();

	let open = $state(false);
	let models = $state<InstalledModel[]>([]);
	let fetched = $state(false);
	let rootEl = $state<HTMLDivElement | null>(null);

	const currentId = $derived(engineStore.currentModelId);
	const current = $derived(models.find((m) => m.id === currentId) ?? null);
	const loadingModel = $derived(models.find((m) => ledgerStore.isLoading(m.id)) ?? null);
	const engineLoading = $derived(engineStore.phase === 'loading');
	const streaming = $derived(chatStore.streaming !== null);

	/** Loaded first, then alphabetical. */
	const sorted = $derived(
		[...models].sort((a, b) => {
			const la = ledgerStore.isLoaded(a.id) ? 0 : 1;
			const lb = ledgerStore.isLoaded(b.id) ? 0 : 1;
			return la - lb || a.display_name.localeCompare(b.display_name);
		}),
	);

	$effect(() => {
		oncurrent?.(current?.display_name ?? null);
	});

	async function refresh() {
		try {
			models = (await listLocalModels()).filter((m) => m.caps.workflows.includes('chat'));
		} catch {
			// Empty list + the Store link below is the failure UI.
		}
		fetched = true;
	}

	onMount(() => {
		void initLedgerStore();
		void refresh();
	});

	function toggle() {
		open = !open;
		if (open) void refresh();
	}

	/** A row that never answers shouldn't hold back the rows after it. */
	const ROW_PLAN_TIMEOUT_MS = 20_000;
	let passKey: string | null = null;
	let engineSeen: string | null = null;
	let planRun = 0;

	// Only a plan says which engine a row needs, whichever engines exist. Each
	// estimate profiles the hardware, so the open list runs one pass, one row at
	// a time, and reuses what the fit store has. New answers are needed only
	// when a load, an unload or an engine change moved the memory picture.
	$effect(() => {
		if (!open) {
			planRun++;
			passKey = null;
			return;
		}
		const phase = engineStore.phase;
		if (phase === 'loading' || phase === 'installing') return;
		const { llama, nativeStreaming } = engineStore.textEngines;
		const engine = `${phase}|${engineStore.currentModelId}|${llama}|${nativeStreaming}`;
		const rows = models.filter(hasFitEstimate).map((m) => ({ id: m.id, ctx: defaultCtx(m) }));
		const key = `${engine}#${rows.map((r) => `${r.id}:${r.ctx}`).join()}`;
		if (key === passKey) return;
		passKey = key;
		if (engineSeen !== null && engineSeen !== engine) untrack(() => invalidateFits([]));
		engineSeen = engine;
		const run = ++planRun;
		void untrack(async () => {
			for (const r of rows) {
				if (run !== planRun) return;
				let timer: ReturnType<typeof setTimeout> | undefined;
				await Promise.race([
					retryFit(r.id, r.ctx),
					new Promise((done) => (timer = setTimeout(done, ROW_PLAN_TIMEOUT_MS))),
				]);
				clearTimeout(timer);
			}
		});
	});

	function engineBlock(m: InstalledModel): string | null {
		return textEngineBlock(m, fitStore.get(m.id), fitStore.error(m.id), engineStore.textEngines);
	}

	// Close on outside click / Escape while the list is open.
	$effect(() => {
		if (!open) return;
		const onDocDown = (e: MouseEvent) => {
			if (rootEl && !rootEl.contains(e.target as Node)) open = false;
		};
		const onKey = (e: KeyboardEvent) => {
			if (e.key === 'Escape') open = false;
		};
		window.addEventListener('mousedown', onDocDown);
		window.addEventListener('keydown', onKey);
		return () => {
			window.removeEventListener('mousedown', onDocDown);
			window.removeEventListener('keydown', onKey);
		};
	});

	async function pick(m: InstalledModel) {
		open = false;
		if (ledgerStore.isLoaded(m.id) || ledgerStore.isLoading(m.id)) return;
		try {
			const ok = await loadModel(m.id, 'text');
			if (ok) addToast(`Loaded ${m.display_name}`, 'success');
		} catch (e) {
			addToast(`Couldn't load ${m.display_name}: ${e}`, 'error');
		}
	}

	function subline(m: InstalledModel): string {
		const parts = [m.quantization, m.params_b ? `${m.params_b}B` : null, formatBytes(m.total_size)];
		return parts.filter((p) => p && p !== '—').join(', ');
	}
</script>

<div bind:this={rootEl} class="picker">
	<button
		type="button"
		class="picker-btn"
		onclick={toggle}
		aria-haspopup="listbox"
		aria-expanded={open}
		aria-label="Choose chat model"
	>
		{#if loadingModel || engineLoading}
			<Spinner size="sm" />
			<span class="name">Loading {loadingModel?.display_name ?? 'model'}…</span>
		{:else if current || currentId}
			<span class="wb-dot" style="--tone: var(--success)"></span>
			<span class="name">{current?.display_name ?? currentId}</span>
		{:else}
			<span class="wb-dot" style="--tone: var(--muted-foreground)"></span>
			<span class="name">Pick a model</span>
		{/if}
		<Icon name="chevron" size={12} class="caret {open ? 'open' : ''}" />
	</button>

	{#if open}
		<div class="list" role="listbox" aria-label="Chat models">
			{#if !fetched}
				<div class="p-2"><div class="h-9 animate-pulse rounded-md bg-muted"></div></div>
			{:else if models.length === 0}
				<div class="wb-empty py-6">
					<b>No chat models installed</b>
					<a href="/models/browse" class="wb-btn sm mt-1">Open the Store</a>
				</div>
			{:else}
				{#each sorted as m (m.id)}
					{@const loaded = ledgerStore.isLoaded(m.id)}
					{@const loading = ledgerStore.isLoading(m.id)}
					{@const blocked = !loaded && streaming}
					{@const noEngine = loaded ? null : engineBlock(m)}
					{@const checking = noEngine === PLAN_PENDING}
					<button
						type="button"
						role="option"
						aria-selected={m.id === currentId}
						class="item {m.id === currentId ? 'on' : ''}"
						disabled={blocked || loading || noEngine !== null}
						title={blocked ? 'Wait for the current reply to finish before switching models' : (noEngine ?? undefined)}
						onclick={() => pick(m)}
					>
						<span class="n">{m.display_name}</span>
						<span>
							{#if loading}
								<Spinner size="sm" />
							{:else if loaded}
								<span class="wb-badge ok">Loaded</span>
							{:else if checking}
								<span class="wb-badge">Checking</span>
							{:else if !noEngine}
								<span class="wb-badge teal">Load</span>
							{/if}
						</span>
						<span class="s">{checking ? subline(m) : (noEngine ?? subline(m))}</span>
					</button>
					{#if !loaded && hasFitEstimate(m) && fitStore.error(m.id)}
						<button type="button" class="wb-btn sm retry" onclick={() => retryFit(m.id, defaultCtx(m))}>
							Retry check
						</button>
					{/if}
				{/each}
			{/if}
			<div class="foot">
				<a href="/models">Load options and memory in Models</a>
			</div>
		</div>
	{/if}
</div>

<style>
	.picker {
		position: relative;
		min-width: 0;
	}
	.picker-btn {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		max-width: 300px;
		height: 32px;
		padding: 0 10px;
		border-radius: 8px;
		border: 1px solid var(--border);
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
		font-size: var(--text-sm);
		font-weight: 500;
		color: var(--foreground);
		cursor: pointer;
		transition: border-color 0.15s ease;
	}
	.picker-btn:hover {
		border-color: color-mix(in oklch, var(--primary) 40%, var(--border));
	}
	.picker-btn:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
	}
	.name {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.picker-btn :global(.caret) {
		flex: none;
		opacity: 0.6;
		transform: rotate(90deg);
		transition: transform 0.15s ease;
	}
	.picker-btn :global(.caret.open) {
		transform: rotate(-90deg);
	}
	.list {
		position: absolute;
		top: 38px;
		left: 0;
		z-index: 30;
		width: 380px;
		max-height: 420px;
		overflow-y: auto;
		padding: 4px;
		border-radius: 10px;
		border: 1px solid var(--border);
		background: var(--elevated);
		box-shadow: 0 18px 48px -12px rgb(0 0 0 / 0.4);
	}
	.item {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: 2px 10px;
		align-items: center;
		width: 100%;
		padding: 8px 10px;
		border-radius: 7px;
		border: 0;
		background: transparent;
		text-align: left;
		cursor: pointer;
		color: var(--foreground);
	}
	.item:hover:not(:disabled) {
		background: var(--muted);
	}
	.item:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: -2px;
	}
	.item:disabled {
		cursor: not-allowed;
		opacity: 0.5;
	}
	.item.on {
		box-shadow: inset 2px 0 0 var(--primary);
	}
	.n {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--text-sm);
		font-weight: 500;
	}
	.s {
		grid-column: 1 / -1;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.retry {
		margin: 0 0 6px 10px;
	}
	.foot {
		margin-top: 4px;
		padding: 8px 10px 6px;
		border-top: 1px solid var(--border);
		font-size: var(--text-2xs);
	}
	.foot a {
		color: var(--muted-foreground);
	}
	.foot a:hover {
		color: var(--foreground);
	}
	@media (prefers-reduced-motion: reduce) {
		.picker-btn :global(.caret) {
			transition: none;
		}
	}
</style>
