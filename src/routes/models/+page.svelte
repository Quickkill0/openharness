<script lang="ts">
	/**
	 * Models — the Workbench home: a dense installed-models table with a 360px
	 * inspector for the selected row. Fit estimates come from one shared cache
	 * (fits.svelte.ts) so the table, inspector and dock preview agree.
	 */
	import { onMount, untrack } from 'svelte';
	import { goto } from '$app/navigation';
	import PageHeader from '$lib/components/shell/PageHeader.svelte';
	import Icon from '$lib/components/shell/Icon.svelte';
	import Menu, { type MenuItem } from '$lib/components/shell/Menu.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import ModelInspector from '$lib/components/models/ModelInspector.svelte';
	import ModelLoadDialog from '$lib/components/models/ModelLoadDialog.svelte';
	import DeleteModelDialog from '$lib/components/models/DeleteModelDialog.svelte';
	import { listLocalModels, deleteLocalModel, revealModel } from '$lib/ipc/commands';
	import { initLedgerStore, ledgerStore, loadModel, unloadModel } from '$lib/stores/ledger.svelte';
	import { engineStore, refreshNativeEngine } from '$lib/stores/engine.svelte';
	import { downloadStore } from '$lib/stores/downloads.svelte';
	import { fitStore, ensureFit, invalidateFits, retryFit } from '$lib/stores/fits.svelte';
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import { unloadLock } from '$lib/stores/chipLocks.svelte';
	import { residentSignature } from '$lib/ledgerSignature';
	import { setDockPreview } from '$lib/stores/ui.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { formatBytes, gbFromMb } from '$lib/format';
	import {
		FORMAT_LABELS,
		TASK_TABS,
		WORKFLOW_LABELS,
		canOpenLoadOptions,
		defaultCtx,
		family,
		hasFitEstimate,
		isLoadable,
		params,
		residentSlot,
		sizeOnDisk,
		studioTask,
		PLAN_PENDING,
		textEngineBlock,
	} from '$lib/components/models/modelMeta';
	import type { InstalledModel, Workflow } from '$lib/types';

	/** NVML-reported use must move this far before fits are refetched... */
	const VRAM_DRIFT_MB = 256;
	/** ...and stay moved this long (one telemetry tick is ~1 s). */
	const VRAM_SETTLE_MS = 2500;

	// --- State -----------------------------------------------------------
	let models = $state<InstalledModel[]>([]);
	let listLoading = $state(true);
	let filter = $state('');
	let tab = $state<'all' | Workflow>('all');
	let selectedId = $state<string | null>(null);
	let busyId = $state<string | null>(null);

	let deleteTarget = $state<InstalledModel | null>(null);
	let optionsOpen = $state(false);
	let optionsTarget = $state<InstalledModel | null>(null);
	let menuAnchor = $state<HTMLElement | null>(null);
	let menuModel = $state<InstalledModel | null>(null);

	// --- Derived ---------------------------------------------------------
	const counts = $derived.by(() => {
		const c: Record<string, number> = { all: models.length };
		for (const m of models) for (const w of m.caps.workflows) c[w] = (c[w] ?? 0) + 1;
		return c;
	});

	const visible = $derived.by(() => {
		const q = filter.trim().toLowerCase();
		return models.filter((m) => {
			if (tab !== 'all' && !m.caps.workflows.includes(tab)) return false;
			if (!q) return true;
			return (
				m.display_name.toLowerCase().includes(q) ||
				(m.repo_id ?? '').toLowerCase().includes(q) ||
				(m.architecture ?? '').toLowerCase().includes(q) ||
				(m.quantization ?? '').toLowerCase().includes(q)
			);
		});
	});

	const totalBytes = $derived(models.reduce((s, m) => s + sizeOnDisk(m), 0));
	const selected = $derived(models.find((m) => m.id === selectedId) ?? null);

	// --- Load list -------------------------------------------------------
	async function load() {
		listLoading = true;
		try {
			models = await listLocalModels();
			// Cached estimates may predate this visit (or this download):
			// free graphics memory has likely moved since. Refetch all of them.
			invalidateFits(models.filter(hasFitEstimate).map((m) => ({ id: m.id, ctx: defaultCtx(m) })));
			if (!selectedId || !models.some((m) => m.id === selectedId)) {
				selectedId = models.find((m) => ledgerStore.isLoaded(m.id))?.id ?? models[0]?.id ?? null;
			}
		} catch (e) {
			addToast(`Couldn't list installed models: ${e}`, 'error');
		} finally {
			listLoading = false;
		}
	}

	onMount(() => {
		void initLedgerStore();
		void load();
		void refreshNativeEngine();
		return () => {
			if (reloadTimer) clearTimeout(reloadTimer);
			if (vramTimer) clearTimeout(vramTimer);
			setDockPreview(null);
		};
	});

	// Re-query when a download finishes (DownloadDrawer keeps downloadStore live).
	let doneSeen: number | null = null;
	let reloadTimer: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		let done = 0;
		for (const j of downloadStore.jobs.values()) if (j.status === 'done') done++;
		if (doneSeen !== null && done > doneSeen) {
			if (reloadTimer) clearTimeout(reloadTimer);
			reloadTimer = setTimeout(() => void load(), 600);
		}
		doneSeen = done;
	});

	// --- Fit estimates ---------------------------------------------------
	// Each request carries the context a plain Load would use for that model.
	const fitTargets = $derived(
		models.filter(hasFitEstimate).map((m) => ({ id: m.id, ctx: defaultCtx(m) })),
	);
	$effect(() => {
		for (const t of fitTargets) void ensureFit(t.id, t.ctx);
	});

	// Free graphics memory moves when a resident comes, goes, is swapped for
	// another model in the same slot, or settles to its measured size.
	const residentSig = $derived(residentSignature(ledgerStore.residents));
	let lastSig: string | null = null;
	$effect(() => {
		const sig = residentSig;
		if (lastSig !== null && sig !== lastSig) {
			invalidateFits(untrack(() => fitTargets));
			// This refetch already covers the VRAM the change moved.
			vramAtFit = untrack(() => (gpuStore.hasGpu ? gpuStore.vramUsed : null));
		}
		lastSig = sig;
	});

	// Other apps (games, a browser) move free graphics memory too. Refetch
	// once NVML-reported use has drifted far enough and then held still.
	let vramAtFit: number | null = null;
	let vramTimer: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		const used = gpuStore.vramUsed;
		if (!gpuStore.hasGpu) return;
		if (vramAtFit === null) {
			vramAtFit = used;
			return;
		}
		if (Math.abs(used - vramAtFit) < VRAM_DRIFT_MB) {
			// Drifted back (a spike); nothing to refetch.
			if (vramTimer) clearTimeout(vramTimer);
			vramTimer = null;
			return;
		}
		if (vramTimer) return;
		vramTimer = setTimeout(() => {
			vramTimer = null;
			vramAtFit = gpuStore.vramUsed;
			invalidateFits(untrack(() => fitTargets));
		}, VRAM_SETTLE_MS);
	});

	// Dock "will load" preview for the selected, unloaded text model. With no
	// GPU there is no graphics memory to preview: llama.cpp runs on the CPU.
	$effect(() => {
		const m = selected;
		const est = m && hasFitEstimate(m) && !ledgerStore.isLoaded(m.id) ? fitStore.get(m.id) : null;
		const fit = est?.gpu ? est : null;
		// estimated_mb already includes the context cache when a ctx was
		// passed. A chat load swaps out the running llama.cpp model.
		const swapped =
			fit && fit.reclaimable_mb > 0
				? (ledgerStore.residents.find((r) => r.kind === 'text_engine') ?? null)
				: null;
		setDockPreview(
			m && fit
				? {
						modelId: m.id,
						label: m.display_name,
						vramMb: fit.estimated_mb,
						replacesId: swapped?.id ?? null,
						freesMb: swapped ? fit.reclaimable_mb : 0,
					}
				: null,
		);
	});

	// --- Actions ---------------------------------------------------------
	/** Why this model can't start with the engines installed, or null. */
	function engineBlock(m: InstalledModel): string | null {
		return textEngineBlock(m, fitStore.get(m.id), fitStore.error(m.id), engineStore.textEngines);
	}

	async function quickLoad(m: InstalledModel) {
		try {
			const ok = await loadModel(m.id, residentSlot(m));
			if (ok) addToast(`Loaded ${m.display_name}`, 'success');
			return ok;
		} catch (e) {
			addToast(`Couldn't load ${m.display_name}: ${e}`, 'error');
			return false;
		}
	}

	async function unload(m: InstalledModel) {
		const r = ledgerStore.residentFor(m.id);
		const lock = r ? unloadLock(r) : null;
		if (lock) {
			addToast(`Can't unload ${m.display_name} yet. ${lock.reason}`, 'error');
			return;
		}
		busyId = m.id;
		try {
			await unloadModel(m.id);
			addToast(`Unloaded ${m.display_name}`, 'success');
		} catch (e) {
			addToast(`Couldn't unload ${m.display_name}: ${e}`, 'error');
		} finally {
			busyId = null;
		}
	}

	function openOptions(m: InstalledModel) {
		optionsTarget = m;
		optionsOpen = true;
	}

	async function useInChat(m: InstalledModel) {
		if (!ledgerStore.isLoaded(m.id)) {
			const blocked = engineBlock(m);
			if (blocked) {
				addToast(blocked, blocked === PLAN_PENDING ? 'info' : 'error');
				return;
			}
			if (!(await quickLoad(m))) return;
		}
		void goto('/chat');
	}

	function useInStudio(m: InstalledModel) {
		const task = studioTask(m);
		if (task) void goto(`/studio?task=${task}`);
	}

	async function reveal(m: InstalledModel) {
		try {
			await revealModel(m.id);
		} catch (e) {
			addToast(`Couldn't open Explorer: ${e}`, 'error');
		}
	}

	async function confirmDelete(id: string) {
		const r = ledgerStore.residentFor(id);
		const lock = r ? unloadLock(r) : null;
		if (lock) {
			addToast(`Can't delete a model that is in use. ${lock.reason}`, 'error');
			deleteTarget = null;
			return;
		}
		try {
			if (r) await unloadModel(id);
			await deleteLocalModel(id);
			const idx = visible.findIndex((m) => m.id === id);
			models = models.filter((m) => m.id !== id);
			if (selectedId === id) selectedId = visible[Math.min(idx, visible.length - 1)]?.id ?? null;
			addToast('Model deleted', 'success');
		} catch (e) {
			addToast(`Couldn't delete: ${e}`, 'error');
		} finally {
			deleteTarget = null;
		}
	}

	function menuItems(m: InstalledModel): MenuItem[] {
		const loadable = isLoadable(m);
		const loaded = ledgerStore.isLoaded(m.id);
		const engineHint = engineBlock(m) ?? undefined;
		const optionsOk = canOpenLoadOptions(m, engineStore.textEngines);
		const items: MenuItem[] = [
			{
				label: 'Load with options…',
				onselect: () => openOptions(m),
				disabled: !loadable || loaded || !optionsOk,
				hint: !loadable ? 'Nothing can run this model yet' : loaded ? 'Already loaded' : optionsOk ? undefined : engineHint,
			},
		];
		if (m.caps.workflows.includes('chat')) {
			items.push({ label: 'Use in Chat', onselect: () => void useInChat(m), disabled: engineHint !== undefined && !loaded, hint: loaded ? undefined : engineHint });
		}
		if (studioTask(m)) {
			items.push({ label: 'Use in Studio', onselect: () => useInStudio(m) });
		}
		items.push(
			{ label: 'Show in Explorer', onselect: () => void reveal(m), separator: true },
			{ label: 'Delete…', onselect: () => (deleteTarget = m), danger: true },
		);
		return items;
	}

	function openMenu(e: MouseEvent, m: InstalledModel) {
		e.stopPropagation();
		const btn = e.currentTarget as HTMLElement;
		if (menuAnchor === btn) {
			menuAnchor = null;
			return;
		}
		selectedId = m.id;
		menuModel = m;
		menuAnchor = btn;
	}

	// Keyboard: arrows move the selection, Enter loads/unloads.
	let tbody = $state<HTMLTableSectionElement | null>(null);
	function onRowKey(e: KeyboardEvent, m: InstalledModel) {
		const i = visible.findIndex((v) => v.id === m.id);
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			const next = visible[e.key === 'ArrowDown' ? Math.min(i + 1, visible.length - 1) : Math.max(i - 1, 0)];
			if (next) {
				selectedId = next.id;
				tbody?.querySelector<HTMLElement>(`[data-id="${CSS.escape(next.id)}"]`)?.focus();
			}
		} else if (e.key === 'Enter' && e.target === e.currentTarget) {
			e.preventDefault();
			if (ledgerStore.isLoaded(m.id)) void unload(m);
			else if (isLoadable(m) && !engineBlock(m)) void quickLoad(m);
		}
	}

	const FIT_LABEL = { fits: 'Fits', tight: 'Tight', wont_fit: "Won't fit" } as const;
	const FIT_TONE = { fits: 'var(--success)', tight: 'var(--warning)', wont_fit: 'var(--error)' } as const;
</script>

<DeleteModelDialog model={deleteTarget} onconfirm={confirmDelete} oncancel={() => (deleteTarget = null)} />
{#if optionsTarget}
	<ModelLoadDialog bind:open={optionsOpen} model={optionsTarget} />
{/if}
<Menu
	anchor={menuAnchor}
	items={menuModel ? menuItems(menuModel) : []}
	label="Model actions"
	onclose={() => (menuAnchor = null)}
/>

<div class="flex h-full min-h-0 flex-col">
	<PageHeader
		title="Models"
		meta={listLoading && models.length === 0
			? 'Loading…'
			: `${models.length} ${models.length === 1 ? 'model' : 'models'} · ${formatBytes(totalBytes)} on disk`}
	>
		{#snippet actions()}
			<div class="tabs" role="tablist" aria-label="Filter by task">
				{#each TASK_TABS as t (t.id)}
					{#if t.id === 'all' || counts[t.id]}
						<button
							type="button"
							role="tab"
							aria-selected={tab === t.id}
							class="tab"
							onclick={() => (tab = t.id)}
						>
							{t.label}
							<span class="num c">{counts[t.id] ?? 0}</span>
						</button>
					{/if}
				{/each}
			</div>
			<label class="search">
				<Icon name="search" size={14} class="text-muted-foreground" />
				<input type="search" bind:value={filter} placeholder="Filter installed" aria-label="Filter installed models" />
			</label>
			<a href="/models/browse" class="store-btn">
				<Icon name="store" size={14} />
				Open Store
			</a>
		{/snippet}
	</PageHeader>

	<div class="body">
		<div class="table-wrap">
			{#if listLoading && models.length === 0}
				<div class="flex justify-center py-16"><Spinner size="lg" /></div>
			{:else if models.length === 0}
				<div class="empty">
					<Icon name="models" size={28} class="text-muted-foreground" />
					<h2 class="text-base font-semibold text-foreground">No models installed yet</h2>
					<p class="max-w-sm text-sm text-muted-foreground">
						Find a model in the Store and download it. It shows up here when the download finishes.
					</p>
					<a href="/models/browse" class="store-btn mt-1">Open the Store</a>
				</div>
			{:else if visible.length === 0}
				<div class="empty">
					<p class="text-sm font-medium text-foreground">
						No installed models match{filter.trim() ? ` "${filter.trim()}"` : ' this tab'}
					</p>
					<button type="button" class="store-btn ghost" onclick={() => { filter = ''; tab = 'all'; }}>
						Clear filters
					</button>
				</div>
			{:else}
				<table class="wb-table">
					<thead>
						<tr>
							<th class="w-7" aria-label="State"></th>
							<th>Name</th>
							<th class="opt">Family</th>
							<th class="r nar">Params</th>
							<th>Quant</th>
							<th class="r">Size</th>
							<th class="opt">Format</th>
							<th class="nar">Task</th>
							<th>Fit</th>
							<th class="r" aria-label="Load"></th>
							<th class="w-9" aria-label="More"></th>
						</tr>
					</thead>
					<tbody bind:this={tbody}>
						{#each visible as m (m.id)}
							{@const resident = ledgerStore.residentFor(m.id)}
							{@const loading = ledgerStore.isLoading(m.id)}
							{@const fit = hasFitEstimate(m) ? fitStore.get(m.id) : null}
							<tr
								data-id={m.id}
								tabindex={selectedId === m.id ? 0 : -1}
								aria-selected={selectedId === m.id}
								onclick={() => (selectedId = m.id)}
								onkeydown={(e) => onRowKey(e, m)}
								onfocus={() => (selectedId = m.id)}
							>
								<td>
									<span
										class="sdot {resident ? 'on' : loading ? 'busy' : ''}"
										title={resident ? 'Loaded' : loading ? 'Loading' : 'Not loaded'}
									></span>
								</td>
								<td class="name" title={m.repo_id}>{m.display_name}</td>
								<td class="muted opt">{family(m)}</td>
								<td class="r num nar">{params(m)}</td>
								<td class="num muted">{m.quantization ?? '—'}</td>
								<td class="r num">{formatBytes(sizeOnDisk(m))}</td>
								<td class="muted opt">{FORMAT_LABELS[m.format]}</td>
								<td class="muted nar">
									{m.caps.workflows.length ? m.caps.workflows.map((w) => WORKFLOW_LABELS[w]).join(', ') : '—'}
								</td>
								<td>
									{#if resident}
										<span class="fit" style="--tone: var(--tier-vram)">
											In use <span class="num muted">{gbFromMb(resident.actual_mb ?? resident.est_mb)} GB</span>
										</span>
									{:else if fit && !fit.gpu}
										<span class="muted" title="No GPU detected. This model runs on the CPU.">CPU only</span>
									{:else if fit}
										<span class="fit" style="--tone: {FIT_TONE[fit.verdict]}">{FIT_LABEL[fit.verdict]}</span>
									{:else if hasFitEstimate(m) && fitStore.error(m.id)}
										<span class="muted" title={fitStore.error(m.id) ?? undefined}>Check failed</span>
									{:else if hasFitEstimate(m)}
										<span class="muted">…</span>
									{:else if isLoadable(m)}
										<span class="muted" title="Loads on first use in Studio">On demand</span>
									{:else}
										<span class="muted" title={m.caps.reason ?? undefined}>Can't run</span>
									{/if}
								</td>
								<td class="r">
									{#if !isLoadable(m)}
										<span></span>
									{:else if resident}
										{@const lock = unloadLock(resident)}
										<button
											type="button"
											class="rbtn"
											disabled={busyId === m.id || lock != null}
											title={lock?.reason}
											onclick={(e) => { e.stopPropagation(); void unload(m); }}
										>
											{busyId === m.id ? 'Unloading…' : 'Unload'}
										</button>
									{:else if hasFitEstimate(m) && fitStore.error(m.id)}
										<button
											type="button"
											class="rbtn"
											title={engineBlock(m) ?? undefined}
											onclick={(e) => { e.stopPropagation(); void retryFit(m.id, defaultCtx(m)); }}
										>
											Retry check
										</button>
									{:else}
										<button
											type="button"
											class="rbtn primary"
											disabled={loading || engineBlock(m) !== null}
											title={engineBlock(m) ?? undefined}
											onclick={(e) => { e.stopPropagation(); void quickLoad(m); }}
										>
											{loading ? 'Loading…' : 'Load'}
										</button>
									{/if}
								</td>
								<td>
									<button
										type="button"
										class="more"
										aria-label="More actions for {m.display_name}"
										aria-haspopup="menu"
										aria-expanded={menuAnchor != null && menuModel?.id === m.id}
										onclick={(e) => openMenu(e, m)}
									>
										<Icon name="more" size={16} />
									</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</div>

		<ModelInspector
			model={selected}
			busy={busyId === selected?.id}
			onload={(m) => void quickLoad(m)}
			onunload={(m) => void unload(m)}
			onoptions={openOptions}
			onchat={(m) => void useInChat(m)}
			onstudio={useInStudio}
			onreveal={(m) => void reveal(m)}
			ondelete={(m) => (deleteTarget = m)}
		/>
	</div>
</div>

<style>
	.body {
		flex: 1;
		min-height: 0;
		display: grid;
		grid-template-columns: minmax(0, 1fr) 360px;
	}
	.table-wrap {
		min-width: 0;
		overflow: auto;
	}
	.empty {
		display: flex;
		height: 100%;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 8px;
		padding: 48px 24px;
		text-align: center;
	}
	.r {
		text-align: right;
	}
	.muted {
		color: var(--muted-foreground);
	}
	.name {
		max-width: 280px;
		overflow: hidden;
		text-overflow: ellipsis;
		font-weight: 500;
		color: var(--foreground);
	}
	.sdot {
		display: block;
		width: 7px;
		height: 7px;
		margin: 0 auto;
		border-radius: 50%;
		background: var(--muted-foreground);
		opacity: 0.35;
	}
	.sdot.on {
		background: var(--success);
		opacity: 1;
		box-shadow: 0 0 0 3px color-mix(in oklch, var(--success) 18%, transparent);
	}
	.sdot.busy {
		background: var(--primary);
		opacity: 1;
		animation: row-pulse 1s ease-in-out infinite;
	}
	@keyframes row-pulse {
		50% {
			opacity: 0.35;
		}
	}
	.fit {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		white-space: nowrap;
		color: var(--tone);
		font-weight: 500;
	}
	.fit::before {
		content: '';
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--tone);
	}
	.rbtn {
		height: 26px;
		min-width: 64px;
		padding: 0 10px;
		border-radius: 6px;
		border: 1px solid var(--border);
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--foreground);
		white-space: nowrap;
	}
	.rbtn.primary {
		border-color: color-mix(in oklch, var(--primary) 55%, var(--border));
		color: var(--primary);
		background: color-mix(in oklch, var(--primary) 10%, var(--elevated));
	}
	.rbtn:hover:not(:disabled) {
		border-color: var(--primary);
	}
	.rbtn:active:not(:disabled) {
		transform: translateY(1px);
	}
	.rbtn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.more {
		display: grid;
		place-items: center;
		width: 28px;
		height: 28px;
		border-radius: 6px;
		color: var(--muted-foreground);
	}
	.more:hover,
	.more[aria-expanded='true'] {
		background: var(--muted);
		color: var(--foreground);
	}
	.rbtn:focus-visible,
	.more:focus-visible,
	.tab:focus-visible,
	.store-btn:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
	}
	.tabs {
		display: flex;
		gap: 2px;
		padding: 2px;
		border-radius: 8px;
		background: var(--muted);
	}
	.tab {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		height: 26px;
		padding: 0 10px;
		border-radius: 6px;
		font-size: var(--text-xs);
		color: var(--muted-foreground);
		white-space: nowrap;
	}
	.tab:hover {
		color: var(--foreground);
	}
	.tab[aria-selected='true'] {
		background: var(--elevated);
		color: var(--foreground);
		box-shadow: inset 0 1px 0 var(--inset-highlight), 0 1px 2px oklch(0 0 0 / 0.2);
	}
	.tab .c {
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.search {
		display: flex;
		align-items: center;
		gap: 6px;
		height: 30px;
		width: 190px;
		padding: 0 10px;
		border-radius: 7px;
		border: 1px solid var(--border);
		background: var(--background);
	}
	.search:focus-within {
		border-color: var(--primary);
	}
	.search input {
		flex: 1;
		min-width: 0;
		background: transparent;
		font-size: var(--text-sm);
		color: var(--foreground);
		outline: none;
	}
	.search input::placeholder {
		color: var(--muted-foreground);
	}
	.store-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		height: 30px;
		padding: 0 12px;
		border-radius: 7px;
		background: var(--button-primary-gradient);
		color: var(--primary-foreground);
		font-size: var(--text-xs);
		font-weight: 500;
		white-space: nowrap;
	}
	.store-btn:hover {
		filter: brightness(1.06);
	}
	.store-btn.ghost {
		background: var(--elevated);
		border: 1px solid var(--border);
		color: var(--foreground);
	}
	@media (max-width: 1279px) {
		.opt {
			display: none;
		}
		.tab .c {
			display: none;
		}
		.search {
			width: 150px;
		}
	}
	/* Beside the 360px inspector at 1024 wide the table has ~590px: keep the
	   fit and Load columns visible; params and task live in the inspector. */
	@media (max-width: 1179px) {
		.nar {
			display: none;
		}
		.name {
			max-width: 160px;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.sdot.busy {
			animation: none;
		}
	}
</style>
