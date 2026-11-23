<script lang="ts">
	/**
	 * Store — Hugging Face browser (Workbench layout).
	 *
	 *   page header: title, search, format, sort      | repo inspector
	 *   task tabs                                     | (400px): about,
	 *   picks for your PC (connected cells)           | best quant, quant
	 *   dense results table, pager                    | list, download
	 *
	 * Filters are persisted across restarts via tauri-plugin-store under the
	 * key `browse.filters` in `settings.json`.
	 */
	import { onMount } from 'svelte';
	import { LazyStore } from '@tauri-apps/plugin-store';

	import PageHeader from '$lib/components/shell/PageHeader.svelte';
	import Icon from '$lib/components/shell/Icon.svelte';
	import DetailPane from '$lib/components/browse/DetailPane.svelte';
	import CuratedRail from '$lib/components/browse/CuratedRail.svelte';
	import RunnableBadge from '$lib/components/browse/RunnableBadge.svelte';
	import { TASK_TABS, taskLabel } from '$lib/components/browse/task-meta';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import { compactCount, timeAgo } from '$lib/format';

	import { hfSearch, storeRunnability } from '$lib/ipc/commands';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { openDownloads } from '$lib/stores/ui.svelte';
	import type { CuratedPick, HfModelCard, HfSearchResponse, Runnability } from '$lib/types';

	// ---------------------------------------------------------------------------
	// Persisted filter state
	// ---------------------------------------------------------------------------

	const store = new LazyStore('settings.json', { defaults: {}, autoSave: true });
	const FILTER_KEY = 'browse.filters';

	interface FilterState {
		query: string;
		/** Active task tab — raw HF pipeline_tag, or 'any'. */
		task: string;
		library: string;
		sort: string;
		hasGguf: boolean;
	}

	const DEFAULT_FILTERS: FilterState = {
		query: '',
		task: 'any',
		library: 'any',
		sort: 'trending',
		hasGguf: false,
	};

	// Svelte 5 rune state — individual so each filter change is tracked
	let query = $state(DEFAULT_FILTERS.query);
	let task = $state(DEFAULT_FILTERS.task);
	let library = $state(DEFAULT_FILTERS.library);
	let sort = $state(DEFAULT_FILTERS.sort);
	let hasGguf = $state(DEFAULT_FILTERS.hasGguf);

	// ---------------------------------------------------------------------------
	// Search result state
	// ---------------------------------------------------------------------------

	let results = $state<HfModelCard[]>([]);
	let page = $state(1);
	let hasNext = $state(false);
	let loading = $state(false);
	let initialLoad = $state(true);
	let error = $state<string | null>(null);

	// Selected model for the detail pane
	let selectedModel = $state<HfModelCard | null>(null);

	// Card-level runnability verdicts keyed by repo id (M9 §9.5).
	let runnabilityMap = $state<Map<string, Runnability>>(new Map());

	/** Batch-resolve runnability for one page of results. */
	async function resolveRunnability(cards: HfModelCard[]) {
		if (cards.length === 0) {
			runnabilityMap = new Map();
			return;
		}
		try {
			const verdicts = await storeRunnability(
				cards.map((c) => ({
					repo_id: c.id,
					task: c.pipeline_tag,
					library: c.library_name,
					tags: c.tags,
				})),
			);
			const next = new Map<string, Runnability>();
			cards.forEach((c, i) => {
				if (verdicts[i]) next.set(c.id, verdicts[i]);
			});
			runnabilityMap = next;
		} catch {
			// Badge-less rows are fine — never block the list on this.
			runnabilityMap = new Map();
		}
	}

	// ---------------------------------------------------------------------------
	// Derived: effective library param (hasGguf forces library=gguf)
	// ---------------------------------------------------------------------------

	const effectiveLibrary = $derived(hasGguf ? 'gguf' : library);

	// ---------------------------------------------------------------------------
	// Search execution
	// ---------------------------------------------------------------------------

	async function runSearch(targetPage = 1) {
		loading = true;
		error = null;
		try {
			const response: HfSearchResponse = await hfSearch({
				query: query.trim() || undefined,
				task: task === 'any' ? undefined : task,
				library: effectiveLibrary === 'any' ? undefined : effectiveLibrary,
				sort,
				page: targetPage,
				per_page: 30,
			});
			results = response.results;
			page = response.page;
			hasNext = response.has_next;
			initialLoad = false;
			void resolveRunnability(response.results);

			// Auto-select the first result if nothing is selected, or if the
			// current selection no longer exists in the result set.
			if (results.length > 0) {
				if (!selectedModel || !results.find((r) => r.id === selectedModel?.id)) {
					selectedModel = results[0];
				}
			} else {
				selectedModel = null;
			}
		} catch (err: unknown) {
			const msg = String(err);
			if (msg.includes('rate-limited') || msg.includes('RateLimited')) {
				addToast('HuggingFace rate-limited us — try again in a moment', 'warning');
			} else {
				addToast(`Search failed: ${msg}`, 'error');
			}
			error = msg;
			initialLoad = false;
		} finally {
			loading = false;
		}
	}

	// ---------------------------------------------------------------------------
	// Filter persistence helpers
	// ---------------------------------------------------------------------------

	async function saveFilters() {
		await store.set(FILTER_KEY, { query, task, library, sort, hasGguf });
	}

	async function loadFilters() {
		try {
			const saved = await store.get<FilterState>(FILTER_KEY);
			if (saved) {
				query = saved.query ?? DEFAULT_FILTERS.query;
				// Pre-M9 saved filters had `capability` instead of `task` — ignore it.
				task = saved.task ?? DEFAULT_FILTERS.task;
				library = saved.library ?? DEFAULT_FILTERS.library;
				sort = saved.sort ?? DEFAULT_FILTERS.sort;
				hasGguf = saved.hasGguf ?? DEFAULT_FILTERS.hasGguf;
			}
		} catch {
			// Not fatal — use defaults
		}
	}

	// ---------------------------------------------------------------------------
	// Filter change handlers (save + re-search)
	// ---------------------------------------------------------------------------

	function handleQueryChange(val: string) {
		query = val;
		page = 1;
		saveFilters();
		runSearch(1);
	}

	function handleTaskChange(val: string) {
		task = val;
		page = 1;
		saveFilters();
		runSearch(1);
	}

	/** A curated pick was clicked — search for its repo so the normal
	 * row → detail → download flow takes over. */
	function handlePick(pick: CuratedPick) {
		query = pick.repo_id;
		page = 1;
		saveFilters();
		runSearch(1);
	}

	function handleLibraryChange(val: string) {
		// Selecting an explicit library implies opting out of the GGUF-only shortcut.
		if (val !== 'gguf' && hasGguf) hasGguf = false;
		library = val;
		page = 1;
		saveFilters();
		runSearch(1);
	}

	function handleSortChange(val: string) {
		sort = val;
		page = 1;
		saveFilters();
		runSearch(1);
	}

	function resetFilters() {
		query = DEFAULT_FILTERS.query;
		task = DEFAULT_FILTERS.task;
		library = DEFAULT_FILTERS.library;
		sort = DEFAULT_FILTERS.sort;
		hasGguf = DEFAULT_FILTERS.hasGguf;
		page = 1;
		saveFilters();
		runSearch(1);
	}

	// ---------------------------------------------------------------------------
	// Pagination
	// ---------------------------------------------------------------------------

	function prevPage() {
		if (page <= 1) return;
		runSearch(page - 1);
	}

	function nextPage() {
		if (!hasNext) return;
		runSearch(page + 1);
	}

	// ---------------------------------------------------------------------------
	// Row click → select model in the detail pane
	// ---------------------------------------------------------------------------

	function selectModel(model: HfModelCard) {
		selectedModel = model;
	}

	// ---------------------------------------------------------------------------
	// Mount: load persisted filters then do initial search
	// ---------------------------------------------------------------------------

	onMount(async () => {
		await loadFilters();
		runSearch(1);
	});

	// Skeleton placeholder count for loading state
	const SKELETON_COUNT = 12;

	const SORT_OPTIONS = [
		{ value: 'trending', label: 'Trending' },
		{ value: 'downloads', label: 'Most downloaded' },
		{ value: 'likes', label: 'Most liked' },
		{ value: 'recent', label: 'Recently updated' },
	];
	const LIBRARY_OPTIONS = [
		{ value: 'any', label: 'Any format' },
		{ value: 'gguf', label: 'GGUF' },
		{ value: 'transformers', label: 'Transformers' },
		{ value: 'diffusers', label: 'Diffusers' },
		{ value: 'safetensors', label: 'Safetensors' },
	];

	// The search box is debounced; the bound value updates instantly.
	let debounce: ReturnType<typeof setTimeout> | null = null;
	function onQueryInput(e: Event) {
		query = (e.currentTarget as HTMLInputElement).value;
		if (debounce) clearTimeout(debounce);
		debounce = setTimeout(() => handleQueryChange(query), 350);
	}

	/** Split a repo id into author + name, marking the part that matches the query. */
	function idParts(id: string, q: string) {
		const slash = id.indexOf('/');
		const author = slash > 0 ? id.slice(0, slash + 1) : '';
		const name = slash > 0 ? id.slice(slash + 1) : id;
		const needle = q.trim().toLowerCase();
		const at = needle ? name.toLowerCase().indexOf(needle) : -1;
		if (at < 0) return { author, pre: name, hit: '', post: '' };
		return {
			author,
			pre: name.slice(0, at),
			hit: name.slice(at, at + needle.length),
			post: name.slice(at + needle.length),
		};
	}

	/** Arrow keys move the selection through the results table. */
	function onTableKey(e: KeyboardEvent) {
		if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
		if (results.length === 0) return;
		e.preventDefault();
		const i = results.findIndex((r) => r.id === selectedModel?.id);
		const next = e.key === 'ArrowDown' ? Math.min(results.length - 1, i + 1) : Math.max(0, i - 1);
		selectedModel = results[next];
		document.getElementById(`repo-${next}`)?.scrollIntoView({ block: 'nearest' });
	}
</script>

<div class="store">
	<section class="main">
		<PageHeader title="Store">
			{#snippet actions()}
				<label class="wb-search search">
					<Icon name="search" size={14} />
					<input
						type="search"
						value={query}
						oninput={onQueryInput}
						placeholder="Search models"
						aria-label="Search models on Hugging Face"
					/>
				</label>
				<select
					class="wb-input"
					aria-label="File format"
					value={hasGguf ? 'gguf' : library}
					onchange={(e) => handleLibraryChange((e.currentTarget as HTMLSelectElement).value)}
				>
					{#each LIBRARY_OPTIONS as o (o.value)}
						<option value={o.value}>{o.label}</option>
					{/each}
				</select>
				<select
					class="wb-input"
					aria-label="Sort"
					value={sort}
					onchange={(e) => handleSortChange((e.currentTarget as HTMLSelectElement).value)}
				>
					{#each SORT_OPTIONS as o (o.value)}
						<option value={o.value}>{o.label}</option>
					{/each}
				</select>
			{/snippet}
		</PageHeader>

		<div class="wb-tabs" role="tablist" aria-label="Model type">
			{#each TASK_TABS as t (t.value)}
				<button
					type="button"
					role="tab"
					class="wb-tab"
					aria-selected={task === t.value}
					title={t.description}
					onclick={() => handleTaskChange(t.value)}
				>
					{t.label}
				</button>
			{/each}
			<span class="tab-meta">
				{#if loading && !initialLoad}<Spinner size="sm" />{/if}
				{#if !initialLoad && results.length > 0}
					<span><span class="num">{results.length}</span> results</span>
				{/if}
			</span>
		</div>

		<div class="scroll">
			<CuratedRail {task} onpick={handlePick} />

			{#if initialLoad && loading}
				<table class="wb-table" aria-hidden="true">
					<tbody>
						{#each { length: SKELETON_COUNT } as _, i (i)}
							<tr>
								<td><span class="sk" style="width: {40 + ((i * 17) % 35)}%"></span></td>
								<td style="width: 120px"><span class="sk" style="width: 70%"></span></td>
								<td style="width: 110px"><span class="sk" style="width: 60%"></span></td>
							</tr>
						{/each}
					</tbody>
				</table>
			{:else if error && results.length === 0}
				<div class="wb-empty">
					<b>Couldn't reach Hugging Face</b>
					<span class="max-w-md">{error}</span>
					<button type="button" class="wb-btn mt-2" onclick={() => runSearch(page)}>Try again</button>
				</div>
			{:else if !loading && results.length === 0}
				<div class="wb-empty">
					<b>No results</b>
					<span>Try a shorter search or another tab.</span>
					<button type="button" class="wb-btn mt-2" onclick={resetFilters}>Reset filters</button>
				</div>
			{:else}
				<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
				<table
					class="wb-table results"
					class:dim={loading && !initialLoad}
					tabindex="0"
					aria-label="Search results. Use the arrow keys to move the selection."
					onkeydown={onTableKey}
				>
					<thead>
						<tr>
							<th>Repository</th>
							<th>Task</th>
							<th class="r">Downloads</th>
							<th class="r wide">Likes</th>
							<th class="wide">Updated</th>
							<th>On this PC</th>
						</tr>
					</thead>
					<tbody>
						{#each results as model, i (model.id)}
							{@const p = idParts(model.id, query)}
							<tr
								id="repo-{i}"
								aria-selected={selectedModel?.id === model.id}
								onclick={() => selectModel(model)}
							>
								<td class="repo" title={model.id}>
									<span class="author">{p.author}</span><span class="name"
										>{p.pre}{#if p.hit}<mark>{p.hit}</mark>{/if}{p.post}</span
									>
									{#if model.gated}
										<span
											class="wb-badge warn gated"
											title="Accept the licence on Hugging Face before downloading"
										>Gated</span>
									{/if}
								</td>
								<td class="muted">{taskLabel(model.pipeline_tag) ?? '—'}</td>
								<td class="r num" title="{model.downloads.toLocaleString()} downloads"
									>{compactCount(model.downloads)}</td
								>
								<td class="r num wide" title="{model.likes.toLocaleString()} likes"
									>{compactCount(model.likes)}</td
								>
								<td class="muted wide">{timeAgo(model.last_modified) ?? '—'}</td>
								<td><RunnableBadge verdict={runnabilityMap.get(model.id) ?? null} /></td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</div>

		{#if !initialLoad && (results.length > 0 || page > 1)}
			<footer class="pager">
				<span class="text-xs text-muted-foreground">Page <span class="num">{page}</span></span>
				<button type="button" class="wb-btn sm" disabled={page <= 1 || loading} onclick={prevPage}>
					Previous
				</button>
				<button type="button" class="wb-btn sm" disabled={!hasNext || loading} onclick={nextPage}>
					Next
				</button>
			</footer>
		{/if}
	</section>

	<aside class="inspector" aria-label="Repository details">
		<DetailPane
			model={selectedModel}
			runnability={selectedModel ? (runnabilityMap.get(selectedModel.id) ?? null) : null}
			ondownloadqueued={openDownloads}
		/>
	</aside>
</div>

<style>
	.store {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 400px;
		height: 100%;
		min-height: 0;
		background: var(--background);
	}
	@media (max-width: 1199px) {
		.store {
			grid-template-columns: minmax(0, 1fr) 360px;
		}
	}
	.main {
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
	}
	.search {
		width: 280px;
	}
	@media (max-width: 1279px) {
		.search {
			width: 180px;
		}
	}
	.tab-meta {
		margin-left: auto;
		display: inline-flex;
		align-items: center;
		gap: 8px;
		padding-right: 4px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
		white-space: nowrap;
	}
	.scroll {
		flex: 1;
		min-height: 0;
		overflow: auto;
	}
	.results {
		outline: none;
		transition: opacity 0.15s ease;
	}
	.results:focus-visible tbody tr[aria-selected='true'] td {
		background: color-mix(in oklch, var(--primary) 14%, transparent);
	}
	.results.dim {
		opacity: 0.5;
	}
	.results tbody tr {
		cursor: pointer;
	}
	.repo {
		max-width: 0;
		width: 52%;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.author {
		color: var(--muted-foreground);
	}
	.name {
		font-weight: 500;
		color: var(--foreground);
	}
	.gated {
		margin-left: 6px;
	}
	mark {
		background: color-mix(in oklch, var(--primary) 22%, transparent);
		color: inherit;
		border-radius: 2px;
	}
	.r {
		text-align: right;
	}
	.muted {
		color: var(--muted-foreground);
	}
	.sk {
		display: block;
		height: 10px;
		border-radius: 4px;
		background: var(--elevated);
		animation: pulse 1.4s ease-in-out infinite;
	}
	@keyframes pulse {
		50% {
			opacity: 0.45;
		}
	}
	.pager {
		display: flex;
		flex-shrink: 0;
		align-items: center;
		justify-content: flex-end;
		gap: 8px;
		height: 44px;
		padding: 0 16px;
		border-top: 1px solid var(--border);
	}
	.inspector {
		min-width: 0;
		min-height: 0;
		border-left: 1px solid var(--border);
		background: var(--surface);
	}
	@media (max-width: 1279px) {
		.wide {
			display: none;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.sk {
			animation: none;
		}
		.results {
			transition: none;
		}
	}
</style>
