<script lang="ts">
	import { onMount } from 'svelte';
	import { listLocalModels } from '$lib/ipc/commands';
	import { initLedgerStore, ledgerStore } from '$lib/stores/ledger.svelte';
	import type { InstalledModel } from '$lib/types';

	interface Props {
		/** HF pipeline_tag to filter installed models by ('text-to-image', …). */
		hfTask: string;
		/** When set, filter by GGUF architecture instead of task (e.g. 'qwen3-tts'). */
		archFilter?: string;
		/** Capability-record filter (M15) — overrides task/arch filtering when set. */
		capsFilter?: (m: InstalledModel) => boolean;
		/** What the option value carries — engine loads need the catalog id. */
		valueKind?: 'path' | 'id';
		/** Selected model path — '' means the default option (when offered). */
		value?: string;
		label?: string;
		/** Override for the "nothing installed" hint (defaults to the hfTask text). */
		emptyHint?: string;
		/**
		 * Built-in engine fallbacks (e.g. Whisper sizes, Kokoro 82M) — not
		 * catalog models, always available, shown grouped below installed models.
		 */
		builtinOptions?: { value: string; label: string }[];
		/** Optgroup header for the built-ins (keep it honest about auto-download). */
		builtinGroupLabel?: string;
		onchange?: (value: string) => void;
		/** Fires with the resolved catalog model on change (null = built-in option). */
		onmodel?: (model: InstalledModel | null) => void;
	}

	let {
		hfTask,
		archFilter,
		capsFilter,
		valueKind = 'path',
		value = $bindable(''),
		label = 'Model',
		emptyHint,
		builtinOptions = [],
		builtinGroupLabel = 'Built-in',
		onchange,
		onmodel,
	}: Props = $props();

	let models = $state<InstalledModel[]>([]);
	let loaded = $state(false);

	/** Dir-layout models load from local_path; single-blob from the file itself. */
	function pathOf(m: InstalledModel): string {
		return m.local_path ?? m.primary_file_path;
	}

	function valueOf(m: InstalledModel): string {
		return valueKind === 'id' ? m.id : pathOf(m);
	}

	// Loaded-first (M16 W1): every installed model matching the filter is
	// selectable — VRAM-resident ones (ledger store) sort on top with a ●
	// marker, unloaded ones lazy-load on first generate (the worker/orchestrator
	// pays the load cost at submit; preload is an accelerator, never a gate).
	// Built-ins (Whisper sizes, Kokoro 82M) are engine features, not catalog
	// models — they group below so Studio works with zero installs.
	// Reactive against the ledger, so a model loaded in /models reorders live.
	const sortedModels = $derived(
		[...models].sort(
			(a, b) => Number(ledgerStore.isLoaded(b.id)) - Number(ledgerStore.isLoaded(a.id)),
		),
	);
	const options = $derived([
		...sortedModels.map((m) => ({ value: valueOf(m) })),
		...builtinOptions.map((o) => ({ value: o.value })),
	]);

	// Spec'd hint (M16 W1): unloaded installed picks pay the load cost on submit.
	const selectedModel = $derived(models.find((mm) => valueOf(mm) === value) ?? null);
	const selectedLoaded = $derived(selectedModel ? ledgerStore.isLoaded(selectedModel.id) : false);
	const selectedUnloaded = $derived(selectedModel !== null && !selectedLoaded);

	const pickerId = `mp-${Math.random().toString(36).slice(2, 8)}`;

	function emit(v: string) {
		onchange?.(v);
		onmodel?.(models.find((m) => valueOf(m) === v) ?? null);
	}

	// Keep the selection valid against the live option set. Installed models
	// stay valid whether loaded or not (only uninstalling invalidates them),
	// so this fires once on mount to resolve the initial value and again only
	// when the selection truly disappears: fall back to the first built-in,
	// else the first installed model.
	$effect(() => {
		if (!loaded) return;
		const valid = options.some((o) => o.value === value);
		if (valid) {
			onmodel?.(models.find((m) => valueOf(m) === value) ?? null);
			return;
		}
		const next =
			builtinOptions[0]?.value ?? (sortedModels[0] ? valueOf(sortedModels[0]) : '');
		if (value !== next) {
			value = next;
			emit(next);
		}
	});

	function matches(m: InstalledModel): boolean {
		return capsFilter ? capsFilter(m) : archFilter ? m.architecture === archFilter : m.task === hfTask;
	}

	/**
	 * Re-read the catalog and return the picker's option list. Callers that
	 * restore a saved selection (Studio remix) decide "still installed" against
	 * this list, so the picker never rejects a model the caller just accepted.
	 * On failure the current list stays and is returned.
	 *
	 * Reads can resolve out of order (mount read vs a remix, or two quick
	 * remixes), so only the newest request writes `models`. An older request
	 * that lands late is dropped and hands back the newest request's result,
	 * so every caller sees the list the picker actually ends up with.
	 */
	let refreshSeq = 0;
	let latestRefresh: Promise<InstalledModel[]> = Promise.resolve([]);

	export function refresh(): Promise<InstalledModel[]> {
		const seq = ++refreshSeq;
		const request = (async (): Promise<InstalledModel[]> => {
			let list: InstalledModel[] | null = null;
			try {
				list = (await listLocalModels()).filter(matches);
			} catch {
				// Empty picker + hint below is the failure UI.
			}
			if (seq !== refreshSeq) return latestRefresh;
			if (list) models = list;
			loaded = true;
			return models;
		})();
		latestRefresh = request;
		return request;
	}

	onMount(() => {
		void initLedgerStore();
		void refresh();
	});
</script>

<div class="st-field mp">
	<label for={pickerId} class="st-label">
		{label}
		{#if selectedLoaded}
			<span class="mp-state"><span class="wb-dot" style="--tone: var(--tier-vram)">Loaded</span></span>
		{:else if selectedUnloaded}
			<span class="mp-state">Loads on first run</span>
		{/if}
	</label>
	{#if options.length > 0}
		<select id={pickerId} class="wb-input" bind:value onchange={() => emit(value)}>
			{#if builtinOptions.length === 0 && !value}
				<option value="" disabled>Select a model…</option>
			{/if}
			{#if builtinOptions.length > 0}
				<optgroup label="Installed">
					{#each sortedModels as m (m.id)}
						<option value={valueOf(m)}>{ledgerStore.isLoaded(m.id) ? `● ${m.display_name}` : m.display_name}</option>
					{/each}
				</optgroup>
				<optgroup label={builtinGroupLabel}>
					{#each builtinOptions as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
				</optgroup>
			{:else}
				{#each sortedModels as m (m.id)}
					<option value={valueOf(m)}>{ledgerStore.isLoaded(m.id) ? `● ${m.display_name}` : m.display_name}</option>
				{/each}
			{/if}
		</select>
	{:else if loaded}
		<a href="/models/browse" class="mp-empty">
			<b>No {emptyHint ?? hfTask.replace(/-/g, ' ')} models installed</b>
			<span>Find one in the Store</span>
		</a>
	{:else}
		<div class="mp-skel" aria-hidden="true"></div>
	{/if}
</div>

<style>
	.mp-state {
		font-size: var(--text-2xs);
		font-weight: 400;
		color: var(--muted-foreground);
	}
	.mp-empty {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 10px 12px;
		border: 1px dashed var(--border);
		border-radius: 8px;
		font-size: var(--text-xs);
		color: var(--muted-foreground);
		transition:
			border-color 0.15s ease,
			color 0.15s ease;
	}
	.mp-empty b {
		font-weight: 500;
		color: var(--foreground);
	}
	.mp-empty span {
		color: var(--primary);
	}
	.mp-empty:hover {
		border-color: var(--ring);
	}
	.mp-empty:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 1px;
	}
	.mp-skel {
		height: 34px;
		border-radius: 8px;
		background: var(--elevated);
		animation: mp-pulse 1.4s ease-in-out infinite;
	}
	@keyframes mp-pulse {
		50% {
			opacity: 0.5;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.mp-skel {
			animation: none;
		}
	}
</style>
