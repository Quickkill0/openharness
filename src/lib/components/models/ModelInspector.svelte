<script lang="ts">
	/**
	 * Models inspector — 360px panel beside the table for the selected model.
	 *
	 * Shows the state, today's fit estimate (estimate_model_fit, via fitStore)
	 * as a graphics-memory bar, the load actions, and a ruled property list.
	 *
	 * PHASE C SLOT: the "Placement" section is the bounded home of the
	 * three-tier load planner (planner::auto_plan, VRAM/RAM/SSD placement and
	 * speed estimates) once that backend lands. Until then it states only what
	 * today's loader really does.
	 */
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import { ledgerStore } from '$lib/stores/ledger.svelte';
	import { engineStore } from '$lib/stores/engine.svelte';
	import { fitStore, retryFit } from '$lib/stores/fits.svelte';
	import { unloadLock } from '$lib/stores/chipLocks.svelte';
	import { formatBytes, gbFromMb, pct } from '$lib/format';
	import Icon from '$lib/components/shell/Icon.svelte';
	import {
		FORMAT_LABELS,
		LLAMA_NEEDED,
		RUNTIME_LABELS,
		WORKFLOW_LABELS,
		canOpenLoadOptions,
		ctxLabel,
		defaultCtx,
		family,
		hasFitEstimate,
		isLoadable,
		params,
		sizeOnDisk,
		studioTask,
		textEngineBlock,
	} from './modelMeta';
	import type { InstalledModel } from '$lib/types';

	interface Props {
		model: InstalledModel | null;
		busy: boolean;
		onload: (m: InstalledModel) => void;
		onunload: (m: InstalledModel) => void;
		onoptions: (m: InstalledModel) => void;
		onchat: (m: InstalledModel) => void;
		onstudio: (m: InstalledModel) => void;
		onreveal: (m: InstalledModel) => void;
		ondelete: (m: InstalledModel) => void;
	}

	let { model, busy, onload, onunload, onoptions, onchat, onstudio, onreveal, ondelete }: Props =
		$props();

	const resident = $derived(model ? ledgerStore.residentFor(model.id) : null);
	const lock = $derived(resident ? unloadLock(resident) : null);
	const loading = $derived(model ? ledgerStore.isLoading(model.id) : false);
	const fit = $derived(model && hasFitEstimate(model) ? fitStore.get(model.id) : null);
	const fitError = $derived(model ? fitStore.error(model.id) : null);
	/** The context the estimate (and a plain Load) uses for this model. */
	const ctx = $derived(model ? (fitStore.ctx(model.id) ?? defaultCtx(model)) : 0);
	/** No NVIDIA GPU: the backend reports gpu:false and llama.cpp runs on the CPU. */
	const cpuOnly = $derived(fit ? !fit.gpu : false);
	/** Why this model can't start with the engines installed, or null. */
	const engineBlock = $derived(
		model ? textEngineBlock(model, fit, fitError, engineStore.textEngines) : null,
	);
	const optionsOk = $derived(model ? canOpenLoadOptions(model, engineStore.textEngines) : false);
	/** The planner chose the native engine for this model. */
	const nativePlan = $derived(fit?.plan?.engine_kind === 'ssd_stream');

	// Graphics-memory bar: other apps | residents | this model.
	const total = $derived(gpuStore.vramTotal || ledgerStore.snapshot?.total_mb || 0);
	const residentSum = $derived(
		ledgerStore.residents.reduce((s, r) => s + (r.actual_mb ?? r.est_mb), 0),
	);
	const used = $derived(gpuStore.hasGpu ? gpuStore.vramUsed : residentSum);
	const other = $derived(Math.max(0, used - residentSum));
	const ownMb = $derived(resident ? (resident.actual_mb ?? resident.est_mb) : 0);
	// estimated_mb already includes the context cache when a ctx is passed.
	const needMb = $derived(fit ? fit.estimated_mb : 0);
	// A chat load swaps out the running llama.cpp model, which frees first.
	const swapped = $derived(
		!resident && fit && fit.reclaimable_mb > 0
			? (ledgerStore.residents.find((r) => r.kind === 'text_engine') ?? null)
			: null,
	);
	const swapMb = $derived(swapped ? (swapped.actual_mb ?? swapped.est_mb) : 0);

	const VERDICT: Record<string, { label: string; tone: string; text: string }> = {
		fits: {
			label: 'Fits',
			tone: 'var(--success)',
			text: 'Loads fully into graphics memory with room to spare.',
		},
		tight: {
			label: 'Tight',
			tone: 'var(--warning)',
			text: 'Fits, but with little headroom. Close other GPU apps or lower the context length.',
		},
		wont_fit: {
			label: "Won't fit",
			tone: 'var(--error)',
			text: 'Needs more graphics memory than is free. Unload something, or load with fewer GPU layers.',
		},
	};

	function installedOn(ts: number): string {
		return new Date(ts * 1000).toLocaleDateString(undefined, {
			year: 'numeric',
			month: 'short',
			day: 'numeric',
		});
	}

	const path = $derived(model ? (model.local_path ?? model.primary_file_path) : '');
</script>

<aside class="inspector" aria-label="Model details">
	{#if !model}
		<div class="flex h-full flex-col items-center justify-center gap-2 p-8 text-center">
			<Icon name="info" size={22} class="text-muted-foreground" />
			<p class="text-sm text-muted-foreground">Select a model to see its details and load options.</p>
		</div>
	{:else}
		<!-- Header -->
		<div class="sec head">
			<div class="flex items-center gap-2 text-2xs text-muted-foreground">
				{#if resident}
					<span class="dot on"></span><span>Loaded</span>
					<span class="num">· {gbFromMb(ownMb)} GB</span>
				{:else if loading}
					<span class="dot busy"></span><span>Loading…</span>
				{:else}
					<span class="dot off"></span><span>Not loaded</span>
				{/if}
			</div>
			<h2 class="mt-1.5 text-lg font-semibold leading-snug text-foreground [overflow-wrap:anywhere]">
				{model.display_name}
			</h2>
			<p class="mt-0.5 truncate text-xs text-muted-foreground" title={model.repo_id}>{model.repo_id}</p>

			<!-- Actions -->
			<div class="mt-3 flex flex-wrap gap-2">
				{#if !isLoadable(model)}
					<p class="text-xs text-muted-foreground">
						{model.caps.reason ?? 'Nothing in OpenHarness can run this model yet.'}
					</p>
				{:else if resident}
					<button
						type="button"
						class="btn"
						disabled={busy || lock != null}
						title={lock?.reason}
						onclick={() => onunload(model)}
					>
						{busy ? 'Unloading…' : lock?.kind === 'replacing' ? 'Being replaced' : 'Unload'}
					</button>
				{:else}
					<button
						type="button"
						class="btn primary"
						disabled={loading || busy || engineBlock !== null}
						title={engineBlock ?? undefined}
						onclick={() => onload(model)}
					>
						{loading ? 'Loading…' : 'Load'}
					</button>
					<button
						type="button"
						class="btn"
						disabled={loading || !optionsOk}
						onclick={() => onoptions(model)}
					>
						<Icon name="sliders" size={14} />
						Load with options
					</button>
				{/if}
				{#if model.caps.workflows.includes('chat')}
					<button type="button" class="btn ghost" onclick={() => onchat(model)}>Use in Chat</button>
				{/if}
				{#if studioTask(model)}
					<button type="button" class="btn ghost" onclick={() => onstudio(model)}>Use in Studio</button>
				{/if}
			</div>
			{#if lock}
				<p class="mt-2 text-2xs text-muted-foreground">{lock.reason}</p>
			{/if}
			{#if engineBlock && isLoadable(model) && !resident}
				<p class="mt-2 text-2xs text-muted-foreground">
					{#if engineBlock === LLAMA_NEEDED}
						Install the llama.cpp engine in <a class="link" href="/settings/engines">Settings, Engines</a> to load this model.
					{:else}
						{engineBlock}
						{#if fitError && hasFitEstimate(model)}
							<button type="button" class="wb-btn sm ml-1" onclick={() => retryFit(model.id, ctx)}>Retry</button>
						{/if}
					{/if}
				</p>
			{/if}
		</div>

		<!-- Fit -->
		<div class="sec">
			<h3 class="sec-title">{cpuOnly ? 'Memory' : 'Graphics memory'}</h3>
			{#if hasFitEstimate(model)}
				{#if fit && !fit.gpu}
					<!-- No NVIDIA GPU: llama.cpp runs on the CPU, so there is no
					     graphics memory to fill and nothing to preview. -->
					<p class="flex items-center gap-2 text-sm">
						<span class="verdict" style="--tone: {fit.verdict === 'wont_fit' ? 'var(--error)' : 'var(--muted-foreground)'}">{fit.verdict === 'wont_fit' ? "Won't fit" : 'CPU only'}</span>
					</p>
					<p class="mt-1.5 text-xs text-muted-foreground">
						{fit.plan?.infeasible_reason ?? (nativePlan ? 'No NVIDIA GPU was detected. Experts stream from the SSD and everything else uses system memory.' : 'No NVIDIA GPU was detected, so this model uses system memory. Replies are slower than on a graphics card.')}
					</p>
					<dl class="wb-props mt-3">
						<dt>Estimated memory use</dt>
						<dd class="num">{gbFromMb(fit.estimated_mb)} GB</dd>
						<dt>Context on Load</dt>
						<dd class="num">{ctxLabel(ctx)}</dd>
					</dl>
				{:else if fit}
					{@const p = fit.plan}
					{@const v = p ? {
						label: !p.feasible ? "Won't fit" : p.tier_mix === 'ssd' ? 'Uses SSD' : p.tier_mix === 'ram' ? 'Uses RAM' : 'Fits',
						tone: !p.feasible ? 'var(--error)' : 'var(--success)',
						text: p.infeasible_reason ?? (p.engine_kind === 'ssd_stream' ? (p.gpu_layers !== 0 ? 'Streams experts from SSD. Dense layers and context run in graphics memory beside the expert tier.' : 'Streams experts from SSD. Dense layers, prefill and context stay in system memory; graphics memory holds only the optional expert tier.') :'The planned placement uses graphics memory, system memory and SSD as shown below.'),
					} : VERDICT[fit.verdict]}
					<div class="tier-bar big" role="img" aria-label="Graphics memory after loading">
						{#if total > 0}
							<i style="width: {pct(other, total)}; background: var(--tier-base)" title="Other apps"></i>
							<i style="width: {pct(residentSum - ownMb - swapMb, total)}; background: color-mix(in oklch, var(--tier-vram) 55%, transparent)" title="Other loaded models"></i>
							{#if resident}
								<i style="width: {pct(ownMb, total)}; background: var(--tier-vram)" title="This model"></i>
							{:else}
								<i class="ghost" style="width: {pct(Math.min(needMb, Math.max(0, total - (used - swapMb))), total)}; --c: {fit.verdict === 'wont_fit' ? 'var(--error)' : 'var(--tier-vram)'}" title="This model (estimate)"></i>
							{/if}
						{/if}
					</div>
					{#if !resident}
						<p class="mt-2.5 flex items-center gap-2 text-sm">
							<span class="verdict" style="--tone: {v.tone}">{v.label}</span>
						</p>
						<p class="mt-1.5 text-xs text-muted-foreground">{v.text}</p>
					{/if}
					<dl class="wb-props mt-3">
						<dt>{p?.engine_kind === 'ssd_stream' && p.gpu_layers === 0 ? 'Expert tier and overhead' : 'Weights and overhead'}</dt>
						<dd class="num">{gbFromMb(fit.estimated_mb - fit.kv_at_ctx_mb)} GB</dd>
						<dt title="A plain Load uses the trained context, capped at 8K">Context cache at {ctxLabel(ctx)}</dt>
						<dd class="num">{gbFromMb(fit.kv_at_ctx_mb)} GB</dd>
						<dt>Free now</dt>
						<dd class="num">{gbFromMb(fit.free_mb)} GB</dd>
						{#if swapped}
							<dt title="Held by {swapped?.label ?? 'the loaded text model'}, which this load replaces">Freed by swapping</dt>
							<dd class="num">{gbFromMb(fit.reclaimable_mb)} GB</dd>
						{/if}
						{#if total > 0}
							<dt>Card total</dt>
							<dd class="num">{gbFromMb(total)} GB</dd>
						{/if}
					</dl>
				{:else if fitError}
					<p class="text-xs text-muted-foreground">Couldn't estimate the fit.</p>
				{:else}
					<div class="tier-bar big shimmer" aria-hidden="true"></div>
					<p class="mt-2 text-xs text-muted-foreground">Estimating…</p>
				{/if}
			{:else if resident}
				<dl class="wb-props">
					<dt>In use</dt>
					<dd class="num">{gbFromMb(ownMb)} GB</dd>
				</dl>
			{:else if isLoadable(model)}
				<p class="text-xs text-muted-foreground">
					Studio models load on first use and report their real memory use once loaded.
				</p>
			{:else}
				<p class="text-xs text-muted-foreground">Not applicable.</p>
			{/if}
		</div>

		<!-- Placement: PHASE C SLOT for the three-tier planner -->
		{#if hasFitEstimate(model)}
			<div class="sec" data-slot="load-planner">
				<h3 class="sec-title">Placement</h3>
				{#if fit?.plan}
					<p class="text-xs text-muted-foreground">
						{gbFromMb((fit.plan.tiers.vram_weights + fit.plan.tiers.vram_expert_cache + fit.plan.tiers.vram_kv + fit.plan.tiers.vram_overhead) / 1024 ** 2)} GB graphics memory,
						{gbFromMb(fit.plan.tiers.ram / 1024 ** 2)} GB system memory{#if fit.plan.native}{' '}(including a {gbFromMb(fit.plan.native.expert_cache_mib)} GB expert cache){/if},
						{gbFromMb(fit.plan.tiers.ssd / 1024 ** 2)} GB on SSD.
						{fit.plan.engine_kind !== 'ssd_stream' ? '' : fit.plan.gpu_layers !== 0 ? 'Dense layers and context run in graphics memory.' : 'Dense layers, prefill and context run on the CPU.'}
						<button type="button" class="link" onclick={() => onoptions(model)}>Load with options</button> changes the plan.
					</p>
				{:else if cpuOnly}
					<div class="flex items-start gap-2.5 text-xs">
						<span class="swatch mt-0.5" style="background: var(--tier-ram)"></span>
						<p class="text-muted-foreground">
							Every layer runs from system memory on the CPU. A plain Load uses a {ctxLabel(ctx)}
							context; <button type="button" class="link" onclick={() => onoptions(model)}>Load with options</button> changes it.
						</p>
					</div>
				{:else}
					<div class="flex items-start gap-2.5 text-xs">
						<span class="swatch mt-0.5" style="background: var(--tier-vram)"></span>
						<p class="text-muted-foreground">
							A plain Load puts every layer in graphics memory with a {ctxLabel(ctx)} context. To keep
							some layers in system memory, use <button type="button" class="link" onclick={() => onoptions(model)}>Load with options</button> and lower the GPU layers.
						</p>
					</div>
				{/if}
			</div>
		{/if}

		<!-- Details -->
		<div class="sec">
			<h3 class="sec-title">Details</h3>
			<dl class="wb-props">
				<dt>Family</dt>
				<dd>{family(model)}</dd>
				<dt>Parameters</dt>
				<dd class="num">{params(model)}</dd>
				<dt>Quantization</dt>
				<dd class="num">{model.quantization ?? '—'}</dd>
				<dt>Format</dt>
				<dd>{FORMAT_LABELS[model.format]}</dd>
				<dt>Works with</dt>
				<dd>
					{model.caps.workflows.length
						? model.caps.workflows.map((w) => WORKFLOW_LABELS[w]).join(', ')
						: 'Nothing yet'}
				</dd>
				<dt>Runs on</dt>
				<dd>{nativePlan ? 'Native engine (SSD streaming)' : model.caps.runtime ? RUNTIME_LABELS[model.caps.runtime] : '—'}</dd>
				{#if model.context_train}
					<dt>Trained context</dt>
					<dd class="num">{(model.context_train / 1024).toFixed(0)}K</dd>
				{/if}
				<dt>Size on disk</dt>
				<dd class="num">{formatBytes(sizeOnDisk(model))}</dd>
				{#if model.file_count > 1}
					<dt>Files</dt>
					<dd class="num">{model.file_count}</dd>
				{/if}
				<dt>Installed</dt>
				<dd>{installedOn(model.installed_at)}</dd>
			</dl>
			<p class="path" title={path}>{path}</p>
		</div>

		<!-- Housekeeping -->
		<div class="sec flex gap-2 border-b-0">
			<button type="button" class="btn ghost" onclick={() => onreveal(model)}>
				<Icon name="folder" size={14} />
				Show in Explorer
			</button>
			<button type="button" class="btn ghost danger" onclick={() => ondelete(model)}>
				<Icon name="trash" size={14} />
				Delete
			</button>
		</div>
	{/if}
</aside>

<style>
	.inspector {
		height: 100%;
		overflow-y: auto;
		border-left: 1px solid var(--border);
		background: var(--surface);
	}
	.sec {
		padding: 14px 16px;
		border-bottom: 1px solid var(--border);
	}
	.sec-title {
		margin-bottom: 10px;
		font-size: var(--text-2xs);
		font-weight: 600;
		color: var(--muted-foreground);
	}
	.tier-bar.big {
		height: 18px;
		border-radius: 5px;
	}
	.tier-bar.big > i {
		border-radius: 3px;
	}
	.shimmer {
		background: linear-gradient(90deg, var(--muted) 0%, var(--elevated) 50%, var(--muted) 100%);
		background-size: 200% 100%;
		animation: shimmer 1.2s linear infinite;
	}
	@keyframes shimmer {
		to {
			background-position: -200% 0;
		}
	}
	.verdict {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-weight: 600;
		color: var(--tone);
	}
	.verdict::before {
		content: '';
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--tone);
	}
	.swatch {
		width: 10px;
		height: 10px;
		border-radius: 3px;
		flex: none;
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		flex: none;
	}
	.dot.on {
		background: var(--success);
		box-shadow: 0 0 0 3px color-mix(in oklch, var(--success) 18%, transparent);
	}
	.dot.off {
		background: var(--muted-foreground);
		opacity: 0.45;
	}
	.dot.busy {
		background: var(--primary);
		animation: insp-pulse 1s ease-in-out infinite;
	}
	@keyframes insp-pulse {
		50% {
			opacity: 0.35;
		}
	}
	.path {
		margin-top: 10px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		direction: rtl;
		text-align: left;
	}
	.link {
		color: var(--primary);
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		height: 30px;
		padding: 0 12px;
		border-radius: 7px;
		border: 1px solid var(--border);
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--foreground);
		white-space: nowrap;
	}
	.btn:hover:not(:disabled) {
		border-color: color-mix(in oklch, var(--primary) 45%, var(--border));
	}
	.btn:active:not(:disabled) {
		transform: translateY(1px);
	}
	.btn:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btn.primary {
		border-color: transparent;
		background: var(--button-primary-gradient);
		color: var(--primary-foreground);
	}
	.btn.ghost {
		background: transparent;
		box-shadow: none;
		border-color: transparent;
		color: var(--muted-foreground);
	}
	.btn.ghost:hover:not(:disabled) {
		background: var(--elevated);
		color: var(--foreground);
		border-color: var(--border);
	}
	.btn.danger:hover:not(:disabled) {
		color: var(--error);
	}
	@media (prefers-reduced-motion: reduce) {
		.shimmer,
		.dot.busy {
			animation: none;
		}
	}
</style>
