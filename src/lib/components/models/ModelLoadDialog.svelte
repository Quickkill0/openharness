<script lang="ts">
	/**
	 * ModelLoadDialog — the single load path for installed models (M16 W1,
	 * folded into one flow post-W1): the Models table's Load button opens it
	 * for every loadable model.
	 *
	 * llama-server GGUFs get the full tuning form (context, offload, batch,
	 * KV cache, memory flags) with the load planner's live plan: where the
	 * weights go (graphics memory / system memory / SSD) and the expected
	 * speed. The context slider starts at the planner's auto choice, and a
	 * blank GPU-layers field lets the planner place the weights, so
	 * "open → Load" is the one-click auto load. Typing a GPU-layer count
	 * switches to the manual (legacy -ngl) path. The Engine select picks the
	 * launcher (Auto / llama.cpp / Native) and re-plans. Worker pipelines and
	 * Qwen3-TTS talkers have no load-time knobs, so the dialog shows a
	 * summary card and loads through the unified ledger flow.
	 */
	import Modal from '$lib/components/ui/Modal.svelte';
	import Button from '$lib/components/ui/Button.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import Select from '$lib/components/ui/Select.svelte';
	import Toggle from '$lib/components/ui/Toggle.svelte';
	import Input from '$lib/components/ui/Input.svelte';
	import FitBadge from '$lib/components/engine/FitBadge.svelte';
	import { engineStore, loadModel as loadTextModel } from '$lib/stores/engine.svelte';
	import { LLAMA_NEEDED, PLAN_PENDING, textEngineBlock } from '$lib/components/models/modelMeta';
	import { loadModel as loadByLedger } from '$lib/stores/ledger.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { planLoad } from '$lib/ipc/commands';
	import { untrack } from 'svelte';
	import type {
		EngineKind,
		FitEstimate,
		InstalledModel,
		KvType,
		LoadPlan,
		PlanOverrides,
	} from '$lib/types';

	interface Props {
		open: boolean;
		model: InstalledModel | null;
	}

	let { open = $bindable(), model }: Props = $props();

	/** Tuning knobs only exist for llama-server loads. */
	const tunable = $derived(model?.caps.runtime === 'llama-server');

	let ctx = $state(4096);
	let ngl = $state('');
	let batchSize = $state('');
	let threads = $state('');
	let kvCacheType = $state('');
	let flashAttn = $state(false);
	let mmap = $state(true);
	let mlock = $state(false);
	let loading = $state(false);
	let backend = $state<'auto' | EngineKind>('auto');
	/** Memory limits as typed, in GB. Blank = whatever is free. */
	let ramLimitGb = $state('');
	let vramLimitGb = $state('');
	/** The limits the planner has been asked about (MiB), a beat behind the typing. */
	let ramLimitMib = $state<number | undefined>(undefined);
	let vramLimitMib = $state<number | undefined>(undefined);
	const LIMIT_DEBOUNCE_MS = 400;

	/** Context the auto plan picked; the slider starts here. `null` while the
	 *  first plan is in flight (or when the model can't be planned). */
	let autoCtx = $state<number | null>(null);
	/** Set once the dialog has its starting context (plan or fallback). */
	let ready = $state(false);
	/** Everything the planner's answer depends on. */
	const planKey = $derived(
		`${model?.id}|${ctx}|${kvCacheType}|${backend}|${ramLimitMib ?? ''}|${vramLimitMib ?? ''}`,
	);
	/** Latest answer from FitBadge, and the settings it was asked for. */
	type Receipt = { key: string } & (
		| { kind: 'plan'; plan: LoadPlan }
		| { kind: 'fit'; fit: FitEstimate }
		| { kind: 'failed'; message: string }
	);
	let receipt = $state<Receipt | null>(null);
	/** Admission for the settings on screen: an answer for other settings is no answer. */
	const current = $derived(receipt?.key === planKey ? receipt : null);
	const fit = $derived(current?.kind === 'fit' ? current.fit : null);
	/** The plan for these settings, including one that came with the fallback estimate. */
	const plan = $derived(current?.kind === 'plan' ? current.plan : (fit?.plan ?? null));
	const planFailed = $derived(current?.kind === 'failed' ? current.message : null);
	/** A request that never answers must not hold Load off for good. */
	const PLAN_TIMEOUT_MS = 30_000;

	function publish(next: Receipt, tag: string | undefined) {
		if (tag === planKey) receipt = next;
	}

	/** Bumped to run the check again with the same settings. */
	let retryCount = $state(0);
	function retryCheck() {
		receipt = null;
		retryCount++;
	}

	/** Blank GPU layers = the planner places the weights. A number input binds
	 *  a number (null when emptied), so the value is read through `String`. */
	const planned = $derived(String(ngl ?? '').trim() === '');
	const engines = $derived(engineStore.textEngines);
	/** A typed GPU-layer count is a llama.cpp launch, so it needs that engine and a llama plan. */
	const manualOk = $derived(
		engines.llama && backend !== 'ssd_stream' && plan?.engine_kind !== 'ssd_stream' && !planFailed,
	);
	const engineOverride = $derived<PlanOverrides>(backend === 'auto' ? {} : { engine: backend });

	/** GB as the plan rows show them (10^9 bytes) → the planner's MiB. Blank or junk = no limit. */
	function limitMib(v: string): number | undefined {
		const t = String(v ?? '').trim();
		if (t === '') return undefined;
		const n = Number(t);
		return Number.isFinite(n) && n >= 0 ? Math.round((n * 1e9) / 1024 ** 2) : undefined;
	}
	/** A typed limit the planner hasn't been asked about yet. */
	const limitsPending = $derived(
		limitMib(ramLimitGb) !== ramLimitMib || limitMib(vramLimitGb) !== vramLimitMib,
	);
	/** Engine choice and memory limits: what the planner is asked on top of context and KV type. */
	const planOverride = $derived<PlanOverrides>({
		...engineOverride,
		...(ramLimitMib !== undefined ? { ram_limit_mib: ramLimitMib } : {}),
		...(vramLimitMib !== undefined ? { vram_limit_mib: vramLimitMib } : {}),
	});
	const hasPlanOverride = $derived(Object.keys(planOverride).length > 0);

	const nativeLabel = $derived.by(() => {
		if (engines.nativeStreaming) return 'Native';
		if (!engines.nativeChecked) return 'Native (checking…)';
		return engineStore.native?.installed ? 'Native (no SSD streaming)' : 'Native (not installed)';
	});
	const BACKENDS = $derived([
		{ value: 'auto', label: 'Auto' },
		{
			value: 'llama_server',
			label: engines.llama ? 'llama.cpp' : 'llama.cpp (not installed)',
			disabled: !engines.llama,
		},
		{ value: 'ssd_stream', label: nativeLabel, disabled: !engines.nativeStreaming },
	]);

	/** Why Load is off with these choices, or null. */
	const loadBlock = $derived.by((): string | null => {
		if (!model || !tunable) return null;
		if (!planned) return engines.llama ? null : LLAMA_NEEDED;
		if (!ready || !current || limitsPending) return PLAN_PENDING;
		if (planFailed) return textEngineBlock(model, null, planFailed, engines);
		if (plan && !plan.feasible && !plan.evict_alternative) {
			return plan.infeasible_reason ?? "This model doesn't fit with these settings.";
		}
		return textEngineBlock(model, plan ? { plan } : fit, null, engines);
	});
	/** Why Load is off, for the note under the Engine select. A pending plan isn't a refusal. */
	const loadNote = $derived(loadBlock === PLAN_PENDING ? null : loadBlock);
	/** Only a context the user moved away from the auto pick is pinned. */
	const ctxOverride = $derived(autoCtx !== null && ctx === autoCtx ? undefined : ctx);

	function gbFromMb(mb: number): string {
		return `${(mb / 1024).toFixed(1)} GB`;
	}
	function gbFromBytes(b: number): string {
		return `${(b / 1e9).toFixed(1)} GB`;
	}
	const ctxLabel = $derived(`${(ctx / 1024).toFixed(ctx % 1024 === 0 ? 0 : 1)}k`);
	const kvHint = $derived.by(() => {
		if (plan) {
			const c = plan.context;
			const bytes = c.kv_bytes + c.swa_bytes + c.indexer_bytes + c.state_bytes;
			if (bytes > 0) return `Context memory ~${gbFromBytes(bytes)} at ${ctxLabel} context`;
		}
		if (!plan && fit && fit.gpu && fit.kv_at_ctx_mb > 0)
			return `+ KV cache ~${gbFromMb(fit.kv_at_ctx_mb)} at ${ctxLabel} context`;
		return null;
	});

	/** Tier split rows for the plan summary (zero tiers omitted). */
	const tierRows = $derived.by(() => {
		if (!plan) return [];
		const t = plan.tiers;
		const rows = [
			{ label: 'Graphics memory', bytes: t.vram_weights + t.vram_kv + t.vram_overhead },
			{ label: 'Expert cache (graphics memory)', bytes: t.vram_expert_cache },
			{ label: !plan.native ? 'System memory' : plan.gpu_layers !== 0 ? 'System memory (expert cache, engine)' : 'System memory (weights, context, expert cache)', bytes: t.ram },
			...(plan.native
				? [{ label: '· of which expert cache', bytes: plan.native.expert_cache_mib * 1024 ** 2 }]
				: []),
			{ label: 'SSD (streamed)', bytes: t.ssd },
		];
		return rows.filter((r) => r.bytes > 0);
	});

	const ctxTrain = $derived(model?.context_train);
	const ctxMax = $derived(
		ctxTrain && ctxTrain >= 1024 ? Math.floor(ctxTrain / 512) * 512 : 32768,
	);
	const ctxMaxLabel = $derived(
		ctxTrain
			? ctxTrain >= 1024 && ctxTrain % 1024 === 0
				? `${ctxTrain / 1024}k`
				: String(ctxTrain)
			: null,
	);

	// Reset the form whenever the dialog opens for a (possibly different) model,
	// then start the context slider at the auto plan's choice so a plain
	// "open → Load" is the one-click auto load.
	$effect(() => {
		if (!open) return;
		const m = model;
		untrack(() => {
			ctx = Math.min(8192, ctxMax);
			ngl = '';
			batchSize = '';
			threads = '';
			kvCacheType = '';
			flashAttn = false;
			mmap = true;
			mlock = false;
			backend = 'auto';
			ramLimitGb = '';
			vramLimitGb = '';
			ramLimitMib = undefined;
			vramLimitMib = undefined;
		});
	});

	// Ask the planner about a typed limit once the typing pauses.
	$effect(() => {
		const ram = limitMib(ramLimitGb);
		const vram = limitMib(vramLimitGb);
		if (ram === untrack(() => ramLimitMib) && vram === untrack(() => vramLimitMib)) return;
		const timer = setTimeout(() => {
			ramLimitMib = ram;
			vramLimitMib = vram;
		}, LIMIT_DEBOUNCE_MS);
		return () => clearTimeout(timer);
	});

	// The opening plan may set the context only until it gives up or the user
	// moves the slider; after that a late answer has no say.
	let openingPlan = 0;
	function retireOpeningPlan() {
		openingPlan++;
		ready = true;
	}
	function setCtxByUser(value: number) {
		ctx = value;
		retireOpeningPlan();
	}

	// Start the slider at the plan's own context, again whenever the engine or
	// a memory limit changes: each can change the context the planner picks.
	$effect(() => {
		if (!open) return;
		const m = model;
		const overrides = planOverride;
		untrack(() => {
			autoCtx = null;
			receipt = null;
			ready = false;
		});
		if (!m || m.caps.runtime !== 'llama-server') return;
		const run = ++openingPlan;
		const giveUp = setTimeout(retireOpeningPlan, PLAN_TIMEOUT_MS);
		planLoad(m.id, overrides)
			.then((p) => {
				if (run !== openingPlan) return;
				ctx = Math.min(p.ctx, ctxMax);
				autoCtx = ctx;
			})
			.catch(() => {
				// Not plannable (header unreadable): legacy defaults stand.
			})
			.finally(() => {
				clearTimeout(giveUp);
				if (run === openingPlan) ready = true;
			});
		return () => {
			openingPlan++;
			clearTimeout(giveUp);
		};
	});

	// No answer for these settings in time counts as a failed check; a late
	// answer still replaces it.
	$effect(() => {
		if (!open || !ready || !tunable || current) return;
		const key = planKey;
		const timer = setTimeout(() => {
			publish({ key, kind: 'failed', message: 'The memory check timed out.' }, key);
		}, PLAN_TIMEOUT_MS);
		return () => clearTimeout(timer);
	});

	// The GPU-layers box locks when the engine changes under a typed value.
	$effect(() => {
		if (!manualOk && untrack(() => ngl !== '')) ngl = '';
	});

	function intOrUndef(v: string): number | undefined {
		const t = String(v ?? '').trim();
		if (t === '') return undefined;
		const n = Number(t);
		return Number.isFinite(n) ? n : undefined;
	}

	const KV_CACHE_TYPES = [
		{ value: '', label: 'Auto' },
		{ value: 'f16', label: 'f16 (full size)' },
		{ value: 'q8_0', label: 'q8_0 (½ size)' },
		{ value: 'q4_0', label: 'q4_0 (¼ size)' },
	];

	async function handleLoad() {
		if (!model || loading || (tunable && loadBlock !== null)) return;
		loading = true;
		try {
			if (tunable) {
				// Blank GPU layers → ngl undefined → the planner places the
				// weights (flash attention / mmap are then the planner's call).
				await loadTextModel(model.id, {
					ctx: planned ? ctxOverride : ctx,
					ngl: planned ? undefined : intOrUndef(ngl),
					batchSize: intOrUndef(batchSize),
					threads: intOrUndef(threads),
					flashAttn,
					kvCacheType: kvCacheType || undefined,
					mmap,
					mlock,
					overrides: planned && hasPlanOverride ? planOverride : undefined,
				});
				addToast(`Loaded ${model.display_name}`, 'success');
			} else {
				// No load-time knobs — unified ledger flow (worker / qwen-tts).
				// Resolves false when the user cancels out of the eviction dialog.
				const loaded = await loadByLedger(model.id);
				if (loaded) addToast(`Loaded ${model.display_name}`, 'success');
			}
			open = false;
		} catch (e) {
			addToast(`Failed to load model: ${e}`, 'error');
		} finally {
			loading = false;
		}
	}

	/** Plain-language runtime description for the untunable summary card. */
	const runtimeLabel = $derived.by(() => {
		switch (model?.caps.runtime) {
			case 'qwen-tts':
				return 'the Qwen3-TTS engine (tts-server)';
			case 'worker':
				return 'the Python worker';
			default:
				return 'a local runtime';
		}
	});
</script>

<Modal bind:open title={model ? `Load ${model.display_name}` : 'Load model'} size="md">
	{#if model && !tunable}
		<!-- Worker pipeline / Qwen3-TTS talker — no load-time knobs, just confirm. -->
		<div class="space-y-3">
			<div class="rounded-[10px] border border-border bg-elevated/50 p-3">
				<p class="truncate text-xs text-muted-foreground" title={model.repo_id}>{model.repo_id}</p>
				<p class="text-2xs text-muted-foreground">
					{model.quantization ?? model.format}
					{#if model.params_b}
						· {model.params_b}B params
					{/if}
				</p>
			</div>
			<p class="text-xs text-muted-foreground">
				Loads into VRAM on {runtimeLabel} with its default settings — there are no
				load-time options for this model type.
			</p>
		</div>
	{:else if model}
		<div class="space-y-4">
			<div class="flex items-center justify-between rounded-[10px] border border-border bg-elevated/50 p-3">
				<div class="min-w-0">
					<p class="truncate text-xs text-muted-foreground" title={model.repo_id}>{model.repo_id}</p>
					<p class="text-2xs text-muted-foreground">
						{model.quantization ?? 'unknown quant'}
						{#if ctxMaxLabel}
							· trained context {ctxMaxLabel}
						{/if}
					</p>
				</div>
				{#if ready}
					{#key `${planKey}#${retryCount}`}
						<FitBadge
							modelId={model.id}
							ctx={ctxOverride}
							overrides={{
								...planOverride,
								...(kvCacheType ? { kv_type: kvCacheType as KvType } : {}),
							}}
							detailed
							receiptKey={planKey}
							onplan={(p, tag) => tag && publish({ key: tag, kind: 'plan', plan: p }, tag)}
							onestimate={(e, tag) => tag && publish({ key: tag, kind: 'fit', fit: e }, tag)}
							onfailed={(message, tag) => tag && publish({ key: tag, kind: 'failed', message }, tag)}
						/>
					{/key}
				{/if}
			</div>

			<div class="flex flex-col gap-1.5">
				<label for="mld-engine" class="text-xs font-medium text-muted-foreground">Engine</label>
				<Select id="mld-engine" bind:value={backend} options={BACKENDS} />
				<p class="text-2xs text-muted-foreground">
					Auto uses llama.cpp when the model fits in memory and the native engine when its experts
					have to stream from the SSD.
				</p>
				{#if loadBlock === PLAN_PENDING}
					<p class="text-2xs text-muted-foreground" data-testid="load-pending">{PLAN_PENDING}</p>
				{:else if loadNote && loadNote !== plan?.infeasible_reason}
					<p class="text-2xs text-warning" data-testid="load-block">
						{loadNote}
						{#if loadNote === LLAMA_NEEDED}<a class="link" href="/settings/engines">Open Engines</a>{/if}
						{#if planFailed}<button type="button" class="wb-btn sm ml-1" data-testid="load-retry" onclick={retryCheck}>Retry</button>{/if}
					</p>
				{/if}
			</div>

			{#if plan && planned}
				<div class="space-y-1.5 rounded-[10px] border border-border p-3" data-testid="load-plan">
					<div class="flex items-baseline justify-between text-xs">
						<span class="font-medium text-foreground">Where it runs</span>
						<span class="text-muted-foreground">
							{plan.speed.decode_tps < 10 ? `~${plan.speed.decode_tps.toFixed(1)}` : `~${Math.round(plan.speed.decode_tps)}`} tok/s generating ({plan.speed.label}) ·
							{plan.speed.prompt_tps > 0 ? `~${Math.round(plan.speed.prompt_tps)} tok/s reading the prompt` : 'prompt speed unknown'}
						</span>
					</div>
					<div class="flex justify-between text-2xs text-muted-foreground">
						<span>Engine</span>
						<span>{plan.engine_kind === 'ssd_stream' ? 'Native (SSD streaming)' : 'llama.cpp'}</span>
					</div>
					{#each tierRows as row (row.label)}
						<div class="flex justify-between text-2xs text-muted-foreground">
							<span>{row.label}</span>
							<span class="font-mono">{gbFromBytes(row.bytes)}</span>
						</div>
					{/each}
					{#each plan.warnings as w (w)}
						<p class="text-2xs text-warning">{w}</p>
					{/each}
					{#if plan.infeasible_reason && !plan.warnings.includes(plan.infeasible_reason)}
						<p class="text-2xs text-warning">{plan.infeasible_reason}</p>
					{/if}
				</div>
			{/if}

			<div class="flex flex-col gap-1">
				<Slider label="Context length" unit="tok" min={512} max={ctxMax} step={512} bind:value={() => ctx, setCtxByUser} />
				<p class="text-2xs text-muted-foreground">
					{#if ctxMaxLabel}
						Model max: {ctxMaxLabel} tokens (trained context)
					{:else}
						Model max unknown — capped at 32k
					{/if}
					{#if kvHint}
						<span class="text-muted-foreground/80"> · {kvHint}</span>
					{/if}
				</p>
			</div>

			<div class="grid grid-cols-2 gap-3">
				<div class="flex flex-col gap-1.5">
					<label for="mld-ram-limit" class="text-xs font-medium text-muted-foreground">System memory limit (GB)</label>
					<Input id="mld-ram-limit" type="number" bind:value={ramLimitGb} placeholder="auto" disabled={!planned} />
				</div>
				<div class="flex flex-col gap-1.5">
					<label for="mld-vram-limit" class="text-xs font-medium text-muted-foreground">Graphics memory limit (GB)</label>
					<Input id="mld-vram-limit" type="number" bind:value={vramLimitGb} placeholder="auto" disabled={!planned} />
				</div>
			</div>
			<p class="-mt-2 text-2xs text-muted-foreground">
				{#if planned}
					Blank uses what is free. Lower one to lean on the other; what fits in neither is read from
					the SSD. A limit never raises what the planner would use.
				{:else}
					Memory limits apply when the planner places the weights (GPU layers blank).
				{/if}
			</p>

			<div class="grid grid-cols-3 gap-3">
				<div class="flex flex-col gap-1.5">
					<label for="mld-ngl" class="text-xs font-medium text-muted-foreground">GPU layers (-ngl)</label>
					<Input id="mld-ngl" type="number" bind:value={ngl} placeholder="auto" disabled={!manualOk} />
				</div>
				<div class="flex flex-col gap-1.5">
					<label for="mld-batch" class="text-xs font-medium text-muted-foreground">Batch size</label>
					<Input id="mld-batch" type="number" bind:value={batchSize} placeholder="auto" />
				</div>
				<div class="flex flex-col gap-1.5">
					<label for="mld-threads" class="text-xs font-medium text-muted-foreground">Threads</label>
					<Input id="mld-threads" type="number" bind:value={threads} placeholder="auto" />
				</div>
			</div>
			<p class="-mt-2 text-2xs text-muted-foreground">
				{#if manualOk}
					Blank = the planner places the weights · -1 = all layers on GPU · 0 = CPU only
				{:else}
					The planner places the weights; GPU layers apply to llama.cpp loads only.
				{/if}
			</p>

			<div class="flex flex-col gap-1.5">
				<label for="mld-kv" class="text-xs font-medium text-muted-foreground">KV-cache type</label>
				<Select id="mld-kv" bind:value={kvCacheType} options={KV_CACHE_TYPES} />
			</div>

			<div class="space-y-2.5">
				{#if planned}
					<p class="text-2xs text-muted-foreground">
						The planner turns on flash attention and memory-mapping itself. Set GPU layers to choose
						these by hand.
					</p>
				{/if}
				<div class="flex items-center justify-between">
					<span class="text-xs text-foreground">Flash attention</span>
					<Toggle label="Flash attention" bind:checked={flashAttn} disabled={planned} />
				</div>
				<div class="flex items-center justify-between">
					<span class="text-xs text-foreground">Memory-map (mmap)</span>
					<Toggle label="Memory-map (mmap)" bind:checked={mmap} disabled={planned} />
				</div>
				<div class="flex items-center justify-between">
					<span class="text-xs text-foreground">Lock in RAM (mlock)</span>
					<Toggle label="Lock in RAM (mlock)" bind:checked={mlock} disabled={planned} />
				</div>
			</div>
		</div>
	{/if}

	{#snippet footer()}
		<div class="flex-1"></div>
		<Button variant="secondary" onclick={() => (open = false)}>Cancel</Button>
		<Button onclick={handleLoad} disabled={loading || !model || (tunable && loadBlock !== null)}>
			{loading ? 'Loading…' : 'Load model'}
		</Button>
	{/snippet}
</Modal>
