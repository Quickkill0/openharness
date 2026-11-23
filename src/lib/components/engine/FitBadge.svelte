<script lang="ts">
	/**
	 * FitBadge — small pill summarizing where a model would run and how fast.
	 *
	 * Asks the load planner (`plan_load`) first: the pill shows the memory tier
	 * the plan reaches (all in graphics memory / spills to system memory /
	 * streams from the SSD / won't fit) and the estimated decode speed. Models
	 * the planner can't read (non-llama-server GGUFs such as TTS talkers) fall
	 * back to the older `estimate_model_fit` VRAM verdict. That estimate knows
	 * only the context, so it never stands in for a request with other overrides.
	 *
	 * Fetches once on mount. When free VRAM moves, re-fetches at most every
	 * 500 ms and only after >100 MB of drift.
	 *
	 * Expects an INSTALLED model id (matches Rust `models::registry::get_by_id`).
	 * Do not feed a raw HuggingFace repo id.
	 */
	import { onMount } from 'svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import { estimateModelFit, planLoad } from '$lib/ipc/commands';
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import type { FitEstimate, FitVerdict, LoadPlan, PlanOverrides } from '$lib/types';

	interface Props {
		modelId: string;
		ctx?: number;
		/** Extra planner overrides (KV type, expert placement…); `ctx` wins. */
		overrides?: PlanOverrides;
		/** If true, also show the graphics-memory footprint. */
		detailed?: boolean;
		/** Handed back with every callback so the host can match an answer to its request. */
		receiptKey?: string;
		/** Fired with each fresh plan (the host dialog shows the tier split). */
		onplan?: (p: LoadPlan, receiptKey?: string) => void;
		/** Fired with each fresh fallback estimate (models without a plan). */
		onestimate?: (e: FitEstimate, receiptKey?: string) => void;
		/** Fired when neither the plan nor the estimate could be computed. */
		onfailed?: (message: string, receiptKey?: string) => void;
	}

	let {
		modelId,
		ctx,
		overrides,
		detailed = false,
		receiptKey,
		onplan,
		onestimate,
		onfailed,
	}: Props = $props();

	let plan = $state<LoadPlan | null>(null);
	let estimate = $state<FitEstimate | null>(null);
	let loading = $state(true);
	let errorMsg = $state<string | null>(null);
	let lastFetchAt = $state(0);
	let lastFreeMb = $state(0);
	/** Only the newest request of a live badge may report; bumped on destroy too. */
	let fetchSeq = 0;

	async function fetchFit() {
		const seq = ++fetchSeq;
		const key = receiptKey;
		loading = true;
		errorMsg = null;
		try {
			try {
				const ov: PlanOverrides | undefined =
					ctx || overrides ? { ...overrides, ...(ctx ? { ctx } : {}) } : undefined;
				const p = await planLoad(modelId, ov);
				if (seq !== fetchSeq) return;
				plan = p;
				estimate = null;
				onplan?.(p, key);
			} catch (planError) {
				if (seq !== fetchSeq) return;
				if (Object.values(overrides ?? {}).some((v) => v !== undefined)) throw planError;
				const e = await estimateModelFit(modelId, ctx);
				if (seq !== fetchSeq) return;
				estimate = e;
				plan = null;
				onestimate?.(e, key);
			}
			lastFetchAt = Date.now();
			lastFreeMb = gpuStore.vramFree;
		} catch (e: unknown) {
			if (seq !== fetchSeq) return;
			errorMsg = e instanceof Error ? e.message : String(e);
			onfailed?.(errorMsg, key);
		} finally {
			if (seq === fetchSeq) loading = false;
		}
	}

	onMount(() => {
		fetchFit();
		return () => {
			fetchSeq++;
		};
	});

	// Re-fetch when VRAM moves: only if >500ms AND >100MB drift.
	$effect(() => {
		const freeNow = gpuStore.vramFree;
		if ((!plan && !estimate) || loading) return;
		const now = Date.now();
		const drift = Math.abs(freeNow - lastFreeMb);
		if (now - lastFetchAt > 500 && drift > 100) {
			fetchFit();
		}
	});

	function humanGB(bytesOrMb: number, unit: 'b' | 'mb' = 'mb'): string {
		const gb = unit === 'b' ? bytesOrMb / 1e9 : bytesOrMb / 1024;
		if (!gb || gb <= 0) return '0 GB';
		return `${gb.toFixed(1)} GB`;
	}

	// ── planner view ────────────────────────────────────────────────────
	const vramBytes = $derived(
		plan
			? plan.tiers.vram_weights + plan.tiers.vram_expert_cache + plan.tiers.vram_kv + plan.tiers.vram_overhead
			: 0,
	);
	const tps = $derived(plan ? `~${Math.round(plan.speed.decode_tps)} tok/s` : '');
	const planTitle = $derived.by(() => {
		if (!plan) return '';
		const t = plan.tiers;
		const parts = [`Graphics memory ${humanGB(vramBytes, 'b')}`];
		if (t.ram > 0) parts.push(`system memory ${humanGB(t.ram, 'b')}`);
		if (t.ssd > 0) parts.push(`SSD ${humanGB(t.ssd, 'b')}`);
		const speed = `${Math.round(plan.speed.decode_tps)} tok/s generating (${plan.speed.label})`;
		return [`${parts.join(', ')}. ${speed}.`, ...plan.warnings].join('\n');
	});
	const planTone = $derived.by<'success' | 'warning' | 'error' | 'muted'>(() => {
		if (!plan) return 'muted';
		if (!plan.feasible) return 'error';
		if (vramBytes === 0 && plan.engine_kind !== 'ssd_stream') return 'muted';
		return plan.tier_mix === 'vram' ? 'success' : 'warning';
	});
	const planText = $derived.by(() => {
		if (!plan) return '';
		if (!plan.feasible) return "Won't fit";
		if (vramBytes === 0 && plan.engine_kind !== 'ssd_stream') return 'CPU only';
		return plan.tier_mix === 'vram' ? 'Fits' : plan.tier_mix === 'ram' ? 'Uses RAM' : 'Uses SSD';
	});

	const TONE_CLASS = {
		success: 'bg-success/15 text-success ring-success/30',
		warning: 'bg-warning/15 text-warning ring-warning/30',
		error: 'bg-error/15 text-error ring-error/30',
		muted: 'bg-elevated text-muted-foreground ring-border/40',
	} as const;

	// ── fallback estimate view ──────────────────────────────────────────
	/** Live free VRAM plus what a same-slot swap would release. */
	const availableMb = $derived(gpuStore.vramFree + (estimate?.reclaimable_mb ?? 0));

	function localVerdict(estMb: number, freeMb: number): FitVerdict {
		if (estMb <= freeMb - 1024) return 'fits';
		if (estMb <= freeMb) return 'tight';
		return 'wont_fit';
	}

	const verdict = $derived.by<FitVerdict | null>(() => {
		if (!estimate) return null;
		if (!estimate.gpu) return 'fits';
		return localVerdict(estimate.estimated_mb, availableMb);
	});
	const estimateTone = $derived(
		!estimate || !estimate.gpu
			? 'muted'
			: verdict === 'fits'
				? 'success'
				: verdict === 'tight'
					? 'warning'
					: 'error',
	);
	const estimateText = $derived(
		!estimate ? '' : !estimate.gpu ? 'CPU only' : verdict === 'fits' ? 'Fits' : verdict === 'tight' ? 'Tight' : "Won't fit",
	);
	const estimateTitle = $derived(
		estimate
			? `Estimated VRAM: ${humanGB(estimate.estimated_mb)} / ${humanGB(availableMb)} available`
			: '',
	);
</script>

{#if loading && !plan && !estimate}
	<span
		class="inline-flex items-center gap-1.5 rounded-md bg-elevated px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-border/40"
		aria-label="Estimating fit"
	>
		<Spinner size="sm" class="h-3 w-3 border" />
		Estimating…
	</span>
{:else if errorMsg}
	<span
		class="inline-flex items-center rounded-md bg-elevated px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-border/40"
		title={errorMsg}
	>
		— estimate unavailable
	</span>
{:else if plan}
	<span
		class="inline-flex items-center gap-1 whitespace-nowrap rounded-md px-2 py-0.5 text-xs font-medium ring-1 {TONE_CLASS[planTone]}"
		title={planTitle}
		data-testid="fit-badge"
	>
		{planText}
		{#if plan.feasible}
			<span class="opacity-80">• {tps}</span>
		{/if}
		{#if detailed && vramBytes > 0}
			<span class="opacity-80">• {humanGB(vramBytes, 'b')} VRAM</span>
		{/if}
	</span>
{:else if estimate}
	<span
		class="inline-flex items-center gap-1 whitespace-nowrap rounded-md px-2 py-0.5 text-xs font-medium ring-1 {TONE_CLASS[estimateTone]}"
		title={estimateTitle}
		data-testid="fit-badge"
	>
		{estimateText}
		{#if detailed}
			<span class="opacity-80">• {humanGB(estimate.estimated_mb)}</span>
		{/if}
	</span>
{/if}
