<script lang="ts">
	/**
	 * Settings, Hardware: what OpenHarness measures about this PC (NVML for the
	 * graphics card, the system_memory command for RAM) plus everything loaded
	 * right now, with Unload and Reload with new settings. Replaces the old
	 * /settings/vram "GPU & Models" page, which now redirects here.
	 *
	 * Rows the backend doesn't measure yet (SSD read speed, memory bandwidth,
	 * PCIe link) are left out until the load planner adds them.
	 */
	import { onMount } from 'svelte';
	import Button from '$lib/components/ui/Button.svelte';
	import ModelLoadDialog from '$lib/components/models/ModelLoadDialog.svelte';
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import { systemStore } from '$lib/stores/system.svelte';
	import { ledgerStore, initLedgerStore } from '$lib/stores/ledger.svelte';
	import { unloadLock } from '$lib/stores/chipLocks.svelte';
	import { initEngineStore } from '$lib/stores/engine.svelte';
	import { listLocalModels, orchestratorUnloadIf } from '$lib/ipc/commands';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { gbFromMb, pct } from '$lib/format';
	import type { InstalledModel, ResidentInfo } from '$lib/types';

	const gpu = $derived(gpuStore.status);
	const ram = $derived(systemStore.memory);
	const ledger = $derived(ledgerStore.snapshot);

	let catalog = $state<Map<string, InstalledModel>>(new Map());
	let unloading = $state<string | null>(null);
	let reloadOpen = $state(false);
	let reloadTarget = $state<InstalledModel | null>(null);

	onMount(() => {
		void initLedgerStore();
		void initEngineStore();
		void listLocalModels()
			.then((ms) => (catalog = new Map(ms.map((m) => [m.id, m]))))
			.catch(() => {});
	});

	const KIND_LABEL: Record<ResidentInfo['kind'], string> = {
		text_engine: 'Text (llama.cpp)',
		tts_engine: 'Speech (Qwen3-TTS)',
		pipeline: 'Studio (Python worker)',
	};

	async function unload(r: ResidentInfo) {
		const lock = unloadLock(r);
		if (lock) {
			addToast(lock.reason, 'info');
			return;
		}
		unloading = r.id;
		try {
			// Guarded: a no-op or a clear error if another model took the slot.
			await orchestratorUnloadIf(r.id, r.model_id ?? null);
		} catch (e) {
			addToast(`Couldn't unload ${r.label}: ${e}`, 'error');
		} finally {
			unloading = null;
		}
	}

	function reload(m: InstalledModel) {
		reloadTarget = m;
		reloadOpen = true;
	}
</script>

<div class="wb-set-h">
	<h2>Hardware</h2>
	<p>What OpenHarness measures on this PC</p>
</div>

{#if gpu.kind === 'available'}
	<div class="wb-prop">
		<div class="pl">Graphics card</div>
		<div class="pv">{gpu.name}, <span class="num">{gbFromMb(gpu.vram_total_mb)} GB</span></div>
	</div>
	<div class="wb-prop">
		<div class="pl">Graphics memory</div>
		<div class="pv">
			<div class="flex items-center gap-3">
				<div class="tier-bar flex-1" style="max-width: 280px" role="img" aria-label="{gbFromMb(gpu.vram_used_mb)} of {gbFromMb(gpu.vram_total_mb)} GB used">
					<i style="width: {pct(gpu.vram_used_mb, gpu.vram_total_mb)}; background: var(--tier-vram)"></i>
				</div>
				<span><span class="num">{gbFromMb(gpu.vram_used_mb)}</span> <span class="text-muted-foreground">of</span> <span class="num">{gbFromMb(gpu.vram_total_mb)} GB</span> <span class="text-muted-foreground">in use</span></span>
			</div>
		</div>
	</div>
	<div class="wb-prop">
		<div class="pl">Driver</div>
		<div class="pv"><span class="num">{gpu.driver_version}</span>, CUDA <span class="num">{gpu.cuda_major}.{gpu.cuda_minor}</span></div>
	</div>
{:else}
	<div class="wb-prop">
		<div class="pl">Graphics card</div>
		<div class="pv">
			No NVIDIA GPU detected.
			<span class="text-muted-foreground">Text models run on the CPU, which works but is slower.</span>
		</div>
	</div>
{/if}

{#if ram}
	<div class="wb-prop">
		<div class="pl">System memory</div>
		<div class="pv">
			<div class="flex items-center gap-3">
				<div class="tier-bar flex-1" style="max-width: 280px" role="img" aria-label="{gbFromMb(ram.used_mb)} of {gbFromMb(ram.total_mb)} GB used">
					<i style="width: {pct(ram.used_mb, ram.total_mb)}; background: var(--tier-ram)"></i>
				</div>
				<span><span class="num">{gbFromMb(ram.used_mb)}</span> <span class="text-muted-foreground">of</span> <span class="num">{gbFromMb(ram.total_mb)} GB</span> <span class="text-muted-foreground">in use</span></span>
			</div>
		</div>
	</div>
{/if}

<div class="wb-set-h sub">
	<h2>Loaded now</h2>
	{#if ledger}
		<p><span class="num">{gbFromMb(ledger.free_mb)} GB</span> graphics memory free</p>
	{/if}
</div>

{#if !ledger}
	<p class="px-5 py-4 text-sm text-muted-foreground">Reading what's loaded…</p>
{:else if ledger.residents.length === 0}
	<p class="px-5 py-4 text-sm text-muted-foreground">
		Nothing is loaded. Load a model from <a href="/models" class="text-primary underline underline-offset-2">Models</a>.
	</p>
{:else}
	{#each ledger.residents as r (r.id)}
		{@const model = r.model_id ? (catalog.get(r.model_id) ?? null) : null}
		{@const lock = unloadLock(r)}
		<div class="wb-prop">
			<div class="pl">
				<span class="truncate text-foreground" title={r.label}>{r.label}</span>
				<small>{KIND_LABEL[r.kind]}</small>
			</div>
			<div class="pv flex flex-wrap items-center justify-between gap-2">
				<span>
					{#if r.actual_mb != null}
						<span class="num">{gbFromMb(r.actual_mb)} GB</span> <span class="text-muted-foreground">measured</span>
					{:else if r.est_mb > 0}
						<span class="num">{gbFromMb(r.est_mb)} GB</span> <span class="text-muted-foreground">estimated</span>
					{:else}
						<span class="text-muted-foreground">Size unknown</span>
					{/if}
				</span>
				<span class="flex gap-2">
					{#if model}
						<Button variant="secondary" size="sm" onclick={() => reload(model)} disabled={unloading === r.id || ledgerStore.isLoading(model.id)}>
							{ledgerStore.isLoading(model.id) ? 'Loading…' : 'Reload with options'}
						</Button>
					{/if}
					<span title={lock?.reason}>
						<Button variant="secondary" size="sm" onclick={() => unload(r)} disabled={unloading === r.id || lock !== null}>
							{unloading === r.id ? 'Unloading…' : lock?.kind === 'replacing' ? 'Being replaced' : 'Unload'}
						</Button>
					</span>
				</span>
				{#if lock}
					<p class="w-full text-xs text-muted-foreground">{lock.reason}</p>
				{/if}
			</div>
		</div>
	{/each}
{/if}

<ModelLoadDialog bind:open={reloadOpen} model={reloadTarget} />
