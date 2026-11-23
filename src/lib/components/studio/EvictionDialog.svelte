<script lang="ts">
	/**
	 * Eviction dialog (Workbench) — shown when a Studio job or model load
	 * doesn't fit in graphics memory. Lists what is loaded as a ruled list the
	 * user can unload from, with a meter showing free memory against what the
	 * job needs. Retry re-submits; Force skips the fit check and lets the load
	 * spill into system memory.
	 */
	import Modal from '$lib/components/ui/Modal.svelte';
	import { studio, dismissEviction, forceSubmit, retrySubmit } from '$lib/stores/studio.svelte';
	import { orchestratorUnload } from '$lib/ipc/commands';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { gbFromMb } from '$lib/format';

	const prompt = $derived(studio.eviction);
	/** Residents unloaded from this dialog (struck out, can't unload twice). */
	let unloaded = $state<Set<string>>(new Set());
	let busy = $state<string | null>(null);

	/** Memory freed by unloads in this dialog, added to the free figure. */
	const freedMb = $derived(
		prompt
			? prompt.candidates
					.filter((c) => unloaded.has(c.id))
					.reduce((sum, c) => sum + (c.actual_mb ?? c.est_mb), 0)
			: 0,
	);
	const freeNow = $derived(prompt ? prompt.freeMb + freedMb : 0);
	const fits = $derived(prompt ? freeNow >= prompt.requiredMb : false);
	/** Bar scale: whichever is larger, so both marks stay inside the track. */
	const scale = $derived(prompt ? Math.max(prompt.requiredMb, freeNow, 1) : 1);

	function kindLabel(kind: string): string {
		if (kind === 'text_engine') return 'Text model';
		if (kind === 'tts_engine') return 'Speech engine';
		return 'Studio model';
	}

	async function unload(id: string) {
		busy = id;
		try {
			await orchestratorUnload(id);
			unloaded = new Set([...unloaded, id]);
		} catch (e) {
			addToast(`Couldn't unload: ${e}`, 'error');
		} finally {
			busy = null;
		}
	}

	async function retry() {
		unloaded = new Set();
		await retrySubmit();
	}

	async function force() {
		unloaded = new Set();
		await forceSubmit();
	}

	function close() {
		unloaded = new Set();
		dismissEviction();
	}
</script>

{#if prompt}
	<Modal open title="Not enough graphics memory" size="md" onclose={close}>
		<p class="ev-lead">
			{prompt.task === 'load_model' ? 'This model' : 'This job'} needs about
			<span class="num">{gbFromMb(prompt.requiredMb)} GB</span> of graphics memory. Unload something below and
			retry, or force it and let the overflow spill into system memory, which is much slower.
		</p>

		<div class="ev-meter">
			<div class="ev-meter-top">
				<span><span class="sw vram" aria-hidden="true"></span>Free graphics memory</span>
				<span>
					<span class="num" class:ok={fits}>{gbFromMb(freeNow)} GB</span>
					of <span class="num need">{gbFromMb(prompt.requiredMb)} GB</span> needed
				</span>
			</div>
			<div
				class="ev-track"
				role="meter"
				aria-label="Free graphics memory against what is needed"
				aria-valuemin={0}
				aria-valuemax={prompt.requiredMb}
				aria-valuenow={Math.min(freeNow, prompt.requiredMb)}
			>
				<i class="ev-free" style="width: {(prompt.freeMb / scale) * 100}%"></i>
				<i class="ev-freed" style="left: {(prompt.freeMb / scale) * 100}%; width: {(freedMb / scale) * 100}%"></i>
				<span class="ev-need" style="left: {(prompt.requiredMb / scale) * 100}%" aria-hidden="true"></span>
			</div>
		</div>

		<div class="ev-list">
			<div class="ev-list-h">Loaded now</div>
			{#each prompt.candidates as c (c.id)}
				{@const gone = unloaded.has(c.id)}
				<div class="ev-row" class:gone>
					<span class="sw vram" aria-hidden="true"></span>
					<div class="ev-name">
						<b class="truncate">{c.label}</b>
						<span>{kindLabel(c.kind)}</span>
					</div>
					<span class="num ev-mb">{gbFromMb(c.actual_mb ?? c.est_mb)} GB</span>
					{#if gone}
						<span class="wb-badge ok">Unloaded</span>
					{:else}
						<button type="button" class="wb-btn sm" disabled={busy !== null} onclick={() => unload(c.id)}>
							{busy === c.id ? 'Unloading…' : 'Unload'}
						</button>
					{/if}
				</div>
			{:else}
				<p class="ev-none">Nothing else is loaded. Force it, or pick a smaller model.</p>
			{/each}
		</div>

		{#snippet footer()}
			<div class="ev-foot">
				<button type="button" class="wb-btn ghost" onclick={close}>Cancel</button>
				<span class="flex-1"></span>
				<button type="button" class="wb-btn danger" onclick={force}>Force anyway</button>
				<button type="button" class="wb-btn primary" disabled={unloaded.size === 0} onclick={retry}>Retry</button>
			</div>
		{/snippet}
	</Modal>
{/if}

<style>
	.ev-lead {
		font-size: var(--text-sm);
		line-height: 1.55;
		color: var(--muted-foreground);
	}
	.ev-lead .num {
		color: var(--foreground);
	}
	.ev-meter {
		margin-top: 16px;
		padding: 12px;
		border: 1px solid var(--border);
		border-radius: 10px;
		background: var(--surface);
	}
	.ev-meter-top {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 8px;
		font-size: var(--text-xs);
		color: var(--muted-foreground);
	}
	.ev-meter-top > span:first-child {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.ev-meter-top .num {
		color: var(--warning);
	}
	.ev-meter-top .num.ok {
		color: var(--success);
	}
	.ev-meter-top .num.need {
		color: var(--foreground);
	}
	.ev-track {
		position: relative;
		height: 8px;
		border-radius: 4px;
		background: color-mix(in oklch, var(--tier-vram) 14%, var(--elevated));
		overflow: visible;
	}
	.ev-free,
	.ev-freed {
		position: absolute;
		top: 0;
		bottom: 0;
		left: 0;
		border-radius: 4px;
		background: var(--tier-vram);
		transition: width 0.3s ease;
	}
	.ev-freed {
		background: repeating-linear-gradient(
			135deg,
			var(--tier-vram) 0 4px,
			color-mix(in oklch, var(--tier-vram) 55%, transparent) 4px 8px
		);
	}
	.ev-need {
		position: absolute;
		top: -4px;
		bottom: -4px;
		width: 2px;
		margin-left: -1px;
		border-radius: 1px;
		background: var(--foreground);
	}
	.ev-list {
		margin-top: 16px;
	}
	.ev-list-h {
		padding-bottom: 6px;
		font-size: var(--text-2xs);
		font-weight: 600;
		color: var(--muted-foreground);
	}
	.ev-row {
		display: flex;
		align-items: center;
		gap: 10px;
		min-height: 48px;
		padding: 6px 0;
		border-top: 1px solid var(--border);
	}
	.ev-row:last-child {
		border-bottom: 1px solid var(--border);
	}
	.ev-name {
		display: flex;
		flex-direction: column;
		flex: 1;
		min-width: 0;
	}
	.ev-name b {
		font-size: var(--text-sm);
		font-weight: 500;
		color: var(--foreground);
	}
	.ev-name span {
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.ev-mb {
		font-size: var(--text-xs);
		color: var(--foreground);
	}
	.ev-row.gone .ev-name b,
	.ev-row.gone .ev-mb {
		text-decoration: line-through;
		opacity: 0.5;
	}
	.ev-none {
		padding: 12px 0;
		border-top: 1px solid var(--border);
		font-size: var(--text-xs);
		color: var(--muted-foreground);
	}
	.sw {
		flex: none;
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 2px;
	}
	.sw.vram {
		background: var(--tier-vram);
	}
	.ev-foot {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
	}
	@media (prefers-reduced-motion: reduce) {
		.ev-free,
		.ev-freed {
			transition: none;
		}
	}
</style>
