<script lang="ts">
	/**
	 * CuratedRail — horizontal strip of curated picks for the active store tab
	 * (M9 §9.5). Picks come from the static manifest baked into the backend,
	 * with `fits` resolved against the detected GPU's total VRAM.
	 *
	 * Clicking a pick hands its repo id to the parent, which searches for it so
	 * the normal row → detail → download flow takes over (one code path).
	 */
	import { storeCuratedPicks } from '$lib/ipc/commands';
	import type { CuratedPick } from '$lib/types';
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import { gbFromMb, pct } from '$lib/format';

	interface Props {
		/** Active task tab value; 'any' shows every pick. */
		task?: string;
		onpick?: (pick: CuratedPick) => void;
	}

	let { task = 'any', onpick }: Props = $props();

	let picks = $state<CuratedPick[]>([]);
	/** One row of four by default; the rest sit behind "Show all". */
	const ROW = 4;
	let showAll = $state(false);
	const visible = $derived(showAll ? picks : picks.slice(0, ROW));
	/** Blank cells that square off a partial last row. */
	const fillers = $derived((ROW - (visible.length % ROW)) % ROW);

	$effect(() => {
		const t = task === 'any' ? undefined : task;
		storeCuratedPicks(t)
			.then((p) => (picks = p))
			.catch(() => (picks = []));
	});

	/** Graphics memory the bar is measured against: the GPU's total, or the
	 * largest pick when there is no GPU so bars still compare. */
	const scaleMb = $derived(
		gpuStore.vramTotal > 0 ? gpuStore.vramTotal : Math.max(1, ...picks.map((p) => p.min_vram_mb)),
	);

	function formatLabel(format: string): string {
		switch (format) {
			case 'gguf': return 'GGUF';
			case 'diffusers-dir': return 'Diffusers';
			case 'ctranslate2-dir': return 'CTranslate2';
			case 'kokoro-repo': return 'Kokoro';
			default: return format;
		}
	}
</script>

{#if picks.length > 0}
	<div class="picks-label">
		<span>Picks for your PC</span>
		<span class="wb-help flex items-center gap-3">
			<span>
				{#if gpuStore.hasGpu}
					{gpuStore.name}, <span class="num">{gbFromMb(gpuStore.vramTotal)}</span> GB graphics memory
				{:else}
					No graphics card found; sizes are graphics memory needed
				{/if}
			</span>
			{#if picks.length > ROW}
				<button type="button" class="more" aria-expanded={showAll} onclick={() => (showAll = !showAll)}>
					{showAll ? 'Show fewer' : `Show all ${picks.length}`}
				</button>
			{/if}
		</span>
	</div>
	<div class="picks-wrap">
		<div class="picks">
			{#each visible as pick (pick.repo_id)}
				<button type="button" class="pick" onclick={() => onpick?.(pick)} title="{pick.repo_id}: {pick.blurb}">
					<span class="pn">{pick.display_name}</span>
					<span
						class="tier-bar mini"
						role="img"
						aria-label="Needs about {gbFromMb(pick.min_vram_mb)} GB of graphics memory"
					>
						<i
							style="width: {pct(pick.min_vram_mb, scaleMb)}; background: {pick.fits
								? 'var(--tier-vram)'
								: 'var(--error)'}"
						></i>
					</span>
					<span class="pm">
						<span>{formatLabel(pick.format)}</span>
						<span class:bad={!pick.fits}>
							{#if !pick.fits}Too big,&nbsp;{/if}<span class="num">{gbFromMb(pick.min_vram_mb)}</span> GB
						</span>
					</span>
				</button>
			{/each}
			{#each { length: fillers } as _, i (i)}
				<span class="pick filler" aria-hidden="true"></span>
			{/each}
		</div>
	</div>
{/if}

<style>
	.picks-label {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 12px;
		padding: 12px 16px 0;
		font-size: var(--text-xs);
		font-weight: 600;
		color: var(--foreground);
	}
	.picks-wrap {
		padding: 8px 16px 14px;
	}
	.picks {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 1px;
		overflow: hidden;
		border: 1px solid var(--border);
		border-radius: 10px;
		background: var(--border);
	}
	.more {
		font-weight: 500;
		color: var(--primary);
	}
	.more:hover {
		text-decoration: underline;
	}
	.more:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
		border-radius: 4px;
	}
	.pick.filler {
		pointer-events: none;
	}
	.pick {
		display: flex;
		min-width: 0;
		flex-direction: column;
		gap: 6px;
		padding: 12px 14px;
		background: var(--background);
		text-align: left;
		transition: background 0.15s ease;
	}
	.pick:hover {
		background: var(--surface);
	}
	.pick:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: -2px;
	}
	.pn {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--text-sm);
		font-weight: 500;
		color: var(--foreground);
	}
	.tier-bar.mini {
		height: 5px;
		width: 100%;
	}
	.pm {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 6px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.pm .bad {
		color: color-mix(in oklch, var(--error) 70%, var(--foreground));
	}
	@media (prefers-reduced-motion: reduce) {
		.pick {
			transition: none;
		}
	}
</style>
