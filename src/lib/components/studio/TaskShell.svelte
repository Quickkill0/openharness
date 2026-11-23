<script lang="ts">
	/**
	 * TaskShell (Workbench) — shared scaffold for the Studio task pages. A
	 * 340px form pane on the left (scrolling fields, Generate pinned in its
	 * footer) and the result canvas on the right: a dotted stage that shows the
	 * idle hint, the running job card or the finished result, with the
	 * recent-generations strip underneath.
	 */
	import type { Snippet } from 'svelte';
	import TaskError from './TaskError.svelte';
	import RecentStrip from './RecentStrip.svelte';
	import { studio, cancelJob, takeRemix, type TaskResult } from '$lib/stores/studio.svelte';
	import { ledgerStore } from '$lib/stores/ledger.svelte';
	import { gbFromMb } from '$lib/format';
	import { stageLabel } from './studioMeta';
	import type { Asset, GenerationTask } from '$lib/types';

	interface Props {
		task: GenerationTask;
		/** Library source-task ids for the recent strip (defaults to [task]). */
		assetTasks?: string[];
		/** Restore a recent asset's prompt/params into the form ("remix"). */
		onremix?: (asset: Asset) => void;
		/** Form fields (model picker, inputs, Advanced disclosure). */
		controls: Snippet;
		/** The primary action, pinned to the form pane's footer. */
		action: Snippet;
		/** Renders the finished result inside the stage. */
		resultView: Snippet<[TaskResult]>;
		/** Idle-stage hint. */
		empty: Snippet;
	}

	let { task, assetTasks, onremix, controls, action, resultView, empty }: Props = $props();

	const active = $derived(studio.active[task]);
	const result = $derived(studio.results[task]);
	const error = $derived(studio.errors[task]);

	/** The ledger resident doing this job, for the memory line on the job card. */
	const resident = $derived.by(() => {
		const rs = ledgerStore.residents;
		return (
			rs.find((r) => r.id === `pipeline:${task}`) ??
			(task === 'text_to_speech' ? rs.find((r) => r.id === 'tts') : undefined) ??
			null
		);
	});
	const residentMb = $derived(resident ? (resident.actual_mb ?? resident.est_mb) : 0);
	const pct = $derived(active ? Math.max(0, Math.min(100, active.pct)) : 0);

	// "Remix" from the Library tab: claim the asset meant for this task page
	// and restore it into the form, the same as clicking it in the recent strip.
	$effect(() => {
		if (studio.remix?.task !== task) return;
		const asset = takeRemix(task);
		if (asset) onremix?.(asset);
	});
</script>

<div class="shell">
	<aside class="form" aria-label="Settings">
		<div class="form-body">
			{@render controls()}
		</div>
		<div class="wb-ins-foot">
			{@render action()}
		</div>
	</aside>

	<div class="canvas">
		{#if error}
			<TaskError {task} message={error} />
		{/if}
		<div class="stage" class:busy={!!active}>
			{#if active}
				<div class="jobcard" role="status" aria-live="polite">
					<div class="jc-top">
						<span class="spin" aria-hidden="true"></span>
						<b>{stageLabel(active.stage)}</b>
						<span class="flex-1"></span>
						<button type="button" class="wb-btn sm" onclick={() => cancelJob(task)}>Cancel</button>
					</div>
					<div
						class="wb-pbar {active.pct <= 0 ? 'indeterminate' : ''}"
						role="progressbar"
						aria-valuemin={0}
						aria-valuemax={100}
						aria-valuenow={Math.round(pct)}
						aria-label="Progress"
					>
						<i style="width: {Math.max(2, pct)}%"></i>
					</div>
					<div class="jc-meta">
						<span>
							{#if active.step != null && active.totalSteps != null}
								Step <span class="num">{active.step}</span> of <span class="num">{active.totalSteps}</span>
							{:else}
								Working
							{/if}
						</span>
						<span class="num">{Math.round(pct)}%</span>
					</div>
					{#if resident}
						<div class="jc-mem">
							<span class="swatch" aria-hidden="true"></span>
							<span class="truncate">{resident.label}</span>
							<span class="flex-none">using <span class="num">{gbFromMb(residentMb)} GB</span> of graphics memory</span>
						</div>
					{/if}
				</div>
			{:else if result}
				<div class="result">
					{@render resultView(result)}
				</div>
			{:else}
				<div class="idle">
					{@render empty()}
				</div>
			{/if}
		</div>

		<RecentStrip tasks={assetTasks ?? [task]} refreshKey={result?.jobId ?? null} {onremix} />
	</div>
</div>

<!-- Shared Studio form styles: every task page renders its fields inside
     `.form-body`, so field layout lives here once. -->
<style>
	.shell {
		display: grid;
		grid-template-columns: 340px minmax(0, 1fr);
		height: 100%;
		min-height: 0;
		overflow: hidden;
	}
	@media (max-width: 1180px) {
		.shell {
			grid-template-columns: 300px minmax(0, 1fr);
		}
	}
	.form {
		display: flex;
		flex-direction: column;
		min-height: 0;
		border-right: 1px solid var(--border);
		background: var(--surface);
	}
	.form-body {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: 16px;
		padding: 16px;
	}
	.form :global(.wb-ins-foot .wb-btn) {
		flex: 1;
		height: 38px;
		font-size: var(--text-sm);
	}

	.canvas {
		display: flex;
		flex-direction: column;
		gap: 16px;
		min-width: 0;
		min-height: 0;
		padding: 20px;
	}
	.stage {
		position: relative;
		flex: 1;
		min-height: 280px;
		display: grid;
		place-items: center;
		overflow: auto;
		border: 1px solid var(--border);
		border-radius: 14px;
		background:
			radial-gradient(color-mix(in oklch, var(--muted-foreground) 22%, transparent) 1px, transparent 1px) 0 0 / 16px
				16px,
			var(--surface);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
	}
	.idle,
	.result {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		width: 100%;
		height: 100%;
		padding: 16px;
	}
	.idle {
		gap: 10px;
		text-align: center;
	}
	.idle :global(.st-idle-ic) {
		display: grid;
		place-items: center;
		width: 52px;
		height: 52px;
		border-radius: 14px;
		border: 1px solid var(--border);
		background: var(--elevated);
		color: var(--muted-foreground);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
	}
	.idle :global(.st-idle-t) {
		margin-top: 4px;
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--foreground);
	}
	.idle :global(.st-idle-p) {
		max-width: 320px;
		font-size: var(--text-xs);
		line-height: 1.5;
		color: var(--muted-foreground);
	}
	.result :global(.st-out) {
		border-radius: 10px;
		border: 1px solid var(--border);
		background: var(--elevated);
		object-fit: contain;
		box-shadow: 0 18px 40px -22px rgb(0 0 0 / 0.55);
	}
	.result :global(.st-cap) {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 12px;
		max-width: 100%;
		min-width: 0;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.result :global(.st-cap > * + *) {
		padding-left: 12px;
		border-left: 1px solid var(--border);
	}

	.jobcard {
		width: min(420px, 90%);
		padding: 16px;
		border: 1px solid var(--border);
		border-radius: 14px;
		background: var(--elevated);
		box-shadow:
			0 18px 40px -18px rgb(0 0 0 / 0.5),
			inset 0 1px 0 var(--inset-highlight);
	}
	.jc-top {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 12px;
	}
	.jc-top b {
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--foreground);
	}
	.jobcard .wb-pbar {
		height: 6px;
	}
	.jc-meta {
		display: flex;
		justify-content: space-between;
		margin-top: 8px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.jc-mem {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 10px;
		padding-top: 10px;
		border-top: 1px solid var(--border);
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
		min-width: 0;
	}
	.swatch {
		flex: none;
		width: 8px;
		height: 8px;
		border-radius: 2px;
		background: var(--tier-vram);
	}
	.spin {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		border: 2px solid color-mix(in oklch, var(--primary) 30%, transparent);
		border-top-color: var(--primary);
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	/* ---- Shared field styles for the task forms ---- */
	.form-body :global(.st-field) {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.form-body :global(.st-label) {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--muted-foreground);
	}
	.form-body :global(.st-label .num) {
		color: var(--foreground);
		font-size: var(--text-2xs);
	}
	.form-body :global(.st-row) {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		font-size: var(--text-sm);
		color: var(--foreground);
	}
	.form-body :global(.st-grid2) {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
	}
	.form-body :global(textarea.wb-input),
	.form-body :global(input.wb-input),
	.form-body :global(select.wb-input) {
		width: 100%;
	}
	.form-body :global(.st-drop) {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		padding: 12px;
		border: 1px dashed var(--border);
		border-radius: 8px;
		background: var(--background);
		font-size: var(--text-xs);
		color: var(--muted-foreground);
		text-align: left;
		cursor: pointer;
		transition:
			border-color 0.15s ease,
			color 0.15s ease;
	}
	.form-body :global(.st-drop:hover:not(:disabled)) {
		border-color: var(--ring);
		color: var(--foreground);
	}
	.form-body :global(.st-drop.has) {
		border-style: solid;
		color: var(--foreground);
	}
	.form-body :global(.st-drop:focus-visible) {
		outline: 2px solid var(--ring);
		outline-offset: 1px;
	}

	@media (prefers-reduced-motion: reduce) {
		.spin {
			animation: none;
		}
	}
</style>
