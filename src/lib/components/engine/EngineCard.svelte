<script lang="ts">
	/**
	 * EngineCard — one ruled engine row per runtime engine (M16 W6; restyled
	 * as a Workbench property-sheet section in the phase A redesign).
	 *
	 * Renders an EngineDescriptor from the engines.svelte.ts registry adapter:
	 * name/family, status badge (+ install progress), facts, primary lifecycle
	 * actions, resident models with unload, and an expandable detail with
	 * engine-specific extras (variant switch, pipeline capability table) and
	 * the logs viewer. Engine-agnostic by design — adding an engine means
	 * appending a descriptor, never touching this component.
	 */
	import Button from '$lib/components/ui/Button.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { unloadResident as storeUnloadResident } from '$lib/stores/engines.svelte';
	import { unloadLock } from '$lib/stores/chipLocks.svelte';
	import type { EngineAction, EngineDescriptor } from '$lib/stores/engines.svelte';
	import type { ResidentInfo } from '$lib/types';

	interface Props {
		engine: EngineDescriptor;
	}

	let { engine }: Props = $props();

	let busyAction = $state<string | null>(null);
	let expanded = $state(false);
	let logs = $state<string[]>([]);
	let logsFetched = $state(false);
	let unloading = $state<string | null>(null);

	async function runAction(a: EngineAction) {
		if (busyAction) return;
		busyAction = a.id;
		try {
			await a.run();
		} catch (e) {
			addToast(`${a.label.replace(/…$/, '')} failed: ${e}`, 'error');
		} finally {
			busyAction = null;
		}
	}

	async function refreshLogs() {
		try {
			logs = await engine.logs.fetch();
		} catch {
			logs = [];
		}
		logsFetched = true;
	}

	async function unloadResident(r: ResidentInfo) {
		unloading = r.id;
		try {
			await storeUnloadResident(r);
		} catch (e) {
			addToast(`Unload failed: ${e}`, 'error');
		} finally {
			unloading = null;
		}
	}

	function fmtGb(mb: number): string {
		return `${(mb / 1024).toFixed(1)} GB`;
	}

	function fmtBytes(bytes: number): string {
		return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
	}
</script>

<section class="engine" aria-labelledby="engine-{engine.id}">
	<!-- Header: status dot, name, family, status badge, primary actions -->
	<div class="eh">
		<div class="min-w-0">
			<h3 id="engine-{engine.id}" class="en">
				<span class="sdot {engine.status.tone}" aria-hidden="true"></span>
				{engine.name}
			</h3>
			<p class="ed">{engine.family}</p>
		</div>
		<div class="ea">
			<Badge variant={engine.status.tone}>{engine.status.label}</Badge>
			{#each engine.actions as a (a.id)}
				<span title={a.disabled ? a.reason : undefined}>
					<Button
						variant={a.tone}
						size="sm"
						onclick={() => runAction(a)}
						disabled={a.disabled || busyAction !== null}
					>
						{busyAction === a.id ? (a.busyLabel ?? a.label) : a.label}
					</Button>
				</span>
			{/each}
		</div>
	</div>

	{#if engine.status.progress}
		{@const p = engine.status.progress}
		<div class="ind mt-3">
			<div class="flex items-center justify-between text-xs text-muted-foreground">
				<span>{p.stage}</span>
				<span class="num">{fmtBytes(p.bytes)} / {fmtBytes(p.total)} · {Math.round(p.pct)}%</span>
			</div>
			<div class="tier-bar mt-1.5">
				<i style="width: {p.pct}%; background: var(--primary)"></i>
			</div>
		</div>
	{/if}

	{#if engine.status.detail}
		<p class="ind mt-2 flex items-center gap-2 text-xs text-muted-foreground">
			{#if engine.status.transitioning}<Spinner size="sm" />{/if}
			<span>{engine.status.detail}</span>
		</p>
	{/if}

	{#if !engine.installed && engine.status.phase !== 'installing'}
		<p class="ind mt-2 text-sm text-muted-foreground">{engine.description}</p>
	{/if}

	{#if engine.status.error}
		<p class="ind mt-2 rounded-md border border-error/30 bg-error/10 px-2.5 py-1.5 text-xs text-error" role="alert">
			{engine.status.error}
		</p>
	{/if}

	{#if engine.facts.length > 0}
		<dl class="ind facts">
			{#each engine.facts as fact (fact.label)}
				<dt>{fact.label}</dt>
				<dd class={fact.mono ? 'wb-code' : ''}>{fact.value}</dd>
			{/each}
		</dl>
	{/if}

	{#if engine.residents.length > 0}
		<ul class="ind residents" aria-label="Loaded on {engine.name}">
			{#each engine.residents as r (r.id)}
				{@const lock = unloadLock(r)}
				<li>
					<span class="sdot green" aria-hidden="true"></span>
					<span class="min-w-0 flex-1 truncate" title={r.label}>{r.label}</span>
					<span class="num text-xs text-muted-foreground">
						{#if r.actual_mb != null}
							{fmtGb(r.actual_mb)}
						{:else if r.est_mb > 0}
							~{fmtGb(r.est_mb)}
						{:else}
							—
						{/if}
					</span>
					{#if lock?.kind === 'in_use'}
						<span class="text-xs font-medium text-primary">In use</span>
					{/if}
					<span title={lock?.reason}>
						<Button
							variant="ghost"
							size="sm"
							onclick={() => unloadResident(r)}
							disabled={unloading === r.id || lock !== null}
						>
							{unloading === r.id ? 'Unloading…' : lock?.kind === 'replacing' ? 'Being replaced' : 'Unload'}
						</Button>
					</span>
				</li>
			{/each}
		</ul>
	{/if}

	<!-- Expandable detail: engine-specific extras + pipelines + logs -->
	<button type="button" class="ind disclose" onclick={() => (expanded = !expanded)} aria-expanded={expanded}>
		<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="h-3.5 w-3.5 transition-transform {expanded ? 'rotate-90' : ''}" aria-hidden="true">
			<path stroke-linecap="round" stroke-linejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
		</svg>
		Details and logs
	</button>

	{#if expanded}
		<div class="ind mt-3 space-y-5">
			{#if engine.detailActions.length > 0}
				<div class="flex flex-wrap gap-2">
					{#each engine.detailActions as a (a.id)}
						<Button variant={a.tone} size="sm" onclick={() => runAction(a)} disabled={a.disabled || busyAction !== null}>
							{busyAction === a.id ? (a.busyLabel ?? a.label) : a.label}
						</Button>
					{/each}
				</div>
			{/if}

			{#if engine.pipelines}
				<div>
					<h4 class="mb-1 text-xs font-semibold text-muted-foreground">Pipelines</h4>
					<ul class="divide-y divide-border">
						{#each engine.pipelines as p (p.task)}
							<li class="flex items-start justify-between gap-4 py-2.5">
								<div class="min-w-0">
									<div class="flex flex-wrap items-center gap-2">
										<span class="wb-code">{p.task}</span>
										{#each p.formats as f (f)}
											<Badge variant="slate">{f}</Badge>
										{/each}
									</div>
									<p class="mt-0.5 text-xs text-muted-foreground">{p.summary}</p>
									{#if !p.available && p.missing.length > 0}
										<p class="mt-0.5 text-xs text-error">
											Missing packages: {p.missing.join(', ')}. Reinstall the worker to repair.
										</p>
									{/if}
								</div>
								<Badge variant={p.available ? 'green' : 'red'}>{p.available ? 'Available' : 'Unavailable'}</Badge>
							</li>
						{/each}
					</ul>
				</div>
			{/if}

			<div>
				<div class="mb-2 flex items-center justify-between gap-3">
					<h4 class="text-xs font-semibold text-muted-foreground">Logs</h4>
					<Button variant="secondary" size="sm" onclick={refreshLogs} disabled={!engine.logs.available}>Refresh</Button>
				</div>
				{#if !engine.logs.available}
					<p class="text-xs text-muted-foreground">{engine.logs.hint}</p>
				{:else if logs.length === 0}
					<p class="text-xs text-muted-foreground">
						{logsFetched ? 'No log output yet.' : 'Select Refresh to fetch the latest output.'}
					</p>
				{:else}
					<pre class="max-h-64 overflow-auto rounded-lg border border-border bg-background p-3 font-mono text-xs leading-relaxed text-muted-foreground">{logs.join('\n')}</pre>
				{/if}
			</div>
		</div>
	{/if}
</section>

<style>
	.engine {
		padding: 14px 20px;
		border-bottom: 1px solid var(--border);
	}
	.eh {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 16px;
	}
	.en {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: var(--text-base);
		font-weight: 600;
		color: var(--foreground);
	}
	.ed {
		margin-top: 2px;
		padding-left: 15px;
		font-size: var(--text-xs);
		color: var(--muted-foreground);
	}
	/* Indent body content under the name, past the status dot. */
	.ind {
		margin-left: 15px;
	}
	.ea {
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		align-items: center;
		gap: 6px;
		flex: none;
	}
	.sdot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		flex: none;
		background: var(--muted-foreground);
		opacity: 0.45;
	}
	.sdot.green {
		background: var(--success);
		opacity: 1;
		box-shadow: 0 0 0 3px color-mix(in oklch, var(--success) 18%, transparent);
	}
	.sdot.yellow,
	.sdot.blue {
		background: var(--primary);
		opacity: 1;
	}
	.sdot.red {
		background: var(--error);
		opacity: 1;
	}
	.facts {
		display: grid;
		grid-template-columns: 160px minmax(0, 1fr);
		gap: 6px 16px;
		margin-top: 12px;
		font-size: var(--text-xs);
	}
	.facts dt {
		color: var(--muted-foreground);
	}
	.facts dd {
		margin: 0;
		min-width: 0;
		overflow-wrap: anywhere;
		color: var(--foreground);
	}
	.residents {
		margin-top: 12px;
		border-top: 1px solid color-mix(in oklch, var(--border) 60%, transparent);
	}
	.residents li {
		display: flex;
		align-items: center;
		gap: 10px;
		min-height: 36px;
		font-size: var(--text-sm);
		border-bottom: 1px solid color-mix(in oklch, var(--border) 60%, transparent);
	}
	.disclose {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		margin-top: 12px;
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--muted-foreground);
	}
	.disclose:hover {
		color: var(--foreground);
	}
	.disclose:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
		border-radius: 4px;
	}
</style>
