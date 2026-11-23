<script lang="ts">
	/**
	 * Workbench bottom dock — the machine's live state on every page.
	 *
	 * Left: graphics-memory (NVML via gpuStore, residents via the ledger) and
	 * system-memory (system_memory poll) meters. The SSD meter from the mockup
	 * is omitted until a real read-rate source exists.
	 * Middle: one chip per ledger resident with an Unload ×, plus a dashed
	 * "will load" ghost when the Models page publishes a preview.
	 * Right: Server pill (API server state from the server store; links to
	 * /server) and Downloads (opens the drawer).
	 */
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import { ledgerStore } from '$lib/stores/ledger.svelte';
	import { unloadLock } from '$lib/stores/chipLocks.svelte';
	import { orchestratorUnloadIf } from '$lib/ipc/commands';
	import type { ChipLock } from './chipLock';
	import { systemStore } from '$lib/stores/system.svelte';
	import { serverState, refreshServerStatus } from '$lib/stores/server.svelte';
	import { downloadStore } from '$lib/stores/downloads.svelte';
	import { dock, openDownloads } from '$lib/stores/ui.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { gbFromMb, pct } from '$lib/format';
	import Icon from './Icon.svelte';
	import type { ResidentInfo } from '$lib/types';

	const residentMb = (r: ResidentInfo) => r.actual_mb ?? r.est_mb;

	// --- Graphics memory -------------------------------------------------
	const vTotal = $derived(gpuStore.vramTotal || ledgerStore.snapshot?.total_mb || 0);
	const residents = $derived(ledgerStore.residents);
	const residentSum = $derived(residents.reduce((s, r) => s + residentMb(r), 0));
	/** NVML-measured use; fall back to the ledger's own sum without NVML. */
	const vUsed = $derived(gpuStore.hasGpu ? gpuStore.vramUsed : residentSum);
	/** What else holds VRAM: desktop, games, other apps. */
	const vOther = $derived(Math.max(0, vUsed - residentSum));
	const preview = $derived(
		dock.preview && !ledgerStore.isLoaded(dock.preview.modelId) ? dock.preview : null,
	);
	/** Used after the load: a chat load first frees the model it replaces. */
	const vAfter = $derived(preview ? vUsed - preview.freesMb + preview.vramMb : vUsed);
	const replaces = $derived(preview?.replacesId ?? null);
	const replacedLabel = $derived(
		replaces ? (residents.find((r) => r.id === replaces)?.label ?? null) : null,
	);
	const overflows = $derived(preview != null && vTotal > 0 && vAfter > vTotal);

	// --- System memory ---------------------------------------------------
	const ram = $derived(systemStore.memory);

	// --- API server ------------------------------------------------------
	// The Server page pushes every status it sees; this slow poll catches
	// changes made elsewhere (auto-start on launch, a failed bind).
	$effect(() => {
		void refreshServerStatus();
		const t = setInterval(() => void refreshServerStatus(), 15_000);
		return () => clearInterval(t);
	});
	const server = $derived(serverState.status);
	const serverTitle = $derived(
		!server
			? 'API server'
			: server.error
				? `API server error: ${server.error}`
				: server.running
					? `API server running on port ${server.settings.port}, ${server.settings.bind === 'home_network' ? 'home network' : 'this PC only'}`
					: 'API server is off',
	);

	// --- Downloads -------------------------------------------------------
	const activeDownloads = $derived(downloadStore.activeCount);

	// --- Chip locks -------------------------------------------------------
	const locks = $derived(
		Object.fromEntries(residents.map((r) => [r.id, unloadLock(r)])) as Record<string, ChipLock>,
	);

	let unloading = $state<string | null>(null);
	async function unload(r: ResidentInfo) {
		if (locks[r.id]) return;
		unloading = r.id;
		try {
			// Only unloads if the slot still holds the model this chip shows;
			// a stale × can't take out the model that replaced it.
			await orchestratorUnloadIf(r.id, r.model_id ?? null);
		} catch (e) {
			addToast(`Couldn't unload ${r.label}: ${e}`, 'error');
		} finally {
			unloading = null;
		}
	}
</script>

<footer class="dock" aria-label="System status">
	<!-- Graphics memory -->
	<div
		class="meter {preview ? 'previewing' : ''}"
		title={gpuStore.hasGpu ? `Graphics memory (VRAM) on ${gpuStore.name}` : 'No NVIDIA GPU detected'}
	>
		<div class="meter-top">
			<span class="meter-name">
				<span class="swatch" style="background: var(--tier-vram)"></span>
				<span class="truncate">Graphics<span class="long">{' '}memory</span></span>
			</span>
			<span class="meter-val">
				{#if vTotal > 0}
					<span class="num">{gbFromMb(vUsed)}</span>
					{#if preview}
						<span class="num {overflows ? 'text-error' : 'text-primary'}">→ {gbFromMb(vAfter)} GB</span>
					{:else}
						<span class="num text-muted-foreground">/ {gbFromMb(vTotal)} GB</span>
					{/if}
				{:else}
					<span class="text-muted-foreground">No GPU</span>
				{/if}
			</span>
		</div>
		<div class="tier-bar" role="img" aria-label="Graphics memory: {gbFromMb(vUsed)} of {gbFromMb(vTotal)} GB used">
			{#if vTotal > 0}
				<i style="width: {pct(vOther, vTotal)}; background: var(--tier-base)" title="Other apps"></i>
				{#each residents as r (r.id)}
					{#if r.id !== replaces}
						<i
							style="width: {pct(residentMb(r), vTotal)}; background: var(--tier-vram)"
							title="{r.label} · {gbFromMb(residentMb(r))} GB"
						></i>
					{/if}
				{/each}
				{#if preview}
					<i
						class="ghost"
						style="width: {pct(Math.min(preview.vramMb, Math.max(0, vTotal - (vUsed - preview.freesMb))), vTotal)}; --c: {overflows
							? 'var(--error)'
							: 'var(--tier-vram)'}"
					></i>
				{/if}
			{/if}
		</div>
	</div>

	<!-- System memory -->
	{#if ram}
		<div class="meter" title="System memory (RAM)">
			<div class="meter-top">
				<span class="meter-name">
					<span class="swatch" style="background: var(--tier-ram)"></span>
					<span class="truncate">System<span class="long">{' '}memory</span></span>
				</span>
				<span class="meter-val">
					<span class="num">{gbFromMb(ram.used_mb)}</span>
					<span class="num text-muted-foreground">/ {gbFromMb(ram.total_mb)} GB</span>
				</span>
			</div>
			<div class="tier-bar" role="img" aria-label="System memory: {gbFromMb(ram.used_mb)} of {gbFromMb(ram.total_mb)} GB used">
				<i style="width: {pct(ram.used_mb, ram.total_mb)}; background: var(--tier-ram)"></i>
			</div>
		</div>
	{/if}

	<!-- Loaded models -->
	<div class="chips" aria-label="Loaded models">
		{#each residents as r (r.id)}
			{@const lock = locks[r.id]}
			<span
				class="rchip {r.id === replaces || lock?.kind === 'replacing' ? 'leaving' : ''} {lock
					? lock.kind
					: ''}"
				title={lock
					? `${r.label} · ${lock.reason}`
					: r.id === replaces
						? `${r.label} · would be replaced by ${preview?.label}`
						: `${r.label} · ${gbFromMb(residentMb(r))} GB graphics memory`}
			>
				<span class="dot {lock ? 'busy' : 'on'}" aria-hidden="true"></span>
				<span class="nm">{r.label}</span>
				{#if lock?.kind === 'replacing'}
					<span class="state">Being replaced</span>
				{:else if lock?.kind === 'in_use'}
					<span class="state">In use</span>
				{:else}
					<span class="num k">{gbFromMb(residentMb(r))} GB</span>
				{/if}
				{#if lock?.kind !== 'replacing'}
					<button
						type="button"
						class="x"
						aria-label={lock ? `Can't unload ${r.label}: ${lock.reason}` : `Unload ${r.label}`}
						title={lock ? lock.reason : 'Unload'}
						disabled={unloading === r.id || lock != null}
						onclick={() => unload(r)}
					>
						<Icon name="x" size={14} />
					</button>
				{/if}
				{#if lock}
					<span class="sr-only">{lock.reason}</span>
				{/if}
			</span>
		{/each}
		{#each [...ledgerStore.loadingIds] as id (id)}
			<span class="rchip loading" aria-live="polite">
				<span class="dot busy" aria-hidden="true"></span>
				<span class="nm">Loading…</span>
			</span>
		{/each}
		{#if preview}
			<span
				class="rchip ghost"
				title="Selected on Models: loading it needs about {gbFromMb(preview.vramMb)} GB{replacedLabel
					? ` and replaces ${replacedLabel}`
					: ''}"
			>
				<span class="nm">{preview.label}</span>
				<span class="num k">+{gbFromMb(preview.vramMb)} GB</span>
			</span>
		{/if}
		{#if residents.length === 0 && ledgerStore.loadingIds.size === 0 && !preview}
			<span class="text-xs text-muted-foreground">Nothing loaded</span>
		{/if}
	</div>

	<!-- Server + downloads -->
	<div class="right">
		<a class="pill" href="/server" title={serverTitle}>
			{#if server}
				<span
					class="sdot {server.error ? 'err' : server.running ? 'on' : ''}"
					aria-hidden="true"
				></span>
			{:else}
				<Icon name="server" size={14} class="text-muted-foreground" />
			{/if}
			<span>Server</span>
			{#if server?.running && !server.error}
				<span class="num text-muted-foreground">:{server.settings.port}</span>
			{:else if server && !server.error}
				<span class="text-muted-foreground">off</span>
			{/if}
			<span class="sr-only">{serverTitle}</span>
		</a>
		<button type="button" class="pill" title="Downloads" onclick={openDownloads}>
			<Icon name="download" size={14} class="text-muted-foreground" />
			<span class="lbl">Downloads</span>
			{#if activeDownloads > 0}
				<span class="count num">{activeDownloads}</span>
			{/if}
		</button>
	</div>
</footer>

<style>
	.dock {
		display: flex;
		align-items: stretch;
		height: 60px;
		border-top: 1px solid var(--border);
		background: var(--surface);
		position: relative;
		z-index: 5;
	}
	.meter {
		flex: 0 1 232px;
		min-width: 150px;
		display: flex;
		flex-direction: column;
		justify-content: center;
		gap: 6px;
		height: 100%;
		padding: 0 14px;
		border-right: 1px solid var(--border);
	}
	.meter.previewing {
		background: linear-gradient(to bottom, color-mix(in oklch, var(--tier-vram) 7%, transparent), transparent);
	}
	.meter.previewing .long {
		display: none;
	}
	.meter-top {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 8px;
		font-size: var(--text-2xs);
	}
	.meter-name {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
		color: var(--muted-foreground);
		font-weight: 500;
		white-space: nowrap;
	}
	.meter-val {
		white-space: nowrap;
	}
	.swatch {
		width: 10px;
		height: 10px;
		border-radius: 3px;
		flex: none;
	}
	.chips {
		flex: 1 1 0;
		min-width: 0;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 0 12px;
		overflow-x: auto;
		scrollbar-width: none;
		/* Fade the trailing edge so a clipped chip reads as "more", not broken. */
		mask-image: linear-gradient(to right, #000 calc(100% - 16px), transparent);
	}
	.rchip {
		display: inline-flex;
		align-items: center;
		gap: 7px;
		/* Names shrink (ellipsis) before the row starts scrolling. */
		flex: 0 1 auto;
		min-width: 88px;
		height: 32px;
		padding: 0 4px 0 10px;
		border-radius: 8px;
		border: 1px solid var(--border);
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
		font-size: var(--text-xs);
		white-space: nowrap;
	}
	.rchip:hover {
		border-color: color-mix(in oklch, var(--primary) 50%, transparent);
	}
	.rchip .nm {
		min-width: 0;
		max-width: 160px;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.rchip .k {
		flex: none;
		color: var(--muted-foreground);
		font-size: var(--text-2xs);
	}
	.rchip .x {
		flex: none;
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
		border-radius: 5px;
		color: var(--muted-foreground);
	}
	.rchip .x:hover {
		background: var(--muted);
		color: var(--foreground);
	}
	.rchip .x:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 1px;
	}
	.rchip .x:disabled {
		opacity: 0.4;
		pointer-events: none;
	}
	.rchip.loading {
		border-color: var(--primary);
		padding-right: 10px;
	}
	.rchip.ghost {
		border-style: dashed;
		border-color: var(--primary);
		background: transparent;
		box-shadow: none;
		color: var(--primary);
		padding-right: 10px;
	}
	.rchip.ghost .k {
		color: var(--primary);
	}
	/* Would be swapped out by the previewed load. */
	.rchip.leaving {
		/* Gives up its room first when the row is tight. */
		flex-shrink: 4;
		opacity: 0.5;
	}
	.rchip.leaving .nm {
		text-decoration: line-through;
	}
	/* Status word in place of the size while the chip is locked. */
	.rchip .state {
		flex: none;
		font-size: var(--text-2xs);
		font-weight: 500;
		color: var(--muted-foreground);
	}
	.rchip.replacing {
		padding-right: 10px;
		cursor: default;
	}
	.rchip.in_use .state {
		color: var(--primary);
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
	.dot.busy {
		background: var(--primary);
		animation: dock-pulse 1s ease-in-out infinite;
	}
	@keyframes dock-pulse {
		50% {
			opacity: 0.35;
		}
	}
	.right {
		flex: none;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 0 12px;
		border-left: 1px solid var(--border);
	}
	.pill {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		height: 32px;
		padding: 0 12px;
		border-radius: 8px;
		border: 1px solid var(--border);
		color: var(--foreground);
		font-size: var(--text-xs);
		white-space: nowrap;
		text-decoration: none;
	}
	.pill:hover {
		background: var(--elevated);
	}
	.pill:active {
		transform: translateY(1px);
	}
	.pill:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
	}
	.sdot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--muted-foreground);
		flex: none;
	}
	.sdot.on {
		background: var(--success);
		box-shadow: 0 0 0 3px color-mix(in oklch, var(--success) 22%, transparent);
	}
	.sdot.err {
		background: var(--error);
	}
	.count {
		display: grid;
		place-items: center;
		min-width: 18px;
		height: 18px;
		padding: 0 5px;
		border-radius: 9px;
		background: var(--primary);
		color: var(--primary-foreground);
		font-size: var(--text-2xs);
	}
	@media (max-width: 1180px) {
		.long,
		.lbl {
			display: none;
		}
		.meter {
			flex-basis: 188px;
		}
		/* Sizes stay in each chip's tooltip; names get the room. */
		.rchip:not(.ghost) .k {
			display: none;
		}
		.rchip .nm {
			max-width: 120px;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.dot.busy {
			animation: none;
		}
	}
</style>
