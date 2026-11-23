<script lang="ts">
	/**
	 * Store inspector file area (Workbench).
	 *
	 *  - "Best for my PC" box: the recommended quant from store_quant_fits with
	 *    a one-click download.
	 *  - "All quants" table: every GGUF with its fit verdict, size, and a
	 *    graphics-memory bar.
	 *  - "All files & formats": the raw file list, collapsed when there is an
	 *    obvious action (best quant or a whole-repo download).
	 *
	 * The bar is single-tier on purpose: store_quant_fits gives a graphics
	 * memory estimate only. The three-tier placement (graphics memory, system
	 * memory, SSD) and speed estimates come from the loader planner in a later
	 * phase and slot into the `.where` cell.
	 */
	import { onMount } from 'svelte';
	import { startDownload, startRepoDownload, listDownloads } from '$lib/ipc/commands';
	import { splitGgufPart, splitGgufFiles } from '$lib/splitGguf';
	import { addJob, seedFromList } from '$lib/stores/downloads.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import { formatBytes, gbFromMb, pct } from '$lib/format';
	import Icon from '$lib/components/shell/Icon.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import { isFileInstalled } from './installed';
	import type { HfFile, InstalledModel, QuantFit } from '$lib/types';

	interface Props {
		files: HfFile[];
		repoId: string;
		revision?: string;
		/** Optional VRAM estimates; split sets remain downloadable without them. */
		quantFits?: QuantFit[];
		task?: string | null;
		/** Installed catalog rows for this repo (marks quants as installed). */
		installed?: InstalledModel[];
		/** Parent offers a whole-repo download (dir layouts) — counts as the obvious action. */
		hasRepoAction?: boolean;
		/** Called when a download is queued so the parent can open the drawer. */
		ondownloadqueued?: () => void;
	}

	let {
		files,
		repoId,
		revision = 'main',
		quantFits,
		task = null,
		installed = [],
		hasRepoAction = false,
		ondownloadqueued,
	}: Props = $props();

	const fits = $derived(quantFits ?? []);
	const bestFit = $derived(fits.find((f) => f.best) ?? null);
	const bestFile = $derived(bestFit ? (files.find((f) => f.path === bestFit.path) ?? null) : null);
	const fileByPath = $derived(new Map(files.map((f) => [f.path, f])));

	/** File membership, not a GPU estimate, makes a split set downloadable. */
	const quantRows = $derived.by(() => {
		const rows: { path: string; size: number; label: string | null; split: boolean;
			part_count: number; best: boolean; estimate?: QuantFit }[] = [];
		for (const fit of fits) {
			if (fileByPath.has(fit.path) && !splitGgufPart(fit.path))
				rows.push({ ...fit, part_count: 1, estimate: fit });
		}
		for (const file of files) {
			const part = splitGgufPart(file.path);
			if (part?.number !== 1) continue;
			try {
				const members = splitGgufFiles(file, files);
				const estimate = fits.find((f) => f.path === file.path);
				rows.push({ path: file.path, size: members.reduce((s, f) => s + f.size, 0),
					label: estimate?.label ?? basename(part.key), split: true,
					part_count: part.count, best: false, estimate });
			} catch { /* Incomplete sets must not offer a broken download. */ }
		}
		return rows.sort((a, b) => a.size - b.size);
	});

	// Raw-file disclosure. The action signals resolve asynchronously, so the
	// default is latched once rather than derived live — otherwise the list
	// renders open and collapses out from under the user when fits land.
	let filesToggled = $state<boolean | null>(null);
	let autoDefault = $state<boolean | null>(null);
	$effect(() => {
		if (autoDefault === null && (bestFit !== null || hasRepoAction || quantRows.length > 0))
			autoDefault = false;
	});
	onMount(() => {
		const t = setTimeout(() => {
			autoDefault ??= !(bestFit !== null || hasRepoAction || quantRows.length > 0);
		}, 800);
		return () => clearTimeout(t);
	});
	const filesOpen = $derived(filesToggled ?? autoDefault ?? false);

	/** Plain-language guidance per quant family. */
	function quantGuidance(label: string | null): string | null {
		if (!label) return null;
		if (label.startsWith('Q8') || label === 'F16' || label === 'BF16' || label === 'F32')
			return 'Near-lossless. Biggest file, little gain over Q6.';
		if (label.startsWith('Q6')) return 'Very high quality, close to the original weights.';
		if (label.startsWith('Q5')) return 'High quality. A great choice when it fits.';
		if (label.startsWith('Q4')) return 'The usual balance of quality and size.';
		if (label.startsWith('Q3') || label.startsWith('IQ3'))
			return 'Noticeable quality loss. Use when memory is tight.';
		if (label.startsWith('Q2') || label.startsWith('IQ2') || label.startsWith('IQ1'))
			return 'Heavy quality loss. Last resort for small cards.';
		return null;
	}

	function verdictText(fit: QuantFit): { text: string; tone: string } {
		switch (fit.verdict) {
			case 'fits':
				return { text: 'Fits in VRAM', tone: 'var(--success)' };
			case 'tight':
				return { text: 'Tight in VRAM', tone: 'var(--warning)' };
			default:
				return {
					text: 'Exceeds VRAM',
					tone: 'color-mix(in oklch, var(--error) 70%, var(--foreground))',
				};
		}
	}

	/** Graphics memory the bars measure against (GPU total, else largest estimate). */
	const scaleMb = $derived(
		gpuStore.vramTotal > 0 ? gpuStore.vramTotal : Math.max(1, ...fits.map((f) => f.estimated_mb)),
	);

	function basename(path: string): string {
		return path.split('/').pop() ?? path;
	}

	/** Installed when a catalog row for this repo is this exact file (hash or file name). */
	function isInstalled(file: HfFile): boolean {
		const part = splitGgufPart(file.path);
		if (part) return installed.some((m) => m.revision === revision
			&& m.primary_rel_path === file.path && m.file_count === part.count
			&& isFileInstalled(file, [m]));
		return isFileInstalled(file, installed);
	}

	/** Track which files are being queued (brief loading state). */
	let queueing = $state<Set<string>>(new Set());

	/** Sort: .gguf files first, then alphabetical. */
	const sortedFiles = $derived(
		[...files].sort((a, b) => {
			const aGguf = a.path.endsWith('.gguf') ? 0 : 1;
			const bGguf = b.path.endsWith('.gguf') ? 0 : 1;
			if (aGguf !== bGguf) return aGguf - bGguf;
			return a.path.localeCompare(b.path);
		}),
	);

	const totalBytes = $derived(files.reduce((s, f) => s + (f.size || 0), 0));

	function queueKey(path: string): string { return splitGgufPart(path)?.key ?? path; }

	async function handleDownload(file: HfFile) {
		const key = queueKey(file.path);
		if (queueing.has(key)) return;
		queueing = new Set(queueing).add(key);
		try {
			if (splitGgufPart(file.path)) {
				const members = splitGgufFiles(file, files);
				await startRepoDownload({ repoId, revision, files: members, format: 'gguf', task });
				seedFromList(await listDownloads());
				addToast(`Downloading ${basename(key)} (${members.length} parts)`, 'success');
				ondownloadqueued?.();
				return;
			}
			const jobId = await startDownload(repoId, revision, file);
			// Seed a minimal job so the drawer shows it immediately.
			addJob({
				id: jobId,
				repo_id: repoId,
				revision,
				file: file.path,
				dest_path: '',
				total_bytes: file.size,
				downloaded: 0,
				chunks: [],
				status: 'queued',
				expected_sha256: file.sha256,
				created_at: Math.floor(Date.now() / 1000),
				updated_at: Math.floor(Date.now() / 1000),
			});
			addToast(`Downloading ${basename(file.path)}`, 'success');
			ondownloadqueued?.();
		} catch (e: unknown) {
			addToast(`Download failed: ${String(e)}`, 'error');
		} finally {
			const next = new Set(queueing);
			next.delete(key);
			queueing = next;
		}
	}
</script>

{#if bestFit && bestFile}
	{@const busy = queueing.has(bestFile.path)}
	{@const have = isInstalled(bestFile)}
	{@const v = verdictText(bestFit)}
	<div class="wb-sec">
		<div class="best">
			<div class="wb-sec-h mb-2">
				<span class="wb-sec-label" title="Full-GPU estimate before download; native streaming feasibility is checked after import">Best for graphics memory</span>
				<span class="wb-dot" style="--tone: {v.tone}">{v.text}</span>
			</div>
			<div class="flex items-baseline gap-2.5">
				<span class="num text-base font-medium text-foreground">{bestFit.label ?? basename(bestFile.path)}</span>
				<span class="text-xs text-muted-foreground">
					<span class="num">{formatBytes(bestFit.size)}</span> download
				</span>
			</div>
			<div class="mt-2.5">
				<div class="tier-bar placement" role="img" aria-label="Uses about {gbFromMb(bestFit.estimated_mb)} GB of graphics memory">
					<i style="width: {pct(bestFit.estimated_mb, scaleMb)}; background: var(--tier-vram)"></i>
				</div>
				<div class="legend">
					<span class="lg"><i style="background: var(--tier-vram)"></i>Graphics memory</span>
					<span class="num">
						{gbFromMb(bestFit.estimated_mb)}{#if gpuStore.vramTotal > 0}&nbsp;of {gbFromMb(gpuStore.vramTotal)}{/if} GB
					</span>
				</div>
			</div>
			{#if quantGuidance(bestFit.label)}
				<p class="mt-2 text-xs text-muted-foreground">{quantGuidance(bestFit.label)}</p>
			{/if}
			<button
				type="button"
				class="wb-btn primary block mt-3"
				disabled={busy || have}
				onclick={() => handleDownload(bestFile)}
			>
				{#if busy}<Spinner size="sm" />Starting download{:else if have}Installed{:else}<Icon
						name="download"
						size={14}
					/>Download {bestFit.label ?? 'this file'}{/if}
			</button>
		</div>
	</div>
{/if}

{#if quantRows.length > 0}
	<div class="wb-sec quants">
		<div class="wb-sec-h px-4">
			<span class="wb-sec-label">All quants</span>
			<span class="wb-help">Estimate assumes nothing else is loaded</span>
		</div>
		<table class="wb-table qt">
			<thead>
				<tr>
					<th>Quant</th>
					<th class="r">Size</th>
					<th>Graphics memory</th>
					<th><span class="sr-only">Download</span></th>
				</tr>
			</thead>
			<tbody>
				{#each quantRows as fit (fit.path)}
					{@const file = fileByPath.get(fit.path)!}
					{@const v = fit.estimate ? verdictText(fit.estimate) : null}
					{@const busy = queueing.has(queueKey(fit.path))}
					<tr class:best-row={fit.best} title={quantGuidance(fit.label) ?? undefined}>
						<td>
							<div class="num text-xs text-foreground">
								{fit.label ?? basename(fit.path)}{#if fit.split}<span class="split"> split · {fit.part_count} parts</span>{/if}
							</div>
							<div class="vt" style:color={v?.tone}>{v?.text ?? 'No GPU estimate'}</div>
						</td>
						<td class="r num text-xs">{formatBytes(fit.size)}</td>
						<td class="where">
							{#if fit.estimate}
							<span
								class="tier-bar mini"
								class:over={fit.estimate.verdict === 'wont_fit'}
								role="img"
								aria-label="About {gbFromMb(fit.estimate.estimated_mb)} GB of graphics memory"
							>
								<i style="width: {pct(fit.estimate.estimated_mb, scaleMb)}; background: var(--tier-vram)"></i>
							</span>
							<span class="num est">{gbFromMb(fit.estimate.estimated_mb)} GB</span>
							{:else}<span class="wb-help">—</span>{/if}
						</td>
						<td class="act">
							{#if isInstalled(file)}
								<span class="wb-badge teal">Installed</span>
							{:else}
								<button
									type="button"
									class="wb-icon-btn"
									disabled={busy}
									onclick={() => handleDownload(file)}
									aria-label="Download {fit.label ?? basename(fit.path)}"
									title="Download"
								>
									{#if busy}<Spinner size="sm" />{:else}<Icon name="download" size={14} />{/if}
								</button>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}

<div class="wb-sec files-sec">
	<button
		type="button"
		class="disclose"
		onclick={() => (filesToggled = !filesOpen)}
		aria-expanded={filesOpen}
	>
		<Icon name="chevron" size={12} class="chev {filesOpen ? 'open' : ''}" />
		<span class="wb-sec-label">All files &amp; formats</span>
		<span class="wb-help ml-auto">
			<span class="num">{files.length}</span>
			{files.length === 1 ? 'file' : 'files'}{#if totalBytes > 0}, <span class="num">{formatBytes(totalBytes)}</span>{/if}
		</span>
	</button>

	{#if filesOpen}
		<ul class="files">
			{#each sortedFiles as file (file.path)}
				{@const isGguf = file.path.endsWith('.gguf')}
				{@const busy = queueing.has(queueKey(file.path))}
				<li>
					<span class="fname" class:gguf={isGguf} title={file.path}>{file.path}</span>
					<span class="num fsize">{formatBytes(file.size)}</span>
					{#if isGguf && !splitGgufPart(file.path)}
						<button
							type="button"
							class="wb-icon-btn sm"
							disabled={busy}
							onclick={() => handleDownload(file)}
							aria-label="Download {basename(file.path)}"
							title="Download"
						>
							{#if busy}<Spinner size="sm" />{:else}<Icon name="download" size={13} />{/if}
						</button>
					{:else}
						<span class="nodl" title={splitGgufPart(file.path) ? 'Download the complete set from All quants.' : 'Single files can only be downloaded as GGUF. Use Download repo for the rest.'}>
							<Icon name="download" size={13} />
						</span>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style>
	.best {
		padding: 12px;
		border-radius: 10px;
		border: 1px solid color-mix(in oklch, var(--primary) 35%, transparent);
		background: color-mix(in oklch, var(--primary) 6%, transparent);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
	}
	.placement {
		height: 10px;
	}
	.legend {
		display: flex;
		justify-content: space-between;
		margin-top: 6px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.lg {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.lg i {
		width: 8px;
		height: 8px;
		border-radius: 2px;
	}
	.quants {
		padding: 14px 0 6px;
	}
	.qt td {
		height: 44px;
	}
	.qt td:first-child {
		white-space: normal;
		overflow-wrap: anywhere;
	}
	.qt tr.best-row td {
		background: color-mix(in oklch, var(--primary) 7%, transparent);
	}
	.qt tr.best-row td:first-child {
		box-shadow: inset 2px 0 0 var(--primary);
	}
	.vt {
		font-size: var(--text-2xs);
		margin-top: 1px;
	}
	.split {
		display: block;
		font-family: var(--font-sans);
		color: var(--muted-foreground);
	}
	.r {
		text-align: right;
	}
	.qt th:nth-child(3),
	.where {
		white-space: normal;
	}
	.where {
		display: table-cell;
	}
	.where .tier-bar.mini {
		display: inline-flex;
		vertical-align: middle;
		width: 52px;
		height: 6px;
	}
	.tier-bar.over {
		box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--error) 60%, transparent);
	}
	.est {
		display: inline-block;
		white-space: nowrap;
		margin-left: 8px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.act {
		width: 1%;
		text-align: right;
	}
	.disclose {
		display: flex;
		width: 100%;
		align-items: center;
		gap: 6px;
		text-align: left;
	}
	.disclose:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
		border-radius: 4px;
	}
	.disclose :global(.chev) {
		color: var(--muted-foreground);
		transition: transform 0.15s ease;
	}
	.disclose :global(.chev.open) {
		transform: rotate(90deg);
	}
	.files {
		display: flex;
		flex-direction: column;
		margin-top: 10px;
		font-size: var(--text-xs);
	}
	.files li {
		display: flex;
		align-items: center;
		gap: 10px;
		min-height: 30px;
		border-bottom: 1px solid color-mix(in oklch, var(--border) 55%, transparent);
	}
	.files li:last-child {
		border-bottom: 0;
	}
	.fname {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-family: var(--font-mono);
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.fname.gguf {
		color: var(--foreground);
	}
	.fsize {
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.wb-icon-btn.sm {
		width: 24px;
		height: 24px;
	}
	.nodl {
		display: inline-grid;
		place-items: center;
		width: 24px;
		height: 24px;
		color: color-mix(in oklch, var(--muted-foreground) 45%, transparent);
	}
	@media (prefers-reduced-motion: reduce) {
		.disclose :global(.chev) {
			transition: none;
		}
	}
</style>
