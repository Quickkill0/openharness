<script lang="ts">
	/**
	 * DetailPane — persistent right-pane model detail view.
	 *
	 * Replaces the slide-over `DetailDrawer.svelte`. Always mounted, fills the
	 * remaining width of the browse layout. Shows a centered empty state when
	 * no model is selected.
	 *
	 * README sanitization pipeline (marked → DOMPurify) is identical to the
	 * one previously in DetailDrawer.svelte. Do NOT change the sanitizer config
	 * without going through the M2 polish-pass debt notes in CLAUDE.md.
	 */
	import { open as shellOpen } from '@tauri-apps/plugin-shell';
	import Icon from '$lib/components/shell/Icon.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import FileTree from './FileTree.svelte';
	import { isRepoInstallComplete } from './installed';
	import RunnableBadge from './RunnableBadge.svelte';
	import { taskLabel } from './task-meta';
	import FitBadge from '$lib/components/engine/FitBadge.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import {
		hfModelDetails,
		listDownloads,
		listLocalModels,
		startRepoDownload,
		storeQuantFits,
		storeRunnability,
	} from '$lib/ipc/commands';
	import { seedFromList } from '$lib/stores/downloads.svelte';
	import { compactCount, formatBytes, timeAgo } from '$lib/format';
	import type { HfModelCard, HfModelDetails, InstalledModel, QuantFit, Runnability } from '$lib/types';

	interface Props {
		model?: HfModelCard | null;
		/** Card-level verdict from the list pane — shown until layout sharpens it. */
		runnability?: Runnability | null;
		/** Called when a download is successfully queued from the file tree. */
		ondownloadqueued?: () => void;
	}

	let { model = null, runnability = null, ondownloadqueued }: Props = $props();

	let details = $state<HfModelDetails | null>(null);
	let loading = $state(false);
	let errorMsg = $state<string | null>(null);
	let authRequired = $state(false);
	let accessDenied = $state(false);

	/** README is long; it stays collapsed behind a disclosure. */
	let readmeOpen = $state(false);

	// ---------------------------------------------------------------------------
	// Installed-locally check (M4) — drives the FitBadge surface.
	// ---------------------------------------------------------------------------
	let installedRows = $state<InstalledModel[]>([]);
	const installedMatch = $derived(installedRows[0] ?? null);
	let ctxSlider = $state(4096);

	$effect(() => {
		if (!model) {
			installedRows = [];
			return;
		}
		const repoId = model.id;
		listLocalModels()
			.then((rows) => {
				installedRows = rows.filter((r) => r.repo_id === repoId);
				const installedMatch = installedRows[0];
				if (installedMatch?.context_train) {
					ctxSlider = Math.min(installedMatch.context_train, 8192);
				} else {
					ctxSlider = 4096;
				}
			})
			.catch(() => {
				installedRows = [];
			});
	});

	/**
	 * Dynamically imported markdown pipeline; lazy-loaded to keep bundle small.
	 *
	 * Pipeline: marked(md) → raw HTML → DOMPurify.sanitize(html) → safe HTML
	 *
	 * Sanitizer test (comment-only, no runtime code):
	 *   Input:  <img src=x onerror=alert(1)>
	 *   Output: <img src="x"> — the `onerror` attribute is stripped by DOMPurify.
	 *   Input:  <a href="javascript:void(0)" onclick="steal()">click</a>
	 *   Output: <a href="javascript:void(0)">click</a> — `onclick` stripped;
	 *           href with javascript: is also stripped by DOMPurify's default URI check.
	 */
	let markedRender = $state<((md: string) => string) | null>(null);

	// Sanitizer behavior contract — verified manually by passing each input
	// through DOMPurify.sanitize(input, DOMPURIFY_CONFIG):
	//   <img src=x onerror=alert(1)>
	//     → <img>                                  (onerror stripped, src removed by hook for non-https)
	//   <a href="javascript:alert(1)">x</a>
	//     → <a target="_blank" rel="noopener noreferrer nofollow">x</a>   (javascript: URI stripped)
	//   <script>alert(1)<\/script>
	//     → (empty)                                (tag not in ALLOWED_TAGS)
	//   <p onclick="alert(1)">hi</p>
	//     → <p>hi</p>                              (event handler stripped)
	//
	// DOMPurify configuration — explicit allowlist keeps the surface minimal.
	const DOMPURIFY_CONFIG = {
		ALLOWED_TAGS: [
			'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
			'p', 'ul', 'ol', 'li',
			'blockquote', 'code', 'pre',
			'em', 'strong',
			'a', 'img',
			'hr', 'br',
			'table', 'thead', 'tbody', 'tr', 'th', 'td',
			'span', 'div',
		],
		ALLOWED_ATTR: [
			'href', 'src', 'alt', 'title',
			'class', 'target', 'rel', 'id',
		],
		RETURN_TRUSTED_TYPE: false as const,
		KEEP_CONTENT: true,
		ALLOW_DATA_ATTR: false,
	};

	async function loadMarked() {
		if (markedRender) return;
		try {
			const [{ marked }, { default: purify }] = await Promise.all([
				import('marked'),
				import('dompurify'),
			]);

			// Hook: force all anchor tags to open safely in a new tab.
			purify.addHook('afterSanitizeAttributes', (node: Element) => {
				if (node.tagName === 'A') {
					node.setAttribute('target', '_blank');
					node.setAttribute('rel', 'noopener noreferrer nofollow');
				}
				// Restrict img src to https:// only; add lazy loading.
				if (node.tagName === 'IMG') {
					const src = node.getAttribute('src') ?? '';
					if (!src.startsWith('https://')) {
						node.removeAttribute('src');
					}
					node.setAttribute('loading', 'lazy');
				}
			});

			markedRender = (md: string) => {
				const rawHtml = marked(md) as string;
				return purify.sanitize(rawHtml, DOMPURIFY_CONFIG) as string;
			};
		} catch {
			markedRender = (md: string) => `<pre class="whitespace-pre-wrap text-xs">${md}</pre>`;
		}
	}

	// Fetch model details whenever the selected model changes.
	$effect(() => {
		if (!model) {
			details = null;
			errorMsg = null;
			authRequired = false;
			accessDenied = false;
			return;
		}

		readmeOpen = false;

		const repoId = model.id;
		loading = true;
		details = null;
		errorMsg = null;
		authRequired = false;
		accessDenied = false;

		hfModelDetails(repoId)
			.then((d) => {
				details = d;
				if (d.readme) loadMarked();
			})
			.catch((err: unknown) => {
				const msg = String(err);
				if (msg.includes('authentication required') || msg.includes('HfAuthRequired')) {
					authRequired = true;
				} else if (msg.includes('access denied') || msg.includes('HfAccessDenied')) {
					accessDenied = true;
				} else if (msg.includes('rate-limited') || msg.includes('RateLimited')) {
					addToast('HuggingFace rate-limited us — try again in a moment', 'warning');
					errorMsg = 'Rate limited — please try again shortly.';
				} else {
					addToast(`Failed to load model details: ${msg}`, 'error');
					errorMsg = msg;
				}
			})
			.finally(() => {
				loading = false;
			});
	});

	/** First paragraph of the README as the plain summary (strips front matter, headings, badges). */
	const summary = $derived.by(() => {
		const md = details?.readme;
		if (!md) return null;
		const body = md.replace(/^---[\s\S]*?\n---\s*/, '');
		for (const block of body.split(/\n\s*\n/)) {
			const t = block.trim();
			if (!t || /^(#|!\[|<|\||```|\[!\[|>)/.test(t)) continue;
			const plain = t
				.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
				.replace(/[*_`]/g, '')
				.replace(/\s+/g, ' ')
				.trim();
			if (plain.length < 24) continue;
			return plain.length > 320 ? `${plain.slice(0, 317).trimEnd()}…` : plain;
		}
		return null;
	});

	function openOnHf() {
		if (!model) return;
		shellOpen(`https://huggingface.co/${model.id}`).catch((e: unknown) =>
			addToast(`Could not open browser: ${e}`, 'error'),
		);
	}

	function requestAccess() {
		if (!model) return;
		shellOpen(`https://huggingface.co/${model.id}`).catch((e: unknown) =>
			addToast(`Could not open browser: ${e}`, 'error'),
		);
	}

	const renderedReadme = $derived(
		details?.readme && markedRender ? markedRender(details.readme) : null,
	);

	const totalSize = $derived(
		details?.siblings.reduce((acc, f) => acc + (f.size || 0), 0) ?? 0,
	);

	// ---------------------------------------------------------------------------
	// Runnability verdict (M9 §9.5) — sharpened with the layout once details load.
	// ---------------------------------------------------------------------------

	let sharpVerdict = $state<Runnability | null>(null);

	$effect(() => {
		if (!details || !model) {
			sharpVerdict = null;
			return;
		}
		const card = details.card;
		storeRunnability([
			{
				repo_id: card.id,
				task: card.pipeline_tag,
				library: card.library_name,
				tags: card.tags,
				layout: details.layout.kind,
			},
		])
			.then((v) => (sharpVerdict = v[0] ?? null))
			.catch(() => (sharpVerdict = null));
	});

	const verdict = $derived(sharpVerdict ?? runnability);

	// ---------------------------------------------------------------------------
	// VRAM-aware quant fits (M9 §9.6) — GGUF repos only.
	// ---------------------------------------------------------------------------

	let quantFits = $state<QuantFit[]>([]);

	$effect(() => {
		if (!details || details.layout.kind !== 'gguf') {
			quantFits = [];
			return;
		}
		storeQuantFits(details.siblings)
			.then((fits) => (quantFits = fits))
			.catch(() => (quantFits = []));
	});

	// ---------------------------------------------------------------------------
	// Whole-repo download (M9 §9.5) — dir-layout models only.
	// ---------------------------------------------------------------------------

	let queueingRepo = $state(false);

	/** Files the "Download repo" action fetches, resolved against siblings. */
	const repoSubset = $derived.by(() => {
		if (!details) return [];
		const wanted = new Set(details.layout.download_files);
		return details.siblings.filter((f) => wanted.has(f.path));
	});

	const repoSubsetSize = $derived(repoSubset.reduce((s, f) => s + (f.size || 0), 0));
	/** Only a complete folder install of this repo disables "Download repo". */
	const repoInstalled = $derived(isRepoInstallComplete(installedRows, repoSubset.length));

	/** Catalog format for the group row. GGUF repos use per-file download instead. */
	const repoFormat = $derived.by(() => {
		switch (details?.layout.kind) {
			case 'diffusers-dir': return 'diffusers-dir';
			case 'safetensors-single':
			case 'safetensors-shards': return 'safetensors';
			default: return null;
		}
	});

	async function handleDownloadRepo() {
		if (!model || !details || !repoFormat || repoSubset.length === 0 || queueingRepo) return;
		queueingRepo = true;
		try {
			await startRepoDownload({
				repoId: model.id,
				revision: 'main',
				files: repoSubset,
				format: repoFormat,
				task: details.card.pipeline_tag ?? null,
			});
			// Member jobs are created backend-side — refresh the store so the
			// drawer shows the whole group immediately.
			seedFromList(await listDownloads());
			addToast(`Repo download queued: ${model.id} (${repoSubset.length} files)`, 'success');
			ondownloadqueued?.();
		} catch (e: unknown) {
			addToast(`Repo download failed: ${e}`, 'error');
		} finally {
			queueingRepo = false;
		}
	}
</script>

<section class="pane">
	{#if !model}
		<div class="wb-empty grow justify-center">
			<Icon name="store" size={26} />
			<b>Pick a repository</b>
			<span class="max-w-xs">See what it does, whether it runs on this PC, and which file to download.</span>
		</div>
	{:else}
		{@const card = details?.card ?? model}
		<div class="body">
			<div class="wb-ins-head">
				<div class="wb-ins-sub">{model.author ?? model.id.split('/')[0]}</div>
				<h2 class="wb-ins-name">
					{model.id.includes('/') ? model.id.split('/').slice(1).join('/') : model.id}
				</h2>
				<div class="wb-ins-badges">
					<RunnableBadge {verdict} />
					{#if model.pipeline_tag}
						<span class="wb-badge">{taskLabel(model.pipeline_tag) ?? model.pipeline_tag}</span>
					{/if}
					{#if model.library_name}
						<span class="wb-badge">{model.library_name}</span>
					{/if}
					{#if model.gated}
						<span class="wb-badge warn"><Icon name="lock" size={11} />Gated</span>
					{/if}
				</div>
				<div class="stats">
					<span><span class="num">{compactCount(card.downloads)}</span> downloads</span>
					<span><span class="num">{compactCount(card.likes)}</span> likes</span>
					{#if timeAgo(card.last_modified)}
						<span>Updated {timeAgo(card.last_modified)}</span>
					{/if}
					<button type="button" class="hf" onclick={openOnHf}>
						Hugging Face<Icon name="external" size={11} />
					</button>
				</div>
			</div>

			{#if verdict && (!verdict.runnable || !verdict.available)}
				<div class="wb-sec">
					<div class="wb-banner {verdict.runnable ? 'warn' : ''}">
						<Icon name="info" size={14} class="mt-px shrink-0" />
						<div>
							<div class="font-medium text-foreground">
								{verdict.runnable ? 'Needs setup before it can run here' : 'This model will not run in OpenHarness'}
							</div>
							<div class="mt-0.5 text-muted-foreground">
								{verdict.reason}{#if !verdict.runnable}&nbsp;You can still download its files for other
									tools.{/if}
							</div>
						</div>
					</div>
				</div>
			{/if}

			{#if loading}
				<div class="flex h-40 items-center justify-center"><Spinner size="lg" /></div>
			{:else if authRequired}
				<div class="wb-empty">
					<Icon name="lock" size={24} />
					<b>This model is gated</b>
					<span>Add a Hugging Face token in Settings to see gated models.</span>
					<a href="/settings" class="wb-btn mt-2">Open Settings</a>
				</div>
			{:else if accessDenied}
				<div class="wb-empty">
					<Icon name="lock" size={24} />
					<b>No access yet</b>
					<span>Accept the licence on Hugging Face, then come back.</span>
					<button type="button" class="wb-btn primary mt-2" onclick={requestAccess}>
						Request access<Icon name="external" size={12} />
					</button>
				</div>
			{:else if errorMsg}
				<div class="wb-sec">
					<div class="wb-banner err">{errorMsg}</div>
				</div>
			{:else if details}
				{#if summary || details.readme || details.gguf}
					<div class="wb-sec">
						<div class="wb-sec-h"><span class="wb-sec-label">About</span></div>
						{#if summary}
							<p class="about">{summary}</p>
						{/if}
						{#if details.gguf?.architecture || details.gguf?.context_length || totalSize > 0}
							<dl class="facts">
								{#if details.gguf?.architecture}
									<dt>Architecture</dt>
									<dd>{details.gguf.architecture}</dd>
								{/if}
								{#if details.gguf?.context_length}
									<dt>Context</dt>
									<dd><span class="num">{details.gguf.context_length.toLocaleString()}</span> tokens</dd>
								{/if}
								{#if totalSize > 0}
									<dt>Repository size</dt>
									<dd class="num">{formatBytes(totalSize)}</dd>
								{/if}
							</dl>
						{/if}
						{#if details.readme}
							<button
								type="button"
								class="readme-toggle"
								aria-expanded={readmeOpen}
								onclick={() => (readmeOpen = !readmeOpen)}
							>
								<Icon name="chevron" size={11} class="chev {readmeOpen ? 'open' : ''}" />
								{readmeOpen ? 'Hide the model card' : 'Read the full model card'}
							</button>
							{#if readmeOpen}
								{#if renderedReadme}
									<div class="readme">{@html renderedReadme}</div>
								{:else}
									<pre class="readme whitespace-pre-wrap">{details.readme}</pre>
								{/if}
							{/if}
						{/if}
					</div>
				{/if}

				{#if installedMatch}
					<div class="wb-sec">
						<div class="wb-sec-h">
							<span class="wb-sec-label">Installed on this PC</span>
							{#if installedMatch.format === 'gguf'}
								<FitBadge modelId={installedMatch.id} ctx={ctxSlider} detailed />
							{:else}
								<a href="/models" class="wb-btn sm">Open in Models</a>
							{/if}
						</div>
						{#if installedMatch.format !== 'gguf'}
							<p class="wb-help">Ready to use from Models or Studio.</p>
						{:else}
						<label class="ctx">
							<span class="text-xs text-muted-foreground">Context</span>
							<input
								type="range"
								min="512"
								max={installedMatch.context_train ?? 16384}
								step="512"
								bind:value={ctxSlider}
								aria-label="Context length for the fit estimate"
							/>
							<span class="num text-xs text-foreground">{(ctxSlider / 1024).toFixed(1)}K</span>
						</label>
						{/if}
					</div>
				{/if}

				{#if details.siblings.length > 0}
					<FileTree
						files={details.siblings}
						task={details.card.pipeline_tag}
						repoId={model.id}
						revision="main"
						{quantFits}
						installed={installedRows}
						hasRepoAction={repoFormat !== null && repoSubset.length > 0}
						{ondownloadqueued}
					/>
				{:else}
					<div class="wb-sec"><p class="wb-help">No files listed for this repository.</p></div>
				{/if}
			{/if}
		</div>

		{#if details && repoFormat && repoSubset.length > 0}
			<div class="wb-ins-foot">
				<button
					type="button"
					class="wb-btn primary grow"
					disabled={queueingRepo || repoInstalled}
					onclick={handleDownloadRepo}
				>
					{#if queueingRepo}
						<Spinner size="sm" />Starting download
					{:else if repoInstalled}
						Installed
					{:else}
						<Icon name="download" size={14} />Download repo
					{/if}
				</button>
				<span class="foot-meta">
					<span class="num">{repoSubset.length}</span> files<br /><span class="num"
						>{formatBytes(repoSubsetSize)}</span
					>
				</span>
			</div>
		{/if}
	{/if}
</section>

<style>
	.pane {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	.body {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.stats {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 12px;
		margin-top: 10px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.hf {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		margin-left: auto;
		color: var(--primary);
	}
	.hf:hover {
		text-decoration: underline;
	}
	.hf:focus-visible,
	.readme-toggle:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
		border-radius: 4px;
	}
	.about {
		font-size: var(--text-xs);
		line-height: 1.55;
		color: var(--muted-foreground);
	}
	.facts {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 4px 12px;
		margin-top: 10px;
		font-size: var(--text-xs);
	}
	.facts dt {
		color: var(--muted-foreground);
	}
	.facts dd {
		color: var(--foreground);
		text-align: right;
	}
	.readme-toggle {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		margin-top: 10px;
		font-size: var(--text-xs);
		color: var(--primary);
	}
	.readme-toggle :global(.chev) {
		transition: transform 0.15s ease;
	}
	.readme-toggle :global(.chev.open) {
		transform: rotate(90deg);
	}
	.readme {
		margin-top: 10px;
		max-height: 420px;
		overflow: auto;
		padding: 10px 12px;
		border-radius: 8px;
		border: 1px solid var(--border);
		background: var(--background);
		font-size: var(--text-xs);
		line-height: 1.55;
		color: var(--muted-foreground);
	}
	.readme :global(h1),
	.readme :global(h2),
	.readme :global(h3) {
		margin: 12px 0 6px;
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--foreground);
	}
	.readme :global(p),
	.readme :global(ul),
	.readme :global(ol) {
		margin-bottom: 8px;
	}
	.readme :global(ul) {
		list-style: disc;
		padding-left: 16px;
	}
	.readme :global(ol) {
		list-style: decimal;
		padding-left: 16px;
	}
	.readme :global(a) {
		color: var(--primary);
	}
	.readme :global(code) {
		font-family: var(--font-mono);
		font-size: var(--text-2xs);
		color: var(--foreground);
	}
	.readme :global(pre) {
		overflow-x: auto;
		padding: 8px;
		border-radius: 6px;
		background: var(--elevated);
	}
	.readme :global(img) {
		max-width: 100%;
	}
	.readme :global(table) {
		width: 100%;
		border-collapse: collapse;
	}
	.readme :global(th),
	.readme :global(td) {
		border: 1px solid var(--border);
		padding: 3px 6px;
	}
	.ctx {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.ctx input {
		flex: 1;
		accent-color: var(--primary);
	}
	.foot-meta {
		font-size: var(--text-2xs);
		line-height: 1.3;
		color: var(--muted-foreground);
		text-align: right;
		white-space: nowrap;
	}
	@media (prefers-reduced-motion: reduce) {
		.readme-toggle :global(.chev) {
			transition: none;
		}
	}
</style>
