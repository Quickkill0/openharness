<script lang="ts">
	/**
	 * Live markdown for a streaming model turn — formats tokens *as they
	 * arrive* instead of dumping raw text and restyling at end of turn.
	 *
	 * - Same pipeline as MarkdownMessage (`$lib/markdown`: marked → DOMPurify,
	 *   `.chat-md` scope), so the finished stream is pixel-identical to the
	 *   persisted render — no snap when the turn completes.
	 * - Incremental: the text up to the last stable block boundary is parsed
	 *   once and cached in its own container (`findStableBoundary`); only the
	 *   small active tail re-parses per update, with incomplete constructs
	 *   repaired (`repairTail`) so an open ``` fence renders as a code block
	 *   from its first token, unclosed bold / inline-code markers keep their
	 *   final style, and unstable fragments (partial table header, dangling
	 *   `[link`) are held back as plain text until they settle.
	 * - Updates coalesce through requestAnimationFrame, so a fast token stream
	 *   re-renders at most once per frame.
	 * - The blinking ▋ cursor is injected inline after the last rendered text
	 *   node (inside the current paragraph / list item / code block).
	 */
	import {
		loadMarkdownRenderer,
		decorateCodeBlocks,
		findStableBoundary,
		repairTail,
		type RenderMarkdown,
	} from '$lib/markdown';
	import './chat-md.css';

	interface Props {
		/** Accumulating raw markdown text (append-only while streaming). */
		content: string;
	}

	let { content }: Props = $props();

	let render = $state<RenderMarkdown | null>(null);
	let prefixHtml = $state('');
	let tailHtml = $state('');
	let inlinePlain = $state('');
	let blockPlain = $state('');
	let prefixEl = $state<HTMLDivElement | null>(null);
	let tailEl = $state<HTMLDivElement | null>(null);

	/** Chars of `content` already baked into prefixHtml (non-reactive cache). */
	let prefixLen = 0;
	let raf = 0;
	let latest = '';

	// Floating cursor + held-back inline text, moved to the end of the
	// rendered stream after every update ({@html} rebuilds wipe them).
	const plainNode =
		typeof document !== 'undefined' ? document.createTextNode('') : null;
	const cursorEl =
		typeof document !== 'undefined' ? document.createElement('span') : null;
	if (cursorEl) {
		cursorEl.className = 'stream-cursor';
		cursorEl.textContent = '▋';
		cursorEl.setAttribute('aria-hidden', 'true');
	}

	$effect(() => {
		loadMarkdownRenderer().then((r) => (render = r));
	});

	function update(text: string, r: RenderMarkdown) {
		if (prefixLen > text.length) {
			// Stream reset (new turn reusing the component) — drop the cache.
			prefixLen = 0;
			prefixHtml = '';
		}
		const boundary = findStableBoundary(text);
		if (boundary > prefixLen) {
			// Boundaries sit between independent blocks, so the new stable part
			// parses alone and appends to the cached prefix HTML.
			prefixHtml += r(text.slice(prefixLen, boundary));
			prefixLen = boundary;
		}
		const t = repairTail(text.slice(prefixLen));
		tailHtml = t.md.trim() ? r(t.md) : '';
		inlinePlain = t.inlinePlain;
		blockPlain = t.blockPlain;
	}

	$effect(() => {
		const text = content;
		const r = render;
		latest = text;
		if (!r || raf) return;
		raf = requestAnimationFrame(() => {
			raf = 0;
			update(latest, r);
		});
	});

	$effect(() => {
		return () => {
			if (raf) cancelAnimationFrame(raf);
		};
	});

	/**
	 * Deepest last content position in `root`: descends last element children
	 * (skipping whitespace-only text, the copy button, and the cursor/plain
	 * nodes themselves) so the cursor lands inside the active paragraph, list
	 * item, table cell, or code block.
	 */
	function lastLeaf(root: HTMLElement): HTMLElement {
		let cur: HTMLElement = root;
		for (;;) {
			let n = cur.lastChild;
			while (
				n &&
				(n === cursorEl ||
					n === plainNode ||
					(n.nodeType === Node.TEXT_NODE && !n.textContent?.trim()) ||
					(n instanceof HTMLElement &&
						(n.classList.contains('code-copy') ||
							n.tagName === 'HR' ||
							n.tagName === 'BR')))
			) {
				n = n.previousSibling;
			}
			if (n instanceof HTMLElement) {
				cur = n;
				continue;
			}
			return cur;
		}
	}

	// After each render: decorate code blocks and move the inline plain text +
	// cursor to the end of the streamed content.
	$effect(() => {
		void prefixHtml;
		void tailHtml;
		void inlinePlain;
		void blockPlain;
		const prefix = prefixEl;
		const tail = tailEl;
		queueMicrotask(() => {
			if (prefix) decorateCodeBlocks(prefix);
			if (tail) decorateCodeBlocks(tail);
			if (!plainNode || !cursorEl) return;
			plainNode.textContent = inlinePlain;
			if (blockPlain) {
				// Template renders the held-back block with its own cursor.
				plainNode.remove();
				cursorEl.remove();
				return;
			}
			const host = tail ?? prefix;
			if (!host) return;
			const leaf = lastLeaf(host);
			leaf.appendChild(plainNode);
			leaf.appendChild(cursorEl);
		});
	});
</script>

<div class="chat-md text-sm leading-relaxed">
	{#if prefixHtml}
		<div bind:this={prefixEl}>
			<!-- eslint-disable-next-line svelte/no-at-html-tags — sanitized via DOMPurify -->
			{@html prefixHtml}
		</div>
	{/if}
	{#if tailHtml}
		<div bind:this={tailEl}>
			<!-- eslint-disable-next-line svelte/no-at-html-tags — sanitized via DOMPurify -->
			{@html tailHtml}
		</div>
	{/if}
	{#if blockPlain}
		<p class="whitespace-pre-wrap">{blockPlain}<span class="stream-cursor" aria-hidden="true">▋</span></p>
	{:else if !tailHtml && !prefixHtml}
		<p>{inlinePlain}<span class="stream-cursor" aria-hidden="true">▋</span></p>
	{/if}
</div>
