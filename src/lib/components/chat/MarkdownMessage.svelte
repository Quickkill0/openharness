<script lang="ts">
	/**
	 * Renders assistant markdown safely (marked → DOMPurify) through the shared
	 * pipeline in `$lib/markdown` — the same parser/sanitizer/CSS scope used by
	 * StreamingMarkdown, so a finished turn renders identically to its live
	 * stream. Decorates fenced code blocks with a copy button after render.
	 */
	import {
		loadMarkdownRenderer,
		decorateCodeBlocks,
		type RenderMarkdown,
	} from '$lib/markdown';
	import './chat-md.css';

	interface Props {
		content: string;
	}

	let { content }: Props = $props();

	let render = $state<RenderMarkdown | null>(null);
	let containerEl = $state<HTMLDivElement | null>(null);

	$effect(() => {
		loadMarkdownRenderer().then((r) => (render = r));
	});

	const html = $derived(render ? render(content) : '');

	// Decorate code blocks with a copy button after the HTML lands in the DOM.
	$effect(() => {
		// touch `html` so this re-runs when content/render changes
		void html;
		const root = containerEl;
		if (!root) return;
		queueMicrotask(() => decorateCodeBlocks(root));
	});
</script>

<div bind:this={containerEl} class="chat-md text-sm leading-relaxed">
	{#if render}
		<!-- eslint-disable-next-line svelte/no-at-html-tags — sanitized via DOMPurify -->
		{@html html}
	{:else}
		<pre class="whitespace-pre-wrap break-words">{content}</pre>
	{/if}
</div>
