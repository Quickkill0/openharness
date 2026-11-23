<script lang="ts">
	/**
	 * Workbench page header: a 48px bar with the page title, an optional muted
	 * meta line beside it, and right-aligned actions.
	 */
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		/** Short muted text after the title (counts, sizes). */
		meta?: string | null;
		/** Right-aligned controls (filter, buttons). */
		actions?: Snippet;
	}

	let { title, meta = null, actions }: Props = $props();
</script>

<header
	class="flex h-12 shrink-0 items-center gap-3 border-b border-border px-4"
>
	<h1 class="text-lg font-semibold tracking-[-0.01em] text-foreground">{title}</h1>
	{#if meta}
		<span class="truncate text-xs text-muted-foreground">{meta}</span>
	{/if}
	{#if actions}
		<div class="ml-auto flex items-center gap-2">
			{@render actions()}
		</div>
	{/if}
</header>
