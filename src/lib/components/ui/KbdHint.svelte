<script lang="ts">
	interface Props {
		/** Keys to render, e.g. ['Ctrl', 'K']. */
		keys: string[];
		class?: string;
	}

	let { keys, class: extraClass = '' }: Props = $props();
</script>

<span class="inline-flex items-center gap-1 {extraClass}">
	{#each keys as key}
		<kbd
			class="inline-flex h-5 min-w-5 items-center justify-center rounded-md border border-border
			       bg-elevated px-1.5 font-mono text-2xs font-medium text-muted-foreground shadow-s"
		>
			{key}
		</kbd>
	{/each}
</span>
