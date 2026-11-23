<script lang="ts">
	type Variant =
		| 'default'
		| 'blue'
		| 'green'
		| 'yellow'
		| 'red'
		| 'purple'
		| 'orange'
		| 'cyan'
		| 'slate';

	interface Props {
		variant?: Variant;
		class?: string;
		children?: import('svelte').Snippet;
	}

	let { variant = 'default', class: extraClass = '', children }: Props = $props();

	// Map each variant to a theme color variable; the chip look (tinted bg +
	// border + colored text) is derived from it via color-mix in the style block.
	const colorVar: Record<Variant, string> = {
		default: '--muted-foreground',
		slate: '--muted-foreground',
		blue: '--info',
		green: '--success',
		yellow: '--warning',
		orange: '--tool-edit',
		red: '--error',
		purple: '--tool-glob',
		cyan: '--accent-cyan',
	};
</script>

<span
	class="chip inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium {extraClass}"
	style="--chip: var({colorVar[variant]})"
>
	{#if children}
		{@render children()}
	{/if}
</span>

<style>
	.chip {
		color: var(--chip);
		background-color: color-mix(in oklch, var(--chip) 15%, transparent);
		border: 1px solid color-mix(in oklch, var(--chip) 30%, transparent);
	}
</style>
