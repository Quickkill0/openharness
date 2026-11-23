<script lang="ts">
	interface Props {
		/** Add hover elevation (for clickable cards). */
		hover?: boolean;
		class?: string;
		children?: import('svelte').Snippet;
	}

	let { hover = false, class: extraClass = '', children }: Props = $props();
</script>

<div
	class="neo-surface rounded-[14px] p-5 transition-base {hover
		? 'hover:shadow-l hover:-translate-y-0.5'
		: ''} {extraClass}"
>
	{#if children}
		{@render children()}
	{/if}
</div>
