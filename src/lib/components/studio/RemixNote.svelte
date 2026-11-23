<script lang="ts">
	/** Why a "reuse settings" only partly restored a past generation. */
	import Icon from '$lib/components/shell/Icon.svelte';

	interface Props {
		note: string | null;
		ondismiss: () => void;
	}

	let { note, ondismiss }: Props = $props();
</script>

{#if note}
	<div class="wb-banner warn" role="status">
		<Icon name="info" size={14} />
		<span class="flex-1">{note}</span>
		<button type="button" class="dismiss" onclick={ondismiss} aria-label="Dismiss">
			<Icon name="x" size={12} />
		</button>
	</div>
{/if}

<style>
	.dismiss {
		display: grid;
		place-items: center;
		width: 20px;
		height: 20px;
		flex: none;
		border-radius: 5px;
		color: inherit;
		opacity: 0.7;
		transition: opacity 0.15s ease, background 0.15s ease;
	}
	.dismiss:hover {
		opacity: 1;
		background: color-mix(in oklch, currentColor 12%, transparent);
	}
	.dismiss:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 1px;
	}
</style>
