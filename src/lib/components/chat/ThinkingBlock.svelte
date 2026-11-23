<script lang="ts">
	/**
	 * Collapsible "Thinking" disclosure for reasoning models (Workbench): a
	 * quiet left-ruled block. Auto-opens while the model reasons, collapses
	 * once the answer starts. Session-scoped UI state; never persisted.
	 */
	import Icon from '$lib/components/shell/Icon.svelte';

	interface Props {
		text: string;
		/** True while thinking deltas are still streaming (before the answer). */
		live?: boolean;
	}

	let { text, live = false }: Props = $props();

	let userToggled = $state<boolean | null>(null);
	const open = $derived(userToggled ?? live);
</script>

<div class="think">
	<button type="button" onclick={() => (userToggled = !open)} aria-expanded={open}>
		<Icon name="chevron" size={12} class="chev {open ? 'open' : ''}" />
		<span class={live ? 'live' : ''}>{live ? 'Thinking' : 'Thought process'}</span>
	</button>
	{#if open}
		<p>{text}</p>
	{/if}
</div>

<style>
	.think {
		margin-bottom: 12px;
		padding-left: 12px;
		border-left: 2px solid var(--border);
	}
	button {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		border: 0;
		background: transparent;
		padding: 0;
		cursor: pointer;
		font-size: var(--text-xs);
		color: var(--muted-foreground);
	}
	button:hover {
		color: var(--foreground);
	}
	button:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
		border-radius: 4px;
	}
	button :global(.chev) {
		transition: transform 0.2s ease;
	}
	button :global(.chev.open) {
		transform: rotate(90deg);
	}
	.live {
		color: var(--primary);
		animation: think-pulse 1.4s ease-in-out infinite;
	}
	@keyframes think-pulse {
		50% {
			opacity: 0.5;
		}
	}
	p {
		margin: 8px 0 0;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		font-size: var(--text-xs);
		line-height: 1.6;
		color: var(--muted-foreground);
	}
	@media (prefers-reduced-motion: reduce) {
		.live {
			animation: none;
		}
		button :global(.chev) {
			transition: none;
		}
	}
</style>
