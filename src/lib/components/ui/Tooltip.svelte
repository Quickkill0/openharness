<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		text: string;
		placement?: 'top' | 'bottom' | 'left' | 'right';
		class?: string;
		children?: Snippet;
	}

	let { text, placement = 'top', class: extraClass = '', children }: Props = $props();
</script>

<span class="oh-tip relative inline-flex {extraClass}">
	{@render children?.()}
	<span class="oh-tip-bubble glass oh-tip-{placement}" role="tooltip">{text}</span>
</span>

<style>
	.oh-tip-bubble {
		position: absolute;
		z-index: 60;
		white-space: nowrap;
		padding: 0.25rem 0.5rem;
		border-radius: 8px;
		font-size: 0.6875rem;
		font-weight: 500;
		color: var(--foreground);
		box-shadow: var(--shadow-m);
		opacity: 0;
		pointer-events: none;
		transform: translateY(2px);
		transition: opacity 150ms cubic-bezier(0.4, 0, 0.2, 1), transform 150ms cubic-bezier(0.4, 0, 0.2, 1);
	}

	.oh-tip:hover .oh-tip-bubble,
	.oh-tip:focus-within .oh-tip-bubble {
		opacity: 1;
		transform: translateY(0);
	}

	.oh-tip-top {
		bottom: 100%;
		left: 50%;
		margin-bottom: 6px;
		transform: translateX(-50%) translateY(2px);
	}
	.oh-tip:hover .oh-tip-top,
	.oh-tip:focus-within .oh-tip-top {
		transform: translateX(-50%) translateY(0);
	}

	.oh-tip-bottom {
		top: 100%;
		left: 50%;
		margin-top: 6px;
		transform: translateX(-50%) translateY(-2px);
	}
	.oh-tip:hover .oh-tip-bottom,
	.oh-tip:focus-within .oh-tip-bottom {
		transform: translateX(-50%) translateY(0);
	}

	.oh-tip-left {
		right: 100%;
		top: 50%;
		margin-right: 6px;
		transform: translateY(-50%) translateX(2px);
	}
	.oh-tip:hover .oh-tip-left,
	.oh-tip:focus-within .oh-tip-left {
		transform: translateY(-50%) translateX(0);
	}

	.oh-tip-right {
		left: 100%;
		top: 50%;
		margin-left: 6px;
		transform: translateY(-50%) translateX(-2px);
	}
	.oh-tip:hover .oh-tip-right,
	.oh-tip:focus-within .oh-tip-right {
		transform: translateY(-50%) translateX(0);
	}
</style>
