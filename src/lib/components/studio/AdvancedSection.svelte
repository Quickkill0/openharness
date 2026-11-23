<script lang="ts">
	/**
	 * Advanced disclosure — collapsed by default, open state persisted per task
	 * in localStorage. Numeric controls live in here so the simple view stays
	 * "model, prompt, Generate".
	 */
	import type { Snippet } from 'svelte';
	import Icon from '$lib/components/shell/Icon.svelte';

	interface Props {
		/** Storage-key suffix — the task id ('image', 'speech', …). */
		storageKey: string;
		label?: string;
		children: Snippet;
	}

	let { storageKey, label = 'Advanced', children }: Props = $props();

	const key = () => `openharness.studio.advanced.${storageKey}`;

	function initial(): boolean {
		try {
			return typeof localStorage !== 'undefined' && localStorage.getItem(key()) === '1';
		} catch {
			return false;
		}
	}

	let open = $state(initial());

	function toggle() {
		open = !open;
		try {
			localStorage.setItem(key(), open ? '1' : '0');
		} catch {
			// persistence is best-effort
		}
	}
</script>

<div class="adv">
	<button type="button" class="adv-sum" onclick={toggle} aria-expanded={open}>
		<span class="chev" class:open><Icon name="chevron" size={13} /></span>
		{label}
	</button>
	{#if open}
		<div class="adv-body">
			{@render children()}
		</div>
	{/if}
</div>

<style>
	.adv {
		border-top: 1px solid var(--border);
		margin: 0 -16px;
		padding: 12px 16px 0;
	}
	.adv-sum {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 2px 0;
		border: 0;
		background: transparent;
		font-size: var(--text-xs);
		font-weight: 600;
		color: var(--foreground);
		cursor: pointer;
	}
	.adv-sum:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
		border-radius: 4px;
	}
	.chev {
		display: inline-flex;
		color: var(--muted-foreground);
		transition: transform 0.2s ease;
	}
	.chev.open {
		transform: rotate(90deg);
	}
	.adv-body {
		display: flex;
		flex-direction: column;
		gap: 16px;
		padding-top: 14px;
	}
	@media (prefers-reduced-motion: reduce) {
		.chev {
			transition: none;
		}
	}
</style>
