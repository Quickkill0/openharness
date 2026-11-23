<script lang="ts" module>
	export interface MenuItem {
		label: string;
		onselect: () => void;
		disabled?: boolean;
		/** Shown under the label (e.g. why it's disabled). */
		hint?: string;
		danger?: boolean;
		/** Draw a divider above this item. */
		separator?: boolean;
	}
</script>

<script lang="ts">
	/**
	 * Small anchored action menu (the Models table's ⋯ menu). Fixed-positioned
	 * from the anchor's rect so table overflow can't clip it; flips upward when
	 * it would run into the dock. Closes on Escape, outside click, or select.
	 */
	import { tick } from 'svelte';

	interface Props {
		/** Element the menu hangs from; null = closed. */
		anchor: HTMLElement | null;
		items: MenuItem[];
		onclose: () => void;
		label?: string;
	}

	let { anchor, items, onclose, label = 'Actions' }: Props = $props();

	let menuEl = $state<HTMLDivElement | null>(null);
	let pos = $state({ top: 0, left: 0 });

	$effect(() => {
		if (!anchor) return;
		const a = anchor;
		void tick().then(() => {
			const r = a.getBoundingClientRect();
			const h = menuEl?.offsetHeight ?? 200;
			const w = menuEl?.offsetWidth ?? 208;
			const below = r.bottom + 4;
			// 60px dock + a little air.
			const top = below + h > window.innerHeight - 68 ? Math.max(8, r.top - h - 4) : below;
			pos = { top, left: Math.max(8, Math.min(r.right - w, window.innerWidth - w - 8)) };
			menuEl?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus();
		});
	});

	function onWindowPointer(e: PointerEvent) {
		if (!anchor) return;
		const t = e.target as Node;
		if (menuEl?.contains(t) || anchor.contains(t)) return;
		onclose();
	}

	function onKey(e: KeyboardEvent) {
		if (!anchor || !menuEl) return;
		if (e.key === 'Escape') {
			e.preventDefault();
			const a = anchor;
			onclose();
			a.focus();
			return;
		}
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			const btns = [...menuEl.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')];
			const i = btns.indexOf(document.activeElement as HTMLButtonElement);
			const next = e.key === 'ArrowDown' ? (i + 1) % btns.length : (i - 1 + btns.length) % btns.length;
			btns[next]?.focus();
		}
	}

	function select(item: MenuItem) {
		onclose();
		item.onselect();
	}
</script>

<svelte:window onpointerdown={onWindowPointer} onkeydown={onKey} onresize={onclose} />

{#if anchor}
	<div
		bind:this={menuEl}
		role="menu"
		aria-label={label}
		tabindex="-1"
		class="menu"
		style="top: {pos.top}px; left: {pos.left}px"
	>
		{#each items as item (item.label)}
			{#if item.separator}<div class="sep" role="separator"></div>{/if}
			<button
				type="button"
				role="menuitem"
				class="item {item.danger ? 'danger' : ''}"
				disabled={item.disabled}
				onclick={() => select(item)}
			>
				<span>{item.label}</span>
				{#if item.hint}<span class="hint">{item.hint}</span>{/if}
			</button>
		{/each}
	</div>
{/if}

<style>
	.menu {
		position: fixed;
		z-index: 60;
		min-width: 208px;
		padding: 4px;
		border-radius: 10px;
		border: 1px solid var(--border);
		background: var(--elevated);
		box-shadow:
			inset 0 1px 0 var(--inset-highlight),
			0 12px 32px -8px oklch(0 0 0 / 0.45);
		animation: menu-in 0.12s cubic-bezier(0.2, 0.8, 0.2, 1);
	}
	@keyframes menu-in {
		from {
			opacity: 0;
			transform: translateY(-3px);
		}
	}
	.item {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		width: 100%;
		padding: 7px 10px;
		border-radius: 6px;
		font-size: var(--text-sm);
		color: var(--foreground);
		text-align: left;
	}
	.item:hover:not(:disabled),
	.item:focus-visible {
		background: var(--muted);
		outline: none;
	}
	.item:disabled {
		color: var(--muted-foreground);
		opacity: 0.6;
		cursor: not-allowed;
	}
	.item.danger {
		color: var(--error);
	}
	.hint {
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.sep {
		height: 1px;
		margin: 4px 6px;
		background: var(--border);
	}
	@media (prefers-reduced-motion: reduce) {
		.menu {
			animation: none;
		}
	}
</style>
