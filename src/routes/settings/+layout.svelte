<script lang="ts">
	/**
	 * Settings shell: Workbench page header, a 200px section sub-nav and a
	 * scrolling property-sheet pane. Each section is its own route so links
	 * like /settings/engines keep working.
	 */
	import type { Snippet } from 'svelte';
	import { page } from '$app/state';
	import PageHeader from '$lib/components/shell/PageHeader.svelte';

	let { children }: { children: Snippet } = $props();

	const SECTIONS = [
		{ href: '/settings/engines', label: 'Engines' },
		{ href: '/settings/hardware', label: 'Hardware' },
		{ href: '/settings/storage', label: 'Storage' },
		{ href: '/settings/appearance', label: 'Appearance' },
		{ href: '/settings/huggingface', label: 'Hugging Face token' },
		{ href: '/settings/about', label: 'About' },
	];

	const path = $derived(page.url.pathname.replace(/\/$/, ''));
</script>

<div class="flex h-full min-h-0 flex-col">
	<PageHeader title="Settings" />
	<div class="settings">
		<nav class="subnav" aria-label="Settings sections">
			{#each SECTIONS as s (s.href)}
				<a href={s.href} aria-current={path === s.href ? 'page' : undefined}>{s.label}</a>
			{/each}
		</nav>
		<div class="pane">
			<div class="body">
				{@render children()}
			</div>
		</div>
	</div>
</div>

<style>
	.settings {
		flex: 1;
		min-height: 0;
		display: grid;
		grid-template-columns: 200px minmax(0, 1fr);
	}
	.subnav {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 10px 8px;
		border-right: 1px solid var(--border);
		background: var(--surface);
	}
	.subnav a {
		display: flex;
		align-items: center;
		height: 32px;
		padding: 0 10px;
		border-radius: 7px;
		font-size: var(--text-sm);
		font-weight: 500;
		color: var(--muted-foreground);
		text-decoration: none;
		transition:
			background 0.15s ease-out,
			color 0.15s ease-out;
	}
	.subnav a:hover {
		background: var(--elevated);
		color: var(--foreground);
	}
	.subnav a[aria-current='page'] {
		background: var(--elevated);
		color: var(--foreground);
		box-shadow:
			inset 0 1px 0 var(--inset-highlight),
			inset 2px 0 0 var(--primary);
	}
	.subnav a:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: -2px;
	}
	.pane {
		min-width: 0;
		overflow-y: auto;
	}
	.body {
		max-width: 880px;
		padding-bottom: 40px;
	}
</style>
