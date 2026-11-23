<script lang="ts">
	/**
	 * Workbench rail — 72px icon-over-label navigation. Models, Store, Chat and
	 * Studio sit at the top; Server and Settings are pinned to the bottom.
	 */
	import { page } from '$app/state';
	import Icon, { type IconName } from './Icon.svelte';

	interface RailLink {
		href: string;
		label: string;
		icon: IconName;
		/** Returns true when this entry owns the current path. */
		match: (path: string) => boolean;
	}

	const TOP: RailLink[] = [
		{
			href: '/models',
			label: 'Models',
			icon: 'models',
			match: (p) => p === '/models' || (p.startsWith('/models/') && !p.startsWith('/models/browse')),
		},
		{
			href: '/models/browse',
			label: 'Store',
			icon: 'store',
			match: (p) => p.startsWith('/models/browse'),
		},
		{ href: '/chat', label: 'Chat', icon: 'chat', match: (p) => p.startsWith('/chat') },
		{ href: '/studio', label: 'Studio', icon: 'studio', match: (p) => p.startsWith('/studio') },
	];

	const BOTTOM: RailLink[] = [
		{ href: '/server', label: 'Server', icon: 'server', match: (p) => p.startsWith('/server') },
		{
			href: '/settings',
			label: 'Settings',
			icon: 'settings',
			match: (p) => p.startsWith('/settings'),
		},
	];

	const path = $derived(page.url.pathname);
</script>

{#snippet item(link: RailLink)}
	{@const active = link.match(path)}
	<a
		href={link.href}
		class="rail-item {active ? 'active' : ''}"
		aria-current={active ? 'page' : undefined}
	>
		<Icon name={link.icon} size={20} />
		<span>{link.label}</span>
	</a>
{/snippet}

<nav class="rail" aria-label="Main">
	<div class="brand" title="OpenHarness">
		<span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
		<span class="sr-only">OpenHarness</span>
	</div>
	{#each TOP as link (link.href)}
		{@render item(link)}
	{/each}
	<div class="flex-1"></div>
	{#each BOTTOM as link (link.href)}
		{@render item(link)}
	{/each}
</nav>

<style>
	.rail {
		display: flex;
		flex-direction: column;
		align-items: stretch;
		gap: 2px;
		height: 100%;
		padding: 8px 0;
		background: var(--surface);
		border-right: 1px solid var(--border);
	}
	.brand {
		display: grid;
		place-items: center;
		height: 40px;
		margin-bottom: 6px;
	}
	/* Three stacked bars in the tier colours: VRAM, RAM, SSD. */
	.brand-mark {
		display: grid;
		grid-template-rows: 1fr 1fr 1fr;
		gap: 2px;
		width: 24px;
		height: 24px;
		padding: 4px;
		border-radius: 7px;
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
	}
	.brand-mark i {
		display: block;
		border-radius: 2px;
	}
	.brand-mark i:nth-child(1) {
		background: var(--tier-vram);
	}
	.brand-mark i:nth-child(2) {
		background: var(--tier-ram);
		width: 75%;
	}
	.brand-mark i:nth-child(3) {
		background: var(--tier-ssd);
		width: 45%;
	}
	.rail-item {
		position: relative;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 3px;
		margin: 0 8px;
		padding: 9px 0 7px;
		border-radius: 10px;
		color: var(--muted-foreground);
		font-size: var(--text-2xs);
		font-weight: 500;
		text-decoration: none;
	}
	.rail-item:hover {
		color: var(--foreground);
		background: var(--elevated);
	}
	.rail-item:active {
		transform: translateY(1px);
	}
	.rail-item.active {
		color: var(--foreground);
		background: var(--elevated);
		box-shadow:
			0 0 0 1px color-mix(in oklch, var(--primary) 45%, transparent),
			0 0 18px color-mix(in oklch, var(--primary) 22%, transparent);
	}
	.rail-item.active :global(svg) {
		color: var(--primary);
	}
	.rail-item.active::before {
		content: '';
		position: absolute;
		left: -8px;
		top: 12px;
		bottom: 12px;
		width: 3px;
		border-radius: 0 3px 3px 0;
		background: var(--primary);
	}
</style>
