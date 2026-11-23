<script lang="ts">
	// Kitchen-sink preview for the OpenHarness design system (M6 §6.1).
	// Renders every UI primitive in the new ai-shuffle-derived style so the
	// tokens + components can be eyeballed in one place.
	import Button from '$lib/components/ui/Button.svelte';
	import Card from '$lib/components/ui/Card.svelte';
	import Input from '$lib/components/ui/Input.svelte';
	import Badge from '$lib/components/ui/Badge.svelte';
	import Toggle from '$lib/components/ui/Toggle.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import Select from '$lib/components/ui/Select.svelte';
	import Tabs from '$lib/components/ui/Tabs.svelte';
	import Modal from '$lib/components/ui/Modal.svelte';
	import Tooltip from '$lib/components/ui/Tooltip.svelte';
	import KbdHint from '$lib/components/ui/KbdHint.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { onMount } from 'svelte';
	import { loadTheme, persistTheme } from '$lib/stores/theme';
	import type { Theme } from '$lib/types';

	let theme = $state<Theme>('system');
	onMount(async () => {
		theme = await loadTheme();
	});
	async function setTheme(t: Theme) {
		theme = t;
		await persistTheme(t);
	}
	const themeOptions: { value: Theme; label: string }[] = [
		{ value: 'dark', label: 'Dark' },
		{ value: 'light', label: 'Light' },
		{ value: 'system', label: 'Match Windows' },
	];
	let previewOpen = $state(false);

	let toggleA = $state(true);
	let toggleB = $state(false);
	let temp = $state(0.7);
	let ctx = $state(4096);
	let selected = $state('qwen');
	let activeTab = $state('model');
	let modalOpen = $state(false);
	let textValue = $state('');

	const modelOptions = [
		{ value: 'qwen', label: 'Qwen2.5 1.5B Instruct' },
		{ value: 'llama', label: 'Llama 3.2 3B' },
		{ value: 'phi', label: 'Phi-3 Mini' },
	];

	const badgeVariants = [
		'default',
		'blue',
		'green',
		'yellow',
		'orange',
		'red',
		'purple',
		'cyan',
	] as const;

	const toolColors = [
		{ name: 'bash', var: '--tool-bash' },
		{ name: 'read', var: '--tool-read' },
		{ name: 'write', var: '--tool-write' },
		{ name: 'edit', var: '--tool-edit' },
		{ name: 'glob', var: '--tool-glob' },
		{ name: 'grep', var: '--tool-grep' },
		{ name: 'web', var: '--tool-web' },
	];

	const surfaceSwatches = [
		{ name: 'background', cls: 'bg-background' },
		{ name: 'surface', cls: 'bg-surface' },
		{ name: 'elevated', cls: 'bg-elevated' },
		{ name: 'muted', cls: 'bg-muted' },
		{ name: 'primary', cls: 'bg-primary' },
		{ name: 'accent-cyan', cls: 'bg-accent-cyan' },
		{ name: 'success', cls: 'bg-success' },
		{ name: 'warning', cls: 'bg-warning' },
		{ name: 'error', cls: 'bg-error' },
		{ name: 'info', cls: 'bg-info' },
	];
</script>

<div class="wb-set-h">
	<h2>Appearance</h2>
</div>

<div class="wb-prop">
	<div class="pl">Theme</div>
	<div class="pv">
		<div class="seg" role="radiogroup" aria-label="Theme">
			{#each themeOptions as opt (opt.value)}
				<button
					type="button"
					role="radio"
					aria-checked={theme === opt.value}
					onclick={() => setTheme(opt.value)}
				>
					{opt.label}
				</button>
			{/each}
		</div>
	</div>
</div>

<div class="wb-prop">
	<div class="pl">
		Component preview
		<small>Every UI primitive, for checking the design tokens</small>
	</div>
	<div class="pv">
		<Button variant="secondary" size="sm" onclick={() => (previewOpen = !previewOpen)}>
			{previewOpen ? 'Hide preview' : 'Show preview'}
		</Button>
	</div>
</div>

{#if previewOpen}
<div class="space-y-10 p-5">
	<!-- Colors -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Color tokens</h2>
		<div class="grid grid-cols-2 gap-3 sm:grid-cols-5">
			{#each surfaceSwatches as s}
				<div class="space-y-1.5">
					<div class="h-12 rounded-[10px] border border-border {s.cls}"></div>
					<p class="font-mono text-2xs text-muted-foreground">{s.name}</p>
				</div>
			{/each}
		</div>
		<h3 class="pt-2 text-xs font-medium text-muted-foreground">Tool colors (M7)</h3>
		<div class="flex flex-wrap gap-3">
			{#each toolColors as t}
				<div class="space-y-1.5">
					<div class="h-10 w-16 rounded-[10px]" style="background: var({t.var})"></div>
					<p class="font-mono text-2xs text-muted-foreground">{t.name}</p>
				</div>
			{/each}
		</div>
	</section>

	<!-- Typography -->
	<section class="space-y-2">
		<h2 class="text-sm font-semibold text-foreground">Typography</h2>
		<p class="text-xl text-foreground">Inter — the quick brown fox (18px)</p>
		<p class="text-lg text-foreground">Inter — the quick brown fox (16px)</p>
		<p class="text-base text-foreground">Inter — the quick brown fox (14px)</p>
		<p class="text-sm text-muted-foreground">Inter — the quick brown fox (13px)</p>
		<p class="text-2xs text-muted-foreground">Inter — the quick brown fox (11px)</p>
		<p class="font-mono text-sm text-foreground">JetBrains Mono — 0123456789 const x = 42;</p>
	</section>

	<!-- Buttons -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Buttons</h2>
		<div class="flex flex-wrap items-center gap-3">
			<Button variant="primary">Primary</Button>
			<Button variant="secondary">Secondary</Button>
			<Button variant="danger">Danger</Button>
			<Button variant="ghost">Ghost</Button>
			<Button variant="primary" disabled>Disabled</Button>
		</div>
		<div class="flex flex-wrap items-center gap-3">
			<Button variant="primary" size="sm">Small</Button>
			<Button variant="primary" size="md">Medium</Button>
			<Button variant="primary" size="lg">Large</Button>
		</div>
	</section>

	<!-- Badges -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Badges</h2>
		<div class="flex flex-wrap gap-2">
			{#each badgeVariants as v}
				<Badge variant={v}>{v}</Badge>
			{/each}
		</div>
	</section>

	<!-- Inputs + Select -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Inputs</h2>
		<div class="grid gap-3 sm:grid-cols-2">
			<Input bind:value={textValue} placeholder="Type something…" />
			<Select bind:value={selected} options={modelOptions} />
		</div>
	</section>

	<!-- Sliders -->
	<section class="space-y-4">
		<h2 class="text-sm font-semibold text-foreground">Sliders</h2>
		<Slider bind:value={temp} min={0} max={2} step={0.05} label="Temperature" />
		<Slider bind:value={ctx} min={512} max={32768} step={512} label="Context length" unit="tok" />
	</section>

	<!-- Toggles -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Toggles</h2>
		<div class="flex items-center gap-6">
			<Toggle bind:checked={toggleA} label="Flash attention" />
			<Toggle bind:checked={toggleB} label="mmap" />
			<Toggle checked={false} disabled label="disabled" />
		</div>
	</section>

	<!-- Tabs -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Tabs</h2>
		<Tabs
			tabs={[
				{ id: 'model', label: 'Model' },
				{ id: 'sampling', label: 'Sampling' },
				{ id: 'system', label: 'System prompt' },
			]}
			bind:active={activeTab}
		/>
		<p class="text-sm text-muted-foreground">Active tab: <span class="font-mono text-foreground">{activeTab}</span></p>
	</section>

	<!-- Cards -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Cards</h2>
		<div class="grid gap-4 sm:grid-cols-2">
			<Card>
				<h3 class="font-medium text-foreground">Neo-glass card</h3>
				<p class="mt-1 text-sm text-muted-foreground">Border + inset highlight + layered shadow.</p>
			</Card>
			<Card hover>
				<h3 class="font-medium text-foreground">Hover card</h3>
				<p class="mt-1 text-sm text-muted-foreground">Lifts on hover.</p>
			</Card>
		</div>
	</section>

	<!-- Misc: spinner, tooltip, kbd, toast, modal -->
	<section class="space-y-3">
		<h2 class="text-sm font-semibold text-foreground">Spinner · Tooltip · Kbd · Toast · Modal</h2>
		<div class="flex flex-wrap items-center gap-6">
			<div class="flex items-center gap-3">
				<Spinner size="sm" />
				<Spinner size="md" />
				<Spinner size="lg" />
			</div>
			<Tooltip text="A helpful hint">
				<Button variant="secondary" size="sm">Hover me</Button>
			</Tooltip>
			<KbdHint keys={['Ctrl', 'K']} />
			<Button variant="ghost" size="sm" onclick={() => addToast('Hello from a toast', 'success')}>
				Show toast
			</Button>
			<Button variant="primary" size="sm" onclick={() => (modalOpen = true)}>Open modal</Button>
		</div>
	</section>
</div>
{/if}

<Modal bind:open={modalOpen} title="Example modal" size="md">
	<p class="text-sm text-muted-foreground">
		This is a token-styled modal using the neo-glass surface, backdrop blur, and the
		<code class="font-mono">cubic-bezier(0.16,1,0.3,1)</code> entrance.
	</p>
	{#snippet footer()}
		<Button variant="ghost" size="sm" onclick={() => (modalOpen = false)}>Cancel</Button>
		<Button variant="primary" size="sm" onclick={() => (modalOpen = false)}>Confirm</Button>
	{/snippet}
</Modal>

<style>
	.seg {
		display: inline-flex;
		gap: 2px;
		padding: 2px;
		border-radius: 8px;
		background: var(--muted);
	}
	.seg button {
		height: 28px;
		padding: 0 12px;
		border-radius: 6px;
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--muted-foreground);
	}
	.seg button:hover {
		color: var(--foreground);
	}
	.seg button[aria-checked='true'] {
		background: var(--elevated);
		color: var(--foreground);
		box-shadow: inset 0 1px 0 var(--inset-highlight), 0 1px 2px oklch(0 0 0 / 0.2);
	}
	.seg button:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 1px;
	}
</style>
