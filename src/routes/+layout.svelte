<script lang="ts">
	import '../app.css';
	import Rail from '$lib/components/shell/Rail.svelte';
	import Dock from '$lib/components/shell/Dock.svelte';
	import Toast from '$lib/components/ui/Toast.svelte';
	import DownloadDrawer from '$lib/components/downloads/DownloadDrawer.svelte';
	import EvictionDialog from '$lib/components/studio/EvictionDialog.svelte';
	import { loadTheme, applyTheme } from '$lib/stores/theme';
	import { uiStore, closeDownloads } from '$lib/stores/ui.svelte';
	import { initEngineStore } from '$lib/stores/engine.svelte';
	import { initGpuStore } from '$lib/stores/gpu.svelte';
	import { initLedgerStore } from '$lib/stores/ledger.svelte';
	import { initSystemStore } from '$lib/stores/system.svelte';
	import { initChatStore } from '$lib/stores/chat.svelte';
	import { initStudio } from '$lib/stores/studio.svelte';
	import { consumeMigrationNotice } from '$lib/ipc/commands';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { onMount } from 'svelte';

	let { children }: { children: import('svelte').Snippet } = $props();

	onMount(async () => {
		// Apply persisted theme on boot. Guarded so a store failure can't abort
		// the rest of onMount (engine/chat/GPU init below).
		try {
			const theme = await loadTheme();
			applyTheme(theme);

			// Watch system preference changes for 'system' mode
			const colorMq = window.matchMedia('(prefers-color-scheme: dark)');
			colorMq.addEventListener('change', async () => {
				const current = await loadTheme();
				if (current === 'system') applyTheme('system');
			});
		} catch (e) {
			console.error('theme init:', e);
		}

		// Dock feeds: engine + GPU telemetry, the VRAM ledger, system RAM.
		initEngineStore();
		initGpuStore();
		void initLedgerStore();
		initSystemStore();

		// Chat store (conversation list + event channel)
		initChatStore();

		// Generation jobs: adopts jobs already running in the backend so the
		// dock's unload guards are right on every page, including after a reload.
		void initStudio();

		// One-off toast if the Hangar→OpenHarness migration ran this launch.
		try {
			if (await consumeMigrationNotice()) {
				addToast(
					'Migrated your data from Hangar AI to OpenHarness — models, engine, and history carried over.',
					'success',
					8000,
				);
			}
		} catch {
			// non-fatal — migration notice is cosmetic
		}
	});
</script>

<!--
	Workbench shell: 72px rail | page, with the 60px dock spanning the full
	width underneath. Pages render their own PageHeader and fill the cell.
-->
<div class="shell bg-background text-foreground">
	<div class="[grid-area:rail] min-h-0">
		<Rail />
	</div>
	<main class="[grid-area:page] min-h-0 min-w-0 overflow-y-auto">
		{@render children()}
	</main>
	<div class="[grid-area:dock] min-w-0">
		<Dock />
	</div>
</div>

<!-- Global toast layer -->
<Toast />

<!-- Global download drawer — persists across all routes -->
<DownloadDrawer bind:open={uiStore.downloadsOpen} onclose={closeDownloads} />

<!-- Global VRAM-eviction prompt — Studio submits and chat voice jobs both land here -->
<EvictionDialog />

<style>
	.shell {
		display: grid;
		grid-template-columns: 72px minmax(0, 1fr);
		grid-template-rows: minmax(0, 1fr) 60px;
		grid-template-areas:
			'rail page'
			'dock dock';
		height: 100vh;
		width: 100vw;
		overflow: hidden;
	}
</style>
