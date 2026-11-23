<script lang="ts">
	/**
	 * Settings, Engines (M16 W6, restyled for Workbench phase A).
	 *
	 * One ruled EngineCard row per runtime engine from the engines.svelte.ts
	 * adapter. Graphics-memory details and the loaded-model list moved to
	 * Settings, Hardware; the dock shows them on every page. The old
	 * /settings/engine and /settings/runtime paths redirect here.
	 */
	import { onMount } from 'svelte';
	import EngineCard from '$lib/components/engine/EngineCard.svelte';
	import InstallWizard from '$lib/components/engine/InstallWizard.svelte';
	import { enginesStore, enginesUi, initEnginesStore } from '$lib/stores/engines.svelte';

	const engines = $derived(enginesStore.engines);

	onMount(() => {
		void initEnginesStore();
	});
</script>

<div class="wb-set-h">
	<h2>Engines</h2>
	<p>What runs your models. The Python worker sets itself up; install llama.cpp here. The native engine is placed by hand for now.</p>
</div>

{#each engines as engine (engine.id)}
	<EngineCard {engine} />
{:else}
	<p class="px-5 py-4 text-sm text-muted-foreground">Reading engine status…</p>
{/each}

<InstallWizard bind:open={enginesUi.llamaWizardOpen} />
