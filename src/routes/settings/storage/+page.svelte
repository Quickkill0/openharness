<script lang="ts">
	/**
	 * Settings, Storage: where OpenHarness keeps things and how much the model
	 * library takes. Disk free space isn't measured by the backend yet, so the
	 * mockup's disk bar is left out rather than faked.
	 */
	import { onMount } from 'svelte';
	import { getAppPaths, listLocalModels } from '$lib/ipc/commands';
	import { formatBytes } from '$lib/format';
	import { sizeOnDisk } from '$lib/components/models/modelMeta';
	import type { AppPaths } from '$lib/types';

	let paths = $state<AppPaths | null>(null);
	let library = $state<{ count: number; bytes: number } | null>(null);
	let error = $state<string | null>(null);

	onMount(async () => {
		const [p, m] = await Promise.allSettled([getAppPaths(), listLocalModels()]);
		if (p.status === 'fulfilled') paths = p.value;
		else error = String(p.reason);
		if (m.status === 'fulfilled') {
			library = { count: m.value.length, bytes: m.value.reduce((s, x) => s + sizeOnDisk(x), 0) };
		}
	});

	const rows = $derived(
		paths
			? [
					{ label: 'Models folder', hint: 'Downloaded weights', value: paths.models_dir },
					{ label: 'Engines folder', hint: 'llama.cpp and other runtimes', value: paths.engine_dir },
					{ label: 'Chat transcripts', hint: undefined, value: paths.transcripts_dir },
					{ label: 'Database', hint: undefined, value: paths.db_path },
				]
			: [],
	);
</script>

<div class="wb-set-h">
	<h2>Storage</h2>
	<p>Where OpenHarness keeps models and data on this PC</p>
</div>

{#if library}
	<div class="wb-prop">
		<div class="pl">Model library</div>
		<div class="pv">
			<span class="num">{formatBytes(library.bytes)}</span>
			<span class="text-muted-foreground">
				across <span class="num">{library.count}</span> {library.count === 1 ? 'model' : 'models'}
			</span>
		</div>
	</div>
{/if}

{#each rows as r (r.label)}
	<div class="wb-prop">
		<div class="pl">
			{r.label}
			{#if r.hint}<small>{r.hint}</small>{/if}
		</div>
		<div class="pv"><span class="wb-code">{r.value}</span></div>
	</div>
{/each}

{#if error}
	<p class="px-5 pt-3 text-xs text-error" role="alert">Couldn't read the app folders: {error}</p>
{/if}
