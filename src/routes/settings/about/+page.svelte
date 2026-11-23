<script lang="ts">
	/** Settings, About: version, data folder, privacy stance. */
	import { onMount } from 'svelte';
	import { getAppPaths } from '$lib/ipc/commands';
	import pkg from '../../../../package.json';
	import type { AppPaths } from '$lib/types';

	let paths = $state<AppPaths | null>(null);

	onMount(async () => {
		try {
			paths = await getAppPaths();
		} catch {
			paths = null;
		}
	});
</script>

<div class="wb-set-h">
	<h2>About</h2>
</div>

<div class="wb-prop">
	<div class="pl">OpenHarness</div>
	<div class="pv"><span class="num">{pkg.version}</span></div>
</div>

{#if paths}
	<div class="wb-prop">
		<div class="pl">Data folder</div>
		<div class="pv"><span class="wb-code">{paths.app_data_dir}</span></div>
	</div>
{/if}

<div class="wb-prop">
	<div class="pl">Privacy</div>
	<div class="pv text-muted-foreground">
		No telemetry and no crash reports. OpenHarness goes online only to browse and download from
		Hugging Face and to fetch engines.
	</div>
</div>
