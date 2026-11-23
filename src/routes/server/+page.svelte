<script lang="ts">
	/**
	 * Server: the OpenAI-compatible API server (PR #59) inside the Workbench
	 * shell. The settings panel is mounted as shipped; its Workbench restyle
	 * is phase C. The rail entry and the dock's server pill point here.
	 */
	import PageHeader from '$lib/components/shell/PageHeader.svelte';
	import ServerSettings from '$lib/components/ServerSettings.svelte';
</script>

<div class="flex h-full flex-col">
	<PageHeader title="Server" meta="OpenAI-compatible API for AI Shuffle and other clients" />
	<div class="min-h-0 flex-1 overflow-y-auto">
		<div class="mx-auto max-w-2xl p-6">
			<ServerSettings />
		</div>
	</div>
</div>
