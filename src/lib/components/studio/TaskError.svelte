<script lang="ts">
	import Icon from '$lib/components/shell/Icon.svelte';
	import { clearError } from '$lib/stores/studio.svelte';
	import type { GenerationTask } from '$lib/types';

	interface Props {
		task: GenerationTask;
		message: string;
	}

	let { task, message }: Props = $props();
</script>

<div class="wb-banner err task-err" role="alert">
	<span class="te-ic" aria-hidden="true"><Icon name="info" size={15} /></span>
	<span class="te-msg">{message}</span>
	<button type="button" class="wb-icon-btn" onclick={() => clearError(task)} aria-label="Dismiss">
		<Icon name="x" size={14} />
	</button>
</div>

<style>
	.task-err {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		flex: none;
	}
	.te-ic {
		flex: none;
		margin-top: 2px;
		color: var(--error);
	}
	.te-msg {
		flex: 1;
		min-width: 0;
		padding-top: 3px;
		overflow-wrap: anywhere;
	}
	.task-err .wb-icon-btn {
		flex: none;
		margin: -2px -4px -2px 0;
	}
</style>
