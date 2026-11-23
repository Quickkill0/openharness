<script lang="ts">
	import Button from '$lib/components/ui/Button.svelte';
	import type { InstalledModel } from '$lib/types';

	interface Props {
		model: InstalledModel | null;
		onconfirm?: (id: string) => void;
		oncancel?: () => void;
	}

	let { model, onconfirm, oncancel }: Props = $props();

	function handleConfirm() {
		if (model) onconfirm?.(model.id);
	}
</script>

{#if model}
	<!-- Backdrop -->
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div
		role="presentation"
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
		onclick={oncancel}
		onkeydown={(e) => e.key === 'Escape' && oncancel?.()}
	>
		<!-- Dialog -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div
			class="relative z-10 w-full max-w-sm rounded-xl border border-border bg-surface p-6 shadow-2xl"
			onclick={(e) => e.stopPropagation()}
			role="dialog"
			tabindex="-1"
			aria-modal="true"
			aria-labelledby="delete-dialog-title"
		>
			<h2 id="delete-dialog-title" class="text-base font-semibold text-foreground">
				Delete model?
			</h2>
			<p class="mt-2 text-sm text-muted-foreground">
				This will remove <span class="font-medium text-foreground">{model.display_name}</span>
				from the catalog and delete its GGUF file from disk. This cannot be undone.
			</p>
			<div class="mt-5 flex items-center justify-end gap-3">
				<Button variant="ghost" onclick={oncancel}>Cancel</Button>
				<Button variant="danger" onclick={handleConfirm}>Delete</Button>
			</div>
		</div>
	</div>
{/if}
