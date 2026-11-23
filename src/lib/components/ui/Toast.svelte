<script lang="ts">
	import { toasts, removeToast } from '$lib/stores/toasts.svelte';
	import { uiStore } from '$lib/stores/ui.svelte';

	const variantVar: Record<string, string> = {
		info: '--info',
		success: '--success',
		warning: '--warning',
		error: '--error',
	};

	const iconPaths: Record<string, string> = {
		info: 'M11.25 11.25l.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z',
		success:
			'M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
		warning:
			'M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z',
		error:
			'M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z',
	};
</script>

<!-- Toast container: fixed bottom-right above the 60px dock, stacked. Steps left
     of the 440px downloads panel while it is open so the two never overlap. -->
<div
	class="pointer-events-none fixed bottom-[76px] z-50 flex flex-col gap-2 {uiStore.downloadsOpen ? 'right-[464px]' : 'right-4'}"
	aria-live="polite"
>
	{#each toasts.value as toast (toast.id)}
		<div
			class="toast glass pointer-events-auto flex w-80 max-w-xs items-start gap-3 rounded-[14px] p-4 shadow-l transition-all"
			style="--accent: var({variantVar[toast.variant]})"
		>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				fill="none"
				viewBox="0 0 24 24"
				stroke-width="1.5"
				stroke="currentColor"
				class="toast-icon mt-0.5 h-5 w-5 shrink-0"
			>
				<path stroke-linecap="round" stroke-linejoin="round" d={iconPaths[toast.variant]} />
			</svg>
			<p class="flex-1 text-sm leading-snug text-foreground">{toast.message}</p>
			<button
				type="button"
				aria-label="Dismiss notification"
				class="shrink-0 text-muted-foreground hover:text-foreground"
				onclick={() => removeToast(toast.id)}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="h-4 w-4"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
				</svg>
			</button>
		</div>
	{/each}
</div>

<style>
	.toast {
		border-left: 3px solid var(--accent);
	}
	.toast-icon {
		color: var(--accent);
	}
</style>
