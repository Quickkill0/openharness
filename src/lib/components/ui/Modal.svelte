<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		open?: boolean;
		title?: string;
		/** SVG path string for an optional header icon. */
		icon?: string;
		size?: 'sm' | 'md' | 'lg' | 'xl';
		/** Hide the default close (X) button + disable backdrop close. */
		dismissible?: boolean;
		onclose?: () => void;
		children?: Snippet;
		/** Optional footer actions (rendered in the footer bar). */
		footer?: Snippet;
	}

	let {
		open = $bindable(false),
		title = '',
		icon,
		size = 'md',
		dismissible = true,
		onclose,
		children,
		footer,
	}: Props = $props();

	const sizeClasses: Record<string, string> = {
		sm: 'max-w-md',
		md: 'max-w-lg',
		lg: 'max-w-2xl',
		xl: 'max-w-4xl',
	};

	function close() {
		if (!dismissible) return;
		open = false;
		onclose?.();
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape' && open) close();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center p-4"
		role="dialog"
		aria-modal="true"
		aria-label={title}
	>
		<!-- Backdrop -->
		<button
			class="absolute inset-0 cursor-default bg-black/50 backdrop-blur-sm"
			onclick={close}
			aria-label="Close"
			tabindex="-1"
		></button>

		<!-- Panel -->
		<div
			class="oh-modal neo-surface relative flex max-h-[88vh] w-full {sizeClasses[
				size
			]} flex-col overflow-hidden rounded-[18px] shadow-l"
		>
			{#if title || dismissible}
				<header class="flex shrink-0 items-center justify-between gap-3 border-b border-border px-6 py-4">
					<div class="flex items-center gap-3">
						{#if icon}
							<div class="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/15">
								<svg class="h-5 w-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d={icon} />
								</svg>
							</div>
						{/if}
						<h2 class="text-lg font-semibold text-foreground">{title}</h2>
					</div>
					{#if dismissible}
						<button
							type="button"
							onclick={close}
							class="rounded-lg p-1.5 text-muted-foreground transition-base hover:bg-elevated hover:text-foreground"
							aria-label="Close"
						>
							<svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18 18 6M6 6l12 12" />
							</svg>
						</button>
					{/if}
				</header>
			{/if}

			<div class="flex-1 overflow-y-auto px-6 py-5">
				{@render children?.()}
			</div>

			{#if footer}
				<footer class="flex shrink-0 items-center justify-end gap-3 border-t border-border px-6 py-4">
					{@render footer()}
				</footer>
			{/if}
		</div>
	</div>
{/if}

<style>
	.oh-modal {
		animation: oh-modal-in 200ms cubic-bezier(0.16, 1, 0.3, 1);
	}

	@keyframes oh-modal-in {
		from {
			opacity: 0;
			transform: scale(0.96) translateY(8px);
		}
		to {
			opacity: 1;
			transform: scale(1) translateY(0);
		}
	}
</style>
