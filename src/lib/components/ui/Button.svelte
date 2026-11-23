<script lang="ts">
	interface Props {
		variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
		size?: 'sm' | 'md' | 'lg';
		disabled?: boolean;
		type?: 'button' | 'submit' | 'reset';
		class?: string;
		onclick?: (e: MouseEvent) => void;
	}

	let {
		variant = 'primary',
		size = 'md',
		disabled = false,
		type = 'button',
		class: extraClass = '',
		onclick,
		children,
	}: Props & { children?: import('svelte').Snippet } = $props();

	const base =
		'inline-flex items-center justify-center font-medium rounded-[10px] select-none ' +
		'transition-base active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 ' +
		'focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background ' +
		'disabled:opacity-50 disabled:pointer-events-none';

	const variants: Record<string, string> = {
		primary: 'btn-primary text-primary-foreground shadow-s',
		secondary:
			'bg-elevated text-foreground border border-border hover:bg-muted',
		danger: 'btn-danger text-white shadow-s',
		ghost: 'bg-transparent text-muted-foreground hover:bg-elevated hover:text-foreground',
	};

	const sizes: Record<string, string> = {
		sm: 'h-7 px-3 text-xs gap-1.5',
		md: 'h-9 px-4 text-sm gap-2',
		lg: 'h-11 px-6 text-base gap-2.5',
	};

	const cls = $derived([base, variants[variant], sizes[size], extraClass].join(' '));
</script>

<button {type} {disabled} class={cls} {onclick}>
	{#if children}
		{@render children()}
	{/if}
</button>

<style>
	.btn-primary {
		background: var(--button-primary-gradient);
	}
	.btn-primary:hover:not(:disabled) {
		box-shadow: var(--shadow-m);
		filter: brightness(1.08);
	}
	.btn-danger {
		background: linear-gradient(135deg, var(--error), oklch(from var(--error) calc(l - 0.05) c h));
	}
	.btn-danger:hover:not(:disabled) {
		box-shadow: var(--shadow-m);
		filter: brightness(1.08);
	}
</style>
