<script lang="ts">
	interface Props {
		size?: 'sm' | 'md' | 'lg';
		class?: string;
	}

	let { size = 'md', class: extraClass = '' }: Props = $props();

	const sizes: Record<string, string> = {
		sm: 'h-4 w-4 border-2',
		md: 'h-6 w-6 border-2',
		lg: 'h-8 w-8 border-[3px]',
	};
</script>

<span
	role="status"
	aria-label="Loading"
	class="inline-block animate-spin rounded-full border-border border-t-primary {sizes[size]} {extraClass}"
></span>
