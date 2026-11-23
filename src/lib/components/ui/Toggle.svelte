<script lang="ts">
	interface Props {
		checked?: boolean;
		disabled?: boolean;
		label?: string;
		onchange?: (checked: boolean) => void;
	}

	let {
		checked = $bindable(false),
		disabled = false,
		label = '',
		onchange,
	}: Props = $props();

	function handleChange() {
		checked = !checked;
		onchange?.(checked);
	}
</script>

<button
	type="button"
	role="switch"
	aria-checked={checked}
	aria-label={label}
	{disabled}
	class="relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent
	       transition-base focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring
	       focus-visible:ring-offset-2 focus-visible:ring-offset-background
	       disabled:opacity-50 disabled:pointer-events-none
	       {checked ? 'bg-primary shadow-glow' : 'bg-muted'}"
	onclick={handleChange}
>
	<span
		class="pointer-events-none block h-5 w-5 rounded-full bg-white shadow-md transition-transform
		       {checked ? 'translate-x-5' : 'translate-x-0'}"
	></span>
</button>
