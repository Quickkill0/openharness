<script lang="ts">
	interface Props {
		type?: string;
		value?: string;
		placeholder?: string;
		disabled?: boolean;
		readonly?: boolean;
		id?: string;
		name?: string;
		autocomplete?: HTMLInputElement['autocomplete'];
		class?: string;
		oninput?: (e: Event & { currentTarget: HTMLInputElement }) => void;
		onchange?: (e: Event & { currentTarget: HTMLInputElement }) => void;
		onkeydown?: (e: KeyboardEvent & { currentTarget: HTMLInputElement }) => void;
		oncompositionstart?: (e: CompositionEvent & { currentTarget: HTMLInputElement }) => void;
		oncompositionend?: (e: CompositionEvent & { currentTarget: HTMLInputElement }) => void;
	}

	let {
		type = 'text',
		value = $bindable(''),
		placeholder = '',
		disabled = false,
		readonly = false,
		id,
		name,
		autocomplete,
		class: extraClass = '',
		oninput,
		onchange,
		onkeydown,
		oncompositionstart,
		oncompositionend,
	}: Props = $props();

	const base =
		'w-full rounded-[10px] border border-border bg-elevated px-3 py-2 text-sm text-foreground transition-base ' +
		'placeholder:text-muted-foreground hover:border-ring/40 ' +
		'focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent ' +
		'disabled:opacity-50 disabled:cursor-not-allowed read-only:opacity-70';
</script>

<input
	{type}
	bind:value
	{placeholder}
	{disabled}
	{readonly}
	{id}
	{name}
	{autocomplete}
	class="{base} {extraClass}"
	{oninput}
	{onchange}
	{onkeydown}
	{oncompositionstart}
	{oncompositionend}
/>
