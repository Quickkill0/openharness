<script lang="ts">
	interface Option {
		value: string;
		label: string;
		/** Optional optgroup label — consecutive options sharing it render grouped. */
		group?: string;
		/** Shown but not choosable (say why in the label). */
		disabled?: boolean;
	}

	interface Props {
		value?: string;
		options: Option[];
		placeholder?: string;
		disabled?: boolean;
		id?: string;
		name?: string;
		class?: string;
		onchange?: (value: string) => void;
	}

	let {
		value = $bindable(''),
		options,
		placeholder = '',
		disabled = false,
		id,
		name,
		class: extraClass = '',
		onchange,
	}: Props = $props();

	function handleChange(e: Event & { currentTarget: HTMLSelectElement }) {
		value = e.currentTarget.value;
		onchange?.(value);
	}

	/** Consecutive options with the same `group` label collapse into an <optgroup>. */
	interface Segment {
		group: string | null;
		options: Option[];
	}
	const segments = $derived(
		options.reduce<Segment[]>((acc, opt) => {
			const group = opt.group ?? null;
			const last = acc[acc.length - 1];
			if (last && last.group === group) last.options.push(opt);
			else acc.push({ group, options: [opt] });
			return acc;
		}, []),
	);
</script>

<select
	{id}
	{name}
	{disabled}
	bind:value
	onchange={handleChange}
	class="oh-select w-full appearance-none rounded-[10px] border border-border bg-elevated px-3 py-2 pr-9
	       text-sm text-foreground transition-base hover:border-ring/40
	       focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent
	       disabled:opacity-50 disabled:cursor-not-allowed {extraClass}"
>
	{#if placeholder}
		<option value="" disabled selected={!value}>{placeholder}</option>
	{/if}
	{#each segments as seg, i (i)}
		{#if seg.group}
			<optgroup label={seg.group}>
				{#each seg.options as opt (opt.value)}
					<option value={opt.value} disabled={opt.disabled}>{opt.label}</option>
				{/each}
			</optgroup>
		{:else}
			{#each seg.options as opt (opt.value)}
				<option value={opt.value} disabled={opt.disabled}>{opt.label}</option>
			{/each}
		{/if}
	{/each}
</select>

<style>
	.oh-select {
		background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%239ca3af' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e");
		background-position: right 0.6rem center;
		background-repeat: no-repeat;
		background-size: 1.1em 1.1em;
	}

	.oh-select option {
		background-color: var(--surface);
		color: var(--foreground);
	}
</style>
