<script lang="ts">
	interface Tab {
		id: string;
		label: string;
	}

	interface Props {
		tabs: Tab[];
		/** The active tab id (bindable). */
		active?: string;
		class?: string;
		onchange?: (id: string) => void;
	}

	let {
		tabs,
		active = $bindable(tabs[0]?.id ?? ''),
		class: extraClass = '',
		onchange,
	}: Props = $props();

	function select(id: string) {
		active = id;
		onchange?.(id);
	}
</script>

<div class="flex gap-1 border-b border-border {extraClass}" role="tablist">
	{#each tabs as tab}
		<button
			type="button"
			role="tab"
			aria-selected={active === tab.id}
			onclick={() => select(tab.id)}
			class="-mb-px whitespace-nowrap border-b-2 px-4 py-2 text-sm font-medium transition-base
			       {active === tab.id
				? 'border-primary text-primary'
				: 'border-transparent text-muted-foreground hover:text-foreground'}"
		>
			{tab.label}
		</button>
	{/each}
</div>
