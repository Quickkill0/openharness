<script lang="ts">
	/**
	 * Plain-language preset control — "Fast / Balanced / Best" instead of raw
	 * numbers, drawn as a Workbench segmented control. `value === null` means
	 * the underlying numbers were edited by hand (custom) and no segment is on.
	 */
	interface Option {
		value: string;
		label: string;
	}

	interface Props {
		label?: string;
		options: Option[];
		value: string | null;
		onselect: (value: string) => void;
		/** Muted text on the right of the label (e.g. the resolved size). */
		hint?: string | null;
	}

	let { label, options, value, onselect, hint = null }: Props = $props();
</script>

<div class="st-field">
	{#if label}
		<span class="st-label">
			{label}
			{#if hint}<span class="num">{hint}</span>{:else if value === null}<span class="custom">Custom</span>{/if}
		</span>
	{/if}
	<div class="wb-seg w-full" role="radiogroup" aria-label={label}>
		{#each options as o (o.value)}
			<button
				type="button"
				role="radio"
				class="flex-1"
				aria-checked={value === o.value}
				onclick={() => onselect(o.value)}
			>
				{o.label}
			</button>
		{/each}
	</div>
</div>

<style>
	.custom {
		font-size: var(--text-2xs);
		font-weight: 400;
		color: var(--muted-foreground);
	}
</style>
