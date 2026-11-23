<script lang="ts">
	interface Props {
		value?: number;
		min?: number;
		max?: number;
		step?: number;
		label?: string;
		/** Optional unit/suffix shown next to the numeric value (e.g. "tok"). */
		unit?: string;
		disabled?: boolean;
		id?: string;
		class?: string;
		onchange?: (value: number) => void;
	}

	let {
		value = $bindable(0),
		min = 0,
		max = 100,
		step = 1,
		label = '',
		unit = '',
		disabled = false,
		id,
		class: extraClass = '',
		onchange,
	}: Props = $props();

	const pct = $derived(
		max > min ? ((Math.min(Math.max(value, min), max) - min) / (max - min)) * 100 : 0
	);

	let numEl: HTMLInputElement | undefined = $state();
	let editing = $state(false);

	// Reflect slider/external value changes into the number field — but never
	// while the user is typing in it, or we'd rewrite their text under them.
	$effect(() => {
		const v = value;
		if (numEl && !editing) numEl.value = String(v);
	});

	const numWidth = $derived(
		Math.max(String(max).length, String(min).length, String(value).length, 2) + 1
	);

	// Live-apply typed values that are already in range; anything else waits
	// for commit so typing "1" toward "1024" isn't clamped mid-keystroke.
	function handleNumInput(e: Event & { currentTarget: HTMLInputElement }) {
		const raw = e.currentTarget.value;
		const v = Number(raw);
		if (raw !== '' && !Number.isNaN(v) && v >= min && v <= max) {
			value = v;
			onchange?.(value);
		}
	}

	function commitNum() {
		editing = false;
		if (!numEl) return;
		const raw = numEl.value;
		let v = Number(raw);
		if (raw === '' || Number.isNaN(v)) v = value;
		v = Math.min(max, Math.max(min, v));
		numEl.value = String(v);
		if (v !== value) {
			value = v;
			onchange?.(value);
		}
	}

	function handleNumKeydown(e: KeyboardEvent & { currentTarget: HTMLInputElement }) {
		if (e.key === 'Enter') {
			e.currentTarget.blur();
		} else if (e.key === 'Escape') {
			e.currentTarget.value = String(value);
			e.currentTarget.blur();
		}
	}
</script>

<div class="flex flex-col gap-1.5 {extraClass}">
	{#if label}
		<div class="flex items-center justify-between text-xs">
			<label for={id} class="font-medium text-muted-foreground">{label}</label>
			<span class="flex items-baseline gap-1 font-mono text-foreground">
				<input
					bind:this={numEl}
					type="number"
					{min}
					{max}
					{step}
					{disabled}
					class="oh-slider-num"
					style="width: calc({numWidth}ch + 0.5rem + 2px)"
					aria-label={label ? `${label} value` : 'value'}
					onfocus={() => (editing = true)}
					oninput={handleNumInput}
					onblur={commitNum}
					onkeydown={handleNumKeydown}
				/>
				{#if unit}<span class="text-muted-foreground">{unit}</span>{/if}
			</span>
		</div>
	{/if}
	<!-- bind:value (not value={…}+oninput): a one-way value attr on a range
	     input is not written back reactively, so typed values in the number
	     field would never move the thumb. -->
	<input
		{id}
		type="range"
		{min}
		{max}
		{step}
		{disabled}
		bind:value
		class="oh-slider"
		style="--pct: {pct}%"
		oninput={() => onchange?.(value)}
	/>
</div>

<style>
	.oh-slider-num {
		appearance: textfield;
		-moz-appearance: textfield;
		background: transparent;
		border: 1px solid transparent;
		border-radius: 6px;
		padding: 0 0.25rem;
		min-width: 3ch;
		text-align: right;
		font: inherit;
		color: inherit;
	}

	.oh-slider-num::-webkit-outer-spin-button,
	.oh-slider-num::-webkit-inner-spin-button {
		-webkit-appearance: none;
		margin: 0;
	}

	.oh-slider-num:hover:not(:disabled) {
		border-color: var(--border);
	}

	.oh-slider-num:focus {
		outline: none;
		border-color: var(--ring);
		background: var(--elevated);
	}

	.oh-slider-num:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.oh-slider {
		-webkit-appearance: none;
		appearance: none;
		width: 100%;
		height: 6px;
		border-radius: 999px;
		background: linear-gradient(
			to right,
			var(--primary) 0%,
			var(--primary) var(--pct),
			var(--muted) var(--pct),
			var(--muted) 100%
		);
		outline: none;
		cursor: pointer;
	}

	.oh-slider:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.oh-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--foreground);
		border: 2px solid var(--primary);
		box-shadow: var(--shadow-s);
		transition: transform 150ms cubic-bezier(0.4, 0, 0.2, 1);
	}

	.oh-slider::-webkit-slider-thumb:hover {
		transform: scale(1.15);
	}

	.oh-slider::-moz-range-thumb {
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--foreground);
		border: 2px solid var(--primary);
		box-shadow: var(--shadow-s);
	}

	.oh-slider:focus-visible {
		box-shadow: 0 0 0 2px var(--ring);
	}
</style>
