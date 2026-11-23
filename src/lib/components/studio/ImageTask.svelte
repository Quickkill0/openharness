<script lang="ts">
	/**
	 * ImageTask (M16 W4) — beginner-first: model + prompt + Generate, with
	 * plain-language Size/Quality presets. Every numeric control lives in the
	 * Advanced disclosure. The preset mappings are capability-adaptive: they
	 * follow the selected model's `caps.image_arch` (SD 1.5 sizes are 512-class,
	 * SDXL/FLUX 1024-class; turbo models pin guidance to 1.0 and run few steps).
	 */
	import Icon from '$lib/components/shell/Icon.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import ModelPicker from './ModelPicker.svelte';
	import TaskShell from './TaskShell.svelte';
	import AdvancedSection from './AdvancedSection.svelte';
	import OptionPills from './OptionPills.svelte';
	import RemixNote from './RemixNote.svelte';
	import { imageRemix, remixModelOf } from './remix';
	import { fileSrc } from '$lib/ipc/backend';
	import { studio, submitJob } from '$lib/stores/studio.svelte';
	import type { Asset, InstalledModel } from '$lib/types';

	const TASK = 'text_to_image' as const;
	const IMAGE_FILTER = (m: InstalledModel) => m.caps.workflows.includes('image');

	let model = $state('');
	let selected = $state<InstalledModel | null>(null);

	let prompt = $state('');
	let negative = $state('');
	let arch = $state(''); // '' = auto-detect from the model name
	let width = $state(1024);
	let height = $state(1024);
	let steps = $state(30);
	let guidance = $state(7);
	let numImages = $state(1);
	let seed = $state('');

	const active = $derived(studio.active[TASK]);
	const canSubmit = $derived(!!model && prompt.trim().length > 0 && !active);

	const SIZES = [512, 640, 768, 832, 896, 1024, 1152, 1216, 1344];
	const sizeOptions = SIZES.map((n) => ({ value: String(n), label: `${n} px` }));
	const archOptions = [
		{ value: '', label: 'Auto-detect' },
		{ value: 'sdxl', label: 'SDXL' },
		{ value: 'sd', label: 'SD 1.5 / 2.x' },
		{ value: 'flux', label: 'FLUX' },
	];

	// Filename fallback for models the capability resolver doesn't know.
	function detectArch(path: string): string {
		const name = path.toLowerCase();
		if (name.includes('flux')) return 'flux';
		if (name.includes('xl') || name.includes('turbo')) return 'sdxl';
		return 'sd';
	}

	// ---- Capability-adaptive presets -----------------------------------------
	const family = $derived(arch || selected?.caps.image_arch || detectArch(model));
	/** Few-step distilled models: hide guidance (cfg must stay ~1.0). */
	const isTurbo = $derived(/turbo|lightning|lcm/.test(model.toLowerCase()));

	type Quality = 'fast' | 'balanced' | 'best';
	type SizeKey = 'square' | 'portrait' | 'wide';

	/** steps + guidance per quality level, by architecture family. */
	const qualityMap = $derived.by((): Record<Quality, { steps: number; guidance: number }> => {
		if (isTurbo) {
			return { fast: { steps: 2, guidance: 1 }, balanced: { steps: 4, guidance: 1 }, best: { steps: 8, guidance: 1 } };
		}
		if (family === 'flux') {
			return { fast: { steps: 12, guidance: 3 }, balanced: { steps: 28, guidance: 3.5 }, best: { steps: 45, guidance: 4 } };
		}
		// sd / sdxl
		return { fast: { steps: 15, guidance: 5 }, balanced: { steps: 30, guidance: 7 }, best: { steps: 50, guidance: 8 } };
	});

	/** Native-resolution classes: SD 1.5 trained at 512, SDXL/FLUX at 1024. */
	const sizeMap = $derived.by((): Record<SizeKey, { w: number; h: number }> => {
		if (family === 'sd') {
			return { square: { w: 512, h: 512 }, portrait: { w: 512, h: 768 }, wide: { w: 768, h: 512 } };
		}
		return { square: { w: 1024, h: 1024 }, portrait: { w: 832, h: 1216 }, wide: { w: 1216, h: 832 } };
	});

	// null = the numbers were edited by hand (custom) — no pill lights up.
	let quality = $state<Quality | null>('balanced');
	let sizePreset = $state<SizeKey | null>('square');

	function applyQuality(q: Quality) {
		quality = q;
		steps = qualityMap[q].steps;
		guidance = qualityMap[q].guidance;
	}

	function applySize(s: SizeKey) {
		sizePreset = s;
		width = sizeMap[s].w;
		height = sizeMap[s].h;
	}

	// Switching to a different model family re-resolves the active presets, so
	// e.g. picking sdxl-turbo lands on steps 4 / guidance 1.0 automatically.
	let lastFamilyKey = '';
	$effect(() => {
		const key = `${family}:${isTurbo}`;
		if (key === lastFamilyKey) return;
		lastFamilyKey = key;
		if (quality) applyQuality(quality);
		if (sizePreset) applySize(sizePreset);
	});

	async function generate() {
		const params: Record<string, unknown> = {
			model,
			prompt: prompt.trim(),
			arch: arch || selected?.caps.image_arch || detectArch(model),
			width,
			height,
			steps,
			// Turbo-class models only work at cfg 1.0 — the slider is hidden.
			guidance: isTurbo ? 1.0 : guidance,
			num_images: numImages,
		};
		if (negative.trim()) params.negative_prompt = negative.trim();
		if (seed.trim()) params.seed = Number(seed.trim());
		await submitJob(TASK, params);
	}

	let remixNote = $state<string | null>(null);
	let remixToken = 0;
	let picker = $state<ModelPicker>();

	/**
	 * Restore a recent generation into the form. Architecture, size, steps and
	 * guidance come back only when its model is in the picker's freshly read
	 * list; otherwise the current model and every setting that depends on it
	 * stay exactly as they are (see remix.ts).
	 */
	async function remix(a: Asset) {
		const token = ++remixToken;
		const installed = (await picker?.refresh()) ?? [];
		if (token !== remixToken) return; // a newer remix won
		const r = imageRemix(a, installed.map(remixModelOf));
		if (r.prompt !== undefined) prompt = r.prompt;
		negative = r.negative;
		seed = r.seed;
		if (r.numImages !== undefined) numImages = r.numImages;
		remixNote = r.note;
		if (r.model === null) return;
		// Presets go to custom, so the family effect leaves the numbers alone.
		quality = null;
		sizePreset = null;
		model = r.model;
		if (r.arch !== undefined) arch = r.arch;
		if (r.width !== undefined) width = r.width;
		if (r.height !== undefined) height = r.height;
		if (r.steps !== undefined) steps = r.steps;
		if (r.guidance !== undefined) guidance = r.guidance;
	}
</script>

<TaskShell task={TASK} onremix={remix}>
	{#snippet controls()}
		<RemixNote note={remixNote} ondismiss={() => (remixNote = null)} />
		<ModelPicker
			bind:this={picker}
			hfTask="text-to-image"
			capsFilter={IMAGE_FILTER}
			bind:value={model}
			onmodel={(m) => (selected = m)}
		/>

		<div class="st-field">
			<label for="img-prompt" class="st-label">Prompt</label>
			<textarea
				id="img-prompt"
				class="wb-input"
				bind:value={prompt}
				rows="5"
				placeholder="A lighthouse at dawn, oil painting"
			></textarea>
		</div>

		<OptionPills
			label="Size"
			value={sizePreset}
			hint={`${width} × ${height}`}
			options={[
				{ value: 'square', label: 'Square' },
				{ value: 'portrait', label: 'Portrait' },
				{ value: 'wide', label: 'Widescreen' },
			]}
			onselect={(v) => applySize(v as SizeKey)}
		/>

		<OptionPills
			label="Quality"
			value={quality}
			options={[
				{ value: 'fast', label: 'Fast' },
				{ value: 'balanced', label: 'Balanced' },
				{ value: 'best', label: 'Best' },
			]}
			onselect={(v) => applyQuality(v as Quality)}
		/>

		<AdvancedSection storageKey="image">
			<div class="st-field">
				<label for="img-negative" class="st-label">Negative prompt</label>
				<textarea
					id="img-negative"
					class="wb-input"
					bind:value={negative}
					rows="2"
					placeholder="Blurry, low quality"
				></textarea>
			</div>

			<div class="st-grid2">
				<div class="st-field">
					<label for="img-w" class="st-label">Width</label>
					<select
						id="img-w"
						class="wb-input num"
						value={String(width)}
						onchange={(e) => {
							width = Number(e.currentTarget.value);
							sizePreset = null;
						}}
					>
						{#each sizeOptions as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
					</select>
				</div>
				<div class="st-field">
					<label for="img-h" class="st-label">Height</label>
					<select
						id="img-h"
						class="wb-input num"
						value={String(height)}
						onchange={(e) => {
							height = Number(e.currentTarget.value);
							sizePreset = null;
						}}
					>
						{#each sizeOptions as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
					</select>
				</div>
			</div>

			<div class="st-field">
				<label for="img-arch" class="st-label">Architecture</label>
				<select id="img-arch" class="wb-input" bind:value={arch}>
					{#each archOptions as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
				</select>
			</div>

			<Slider bind:value={steps} min={1} max={150} step={1} label="Steps" onchange={() => (quality = null)} />
			{#if isTurbo}
				<p class="wb-help">This is a turbo model, so guidance stays at 1.0. Higher values break few-step models.</p>
			{:else}
				<Slider bind:value={guidance} min={0} max={20} step={0.5} label="Guidance" onchange={() => (quality = null)} />
			{/if}
			<Slider bind:value={numImages} min={1} max={8} step={1} label="Images" />

			<div class="st-field">
				<label for="img-seed" class="st-label">Seed</label>
				<div class="flex gap-2">
					<input id="img-seed" type="number" class="wb-input num" bind:value={seed} placeholder="Random" />
					<button
						type="button"
						class="wb-icon-btn flex-none"
						onclick={() => (seed = String(Math.floor(Math.random() * 2 ** 31)))}
						aria-label="Pick a random seed"
						title="Pick a random seed"
					>
						<Icon name="dice" size={15} />
					</button>
				</div>
			</div>
		</AdvancedSection>
	{/snippet}

	{#snippet action()}
		<button type="button" class="wb-btn primary" disabled={!canSubmit} onclick={generate}>
			<Icon name="sparkles" size={15} />
			{active ? 'Generating…' : 'Generate'}
		</button>
	{/snippet}

	{#snippet resultView(result)}
		<div class="flex w-full max-w-3xl flex-col items-center gap-3">
			<div class="grid w-full gap-3 {result.outputs.length > 1 ? 'grid-cols-2' : 'grid-cols-1'}">
				{#each result.outputs as out (out)}
					<img
						src={fileSrc(out)}
						alt={String(result.params['prompt'] ?? 'Generated image')}
						class="st-out mx-auto max-h-[58vh] w-auto max-w-full"
					/>
				{/each}
			</div>
			<p class="st-cap">
				<span class="truncate">{result.params['prompt']}</span>
				{#if typeof result.params['width'] === 'number'}
					<span class="num flex-none">{result.params['width']} × {result.params['height']}</span>
				{/if}
				{#if typeof result.params['steps'] === 'number'}
					<span class="flex-none"><span class="num">{result.params['steps']}</span> steps</span>
				{/if}
				<span class="flex-none">Saved to the library</span>
			</p>
		</div>
	{/snippet}

	{#snippet empty()}
		<span class="st-idle-ic"><Icon name="image" size={24} /></span>
		<b class="st-idle-t">Nothing generated yet</b>
		<p class="st-idle-p">Pick a model, describe the image and press Generate.</p>
	{/snippet}
</TaskShell>
