<script lang="ts">
	/**
	 * VideoTask (M16 W4) — simple by default: model + prompt + plain-language
	 * Length/Quality presets + Generate. Resolution, frames (8n+1), fps, steps,
	 * guidance and seed live in the Advanced disclosure. The presets are
	 * family-adaptive: LTX runs 24 fps at 704×480 native, Wan 16 fps at
	 * 832×480 — picking a model applies the right frame/fps rules.
	 */
	import Icon from '$lib/components/shell/Icon.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import ModelPicker from './ModelPicker.svelte';
	import TaskShell from './TaskShell.svelte';
	import AdvancedSection from './AdvancedSection.svelte';
	import OptionPills from './OptionPills.svelte';
	import RemixNote from './RemixNote.svelte';
	import { videoSeconds } from './studioMeta';
	import { remixModelOf, videoRemix } from './remix';
	import { fileSrc } from '$lib/ipc/backend';
	import { studio, submitJob } from '$lib/stores/studio.svelte';
	import type { Asset, InstalledModel } from '$lib/types';

	const TASK = 'text_to_video' as const;
	const VIDEO_FILTER = (m: InstalledModel) => m.caps.workflows.includes('video');

	let model = $state('');
	let prompt = $state('');
	let negative = $state('');
	let resolution = $state('768x512');
	let numFrames = $state(97); // must be 8n+1
	let fps = $state(24);
	let steps = $state(30);
	let guidance = $state(4); // LTX-2 default (Wan applies its own on family switch)
	let seed = $state('');

	const active = $derived(studio.active[TASK]);
	const canSubmit = $derived(!!model && prompt.trim().length > 0 && !active);

	// All dimensions are multiples of 32 (pipeline requirement).
	const RESOLUTIONS = [
		{ value: '512x512', label: '512 × 512 (square)' },
		{ value: '704x480', label: '704 × 480 (LTX native)' },
		{ value: '768x512', label: '768 × 512 (3:2)' },
		{ value: '832x480', label: '832 × 480 (16:9)' },
		{ value: '1024x576', label: '1024 × 576 (16:9 large)' },
	];
	// Remixed/older params may carry a size outside the preset list.
	const resolutionOptions = $derived(
		RESOLUTIONS.some((r) => r.value === resolution)
			? RESOLUTIONS
			: [...RESOLUTIONS, { value: resolution, label: resolution.replace('x', ' × ') }],
	);

	const seconds = $derived(videoSeconds(numFrames, fps));

	// ---- Family-adaptive presets ----------------------------------------------
	type Family = 'ltx' | 'wan';
	const family = $derived<Family>(model.toLowerCase().includes('wan') ? 'wan' : 'ltx');
	/** Native fps + resolution + default guidance per model family. */
	const FAMILY_DEFAULTS: Record<Family, { fps: number; resolution: string; guidance: number }> = {
		ltx: { fps: 24, resolution: '704x480', guidance: 4 },
		wan: { fps: 16, resolution: '832x480', guidance: 5 },
	};

	type Length = 'short' | 'medium' | 'long';
	const LENGTH_SECONDS: Record<Length, number> = { short: 2, medium: 4, long: 8 };
	type Quality = 'fast' | 'balanced' | 'best';
	const QUALITY_STEPS: Record<Quality, number> = { fast: 20, balanced: 30, best: 50 };

	// null = the numbers were edited by hand (custom) — no pill lights up.
	let length = $state<Length | null>('medium');
	let quality = $state<Quality | null>('balanced');

	/** Nearest 8n+1 frame count for the target seconds at the current fps. */
	function framesFor(secs: number): number {
		const n = Math.round((fps * secs) / 8) * 8 + 1;
		return Math.min(257, Math.max(9, n));
	}

	function applyLength(l: Length) {
		length = l;
		numFrames = framesFor(LENGTH_SECONDS[l]);
	}

	function applyQuality(q: Quality) {
		quality = q;
		steps = QUALITY_STEPS[q];
	}

	// Switching model families applies the family's native fps/resolution and
	// re-resolves the active length preset against the new frame rate.
	let lastFamily = '';
	$effect(() => {
		const fam = family;
		if (fam === lastFamily) return;
		lastFamily = fam;
		fps = FAMILY_DEFAULTS[fam].fps;
		resolution = FAMILY_DEFAULTS[fam].resolution;
		guidance = FAMILY_DEFAULTS[fam].guidance;
		if (length) applyLength(length);
	});

	async function generate() {
		const [w, h] = resolution.split('x').map(Number);
		const params: Record<string, unknown> = {
			model,
			prompt: prompt.trim(),
			width: w,
			height: h,
			num_frames: numFrames,
			fps,
			steps,
			guidance,
		};
		if (negative.trim()) params.negative_prompt = negative.trim();
		if (seed.trim()) params.seed = Number(seed.trim());
		await submitJob(TASK, params);
	}

	/** Restore a recent generation's prompt + params into the form. */
	let remixNote = $state<string | null>(null);
	let remixToken = 0;
	let picker = $state<ModelPicker>();

	/**
	 * Restore a recent generation into the form. Resolution, fps, frames,
	 * steps and guidance follow the model family, so they come back only when
	 * its model is still installed (see remix.ts).
	 */
	async function remix(a: Asset) {
		const token = ++remixToken;
		const installed = (await picker?.refresh()) ?? [];
		if (token !== remixToken) return; // a newer remix won
		const r = videoRemix(a, installed.map(remixModelOf));
		if (r.prompt !== undefined) prompt = r.prompt;
		negative = r.negative;
		seed = r.seed;
		remixNote = r.note;
		// Source model gone: the current model and its settings stay as they are.
		if (r.model === null) return;
		model = r.model;
		// Pre-acknowledge the family so the defaults effect doesn't clobber
		// the remixed fps/resolution below.
		lastFamily = r.model.toLowerCase().includes('wan') ? 'wan' : 'ltx';
		if (r.resolution !== undefined) resolution = r.resolution;
		if (r.fps !== undefined) fps = r.fps;
		if (r.numFrames !== undefined) numFrames = r.numFrames;
		if (r.steps !== undefined) steps = r.steps;
		if (r.guidance !== undefined) guidance = r.guidance;
		length = null;
		quality = null;
	}
</script>

<TaskShell task={TASK} onremix={remix}>
	{#snippet controls()}
		<RemixNote note={remixNote} ondismiss={() => (remixNote = null)} />
		<ModelPicker
			bind:this={picker}
			hfTask="text-to-video"
			capsFilter={VIDEO_FILTER}
			bind:value={model}
		/>

		<div class="st-field">
			<label for="vid-prompt" class="st-label">Prompt</label>
			<textarea
				id="vid-prompt"
				class="wb-input"
				bind:value={prompt}
				rows="5"
				placeholder="Aerial drone shot over a misty pine forest at sunrise"
			></textarea>
		</div>

		<OptionPills
			label="Length"
			value={length}
			hint={`${seconds} s`}
			options={[
				{ value: 'short', label: 'About 2 s' },
				{ value: 'medium', label: 'About 4 s' },
				{ value: 'long', label: 'About 8 s' },
			]}
			onselect={(v) => applyLength(v as Length)}
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

		<p class="wb-help">Video is heavy. Expect a few minutes per clip, even on a big GPU.</p>

		<AdvancedSection storageKey="video">
			<div class="st-field">
				<label for="vid-negative" class="st-label">Negative prompt</label>
				<textarea
					id="vid-negative"
					class="wb-input"
					bind:value={negative}
					rows="2"
					placeholder="Jittery, distorted, watermark"
				></textarea>
			</div>

			<div class="st-field">
				<label for="vid-res" class="st-label">Resolution</label>
				<select id="vid-res" class="wb-input num" bind:value={resolution}>
					{#each resolutionOptions as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
				</select>
			</div>

			<div class="st-field">
				<Slider
					bind:value={numFrames}
					min={9}
					max={257}
					step={8}
					label="Frames"
					onchange={() => (length = null)}
				/>
				<p class="wb-help text-right">About <span class="num">{seconds}</span> s at <span class="num">{fps}</span> fps</p>
			</div>

			<Slider
				bind:value={fps}
				min={8}
				max={30}
				step={1}
				label="Frames per second"
				onchange={() => {
					if (length) applyLength(length);
				}}
			/>
			<Slider bind:value={steps} min={1} max={100} step={1} label="Steps" onchange={() => (quality = null)} />
			<Slider bind:value={guidance} min={0} max={15} step={0.5} label="Guidance" />

			<div class="st-field">
				<label for="vid-seed" class="st-label">Seed</label>
				<div class="flex gap-2">
					<input id="vid-seed" type="number" class="wb-input num" bind:value={seed} placeholder="Random" />
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
			<Icon name="film" size={15} />
			{active ? 'Generating…' : 'Generate'}
		</button>
	{/snippet}

	{#snippet resultView(result)}
		<div class="flex w-full max-w-3xl flex-col items-center gap-3">
			{#each result.outputs as out (out)}
				<!-- svelte-ignore a11y_media_has_caption -->
				<video controls loop src={fileSrc(out)} class="st-out max-h-[58vh] w-full"></video>
			{/each}
			<p class="st-cap">
				<span class="truncate">{result.params['prompt']}</span>
				{#if typeof result.params['num_frames'] === 'number' && typeof result.params['fps'] === 'number'}
					<span class="flex-none"><span class="num">{result.params['num_frames']}</span> frames, <span class="num">{result.params['fps']}</span> fps</span>
				{/if}
				<span class="flex-none">Saved to the library</span>
			</p>
		</div>
	{/snippet}

	{#snippet empty()}
		<span class="st-idle-ic"><Icon name="film" size={24} /></span>
		<b class="st-idle-t">No clip yet</b>
		<p class="st-idle-p">Pick a video model, describe the shot and press Generate.</p>
	{/snippet}
</TaskShell>
