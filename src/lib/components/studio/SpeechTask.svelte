<script lang="ts">
	/**
	 * SpeechTask — capability-driven TTS form (M15, redesigned M16 W4).
	 *
	 * One model picker covers every speech-capable model (built-in Kokoro,
	 * Kokoro repos, Qwen3-TTS talkers, VibeVoice cloners). The controls render
	 * from the selected model's capability record: voice list (preset engines),
	 * reference-audio inputs (cloning), and the Advanced disclosure carries the
	 * numeric/optional knobs — speed, style instructions, speaker count + extra
	 * reference samples, cfg guidance.
	 */
	import { onMount } from 'svelte';
	import Icon from '$lib/components/shell/Icon.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import Toggle from '$lib/components/ui/Toggle.svelte';
	import ModelPicker from './ModelPicker.svelte';
	import TaskShell from './TaskShell.svelte';
	import AdvancedSection from './AdvancedSection.svelte';
	import RemixNote from './RemixNote.svelte';
	import { remixModelOf, speechRemix } from './remix';
	import { fileSrc } from '$lib/ipc/backend';
	import { ttsCliInstalled, ttsInstalled, ttsStatus } from '$lib/ipc/commands';
	import { studio, submitJob, type TaskResult } from '$lib/stores/studio.svelte';
	import { setSpeechVoiceConfig, voiceStore, type SpeechEngine } from '$lib/stores/voice.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import type { Asset, InstalledModel, SpeechCaps } from '$lib/types';

	const TASK = 'text_to_speech' as const;
	const SPEECH_FILTER = (m: InstalledModel) => m.caps.speech !== null;

	// Fallback caps used only while no model is selected — keeps the derived
	// chips defined; the controls below are guarded on `selected` so the user
	// never sees voice/speed/clone UI until they pick an installed model.
	const NO_CAPS: SpeechCaps = {
		engine: 'kokoro',
		voices: [],
		supports_speed: false,
		supports_instructions: false,
		supports_voice_cloning: false,
		cloning_requires_ref_text: false,
	};

	let text = $state('');
	let modelValue = $state(''); // catalog id; '' = nothing selected
	let selected = $state<InstalledModel | null>(null);
	let voice = $state('af_heart');
	let speed = $state(1);
	let instructions = $state('');
	/** Reference samples, one per speaker (VibeVoice cloning). */
	let referenceAudios = $state<string[]>(['']);
	/** Transcript of the reference sample (Qwen3-TTS `--ref-text` cloning). */
	let refText = $state('');
	/** "Clone a voice" toggle — only shown when a model offers presets AND cloning. */
	let cloneVoice = $state(false);
	let numSpeakers = $state(1);
	let cfgScale = $state(1.3);
	let qwenEngineInstalled = $state(true);
	let qwenCliInstalled = $state(true);
	let liveQwenVoices = $state<string[]>([]);

	// Everything below the picker renders from this descriptor. No model
	// selected → NO_CAPS (empty), and the controls stay hidden until a real
	// installed model is chosen — there is no built-in fallback.
	const caps = $derived<SpeechCaps>(selected?.caps.speech ?? NO_CAPS);
	const engine = $derived(caps.engine as SpeechEngine);
	// Voice cloning: reference sample(s) instead of (or alongside) presets.
	const cloning = $derived(caps.supports_voice_cloning);
	// Pure-cloning models (no presets) always clone; models that also ship
	// preset voices clone only when the toggle is on.
	const cloneActive = $derived(cloning && (caps.voices.length === 0 || cloneVoice));
	// Qwen clones run through the one-shot qwen-tts CLI and need the
	// reference transcript; VibeVoice keeps its multi-speaker worker layout.
	const qwenClone = $derived(cloneActive && engine === 'qwen3-tts');
	const vibeClone = $derived(cloneActive && caps.engine === 'vibevoice');
	const needsRefText = $derived(cloneActive && caps.cloning_requires_ref_text);

	const active = $derived(studio.active[TASK]);
	const canSubmit = $derived(
		text.trim().length > 0 &&
			selected !== null &&
			!active &&
			(!cloneActive ||
				(referenceAudios.slice(0, numSpeakers).every((r) => r.trim().length > 0) &&
					(!needsRefText || refText.trim().length > 0))),
	);

	function prettyVoice(v: string): string {
		return v.split('_').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
	}

	// Voice options come from the capability record; the qwentts sidecar's live
	// /v1/voices list replaces the static fallback once the engine is Ready.
	const voiceOptions = $derived(
		engine === 'qwen3-tts' && liveQwenVoices.length > 0
			? liveQwenVoices.map((v) => ({ value: v, label: prettyVoice(v) }))
			: caps.voices.map((v) => ({ value: v.id, label: v.label })),
	);

	// Model switches reset the voice when the old pick isn't in the new list.
	$effect(() => {
		if (voiceOptions.length > 0 && !voiceOptions.some((o) => o.value === voice)) {
			voice = voiceOptions[0].value;
		}
	});

	/** Keep one reference input per speaker, preserving already-typed paths. */
	function setNumSpeakers(n: number) {
		numSpeakers = n;
		const next = referenceAudios.slice(0, n);
		while (next.length < n) next.push('');
		referenceAudios = next;
	}

	/** Worker pipelines load Kokoro repos from disk path; qwentts wants the catalog id. */
	function pathOf(m: InstalledModel): string {
		return m.local_path ?? m.primary_file_path;
	}

	onMount(async () => {
		try {
			qwenEngineInstalled = await ttsInstalled();
			qwenCliInstalled = await ttsCliInstalled();
			const status = await ttsStatus();
			if (status.phase === 'ready') liveQwenVoices = status.voices;
		} catch {
			// Engine probe is advisory — synthesis errors surface in TaskError.
		}
	});

	async function generate() {
		if (!selected) return; // submit is gated on a selected installed model
		const params: Record<string, unknown> = { text: text.trim() };
		if (qwenClone && selected) {
			// One-shot qwen-tts CLI clone: talker by catalog id, single
			// reference sample + its transcript.
			params.model = selected.id;
			params.engine = 'qwen3-tts';
			params.reference_audio = referenceAudios[0].trim();
			params.ref_text = refText.trim();
		} else if (cloneActive && selected) {
			const refs = referenceAudios.slice(0, numSpeakers).map((r) => r.trim());
			params.model = pathOf(selected);
			params.engine = 'vibevoice';
			params.reference_audio = refs.length === 1 ? refs[0] : refs;
			if (refs.length > 1) params.num_speakers = refs.length;
			params.cfg_scale = cfgScale;
		} else if (engine === 'qwen3-tts') {
			params.voice = voice;
			params.model = selected.id;
			params.engine = 'qwen3-tts';
			if (caps.supports_instructions && instructions.trim()) {
				params.instructions = instructions.trim();
			}
		} else if (caps.engine === 'vibevoice') {
			// VibeVoice preset (clone toggle off): the pipeline resolves the
			// chosen voice id to its bundled reference clip — no user-supplied
			// reference sample needed.
			params.voice = voice;
			params.model = pathOf(selected);
			params.engine = 'vibevoice';
			params.cfg_scale = cfgScale;
		} else {
			// Worker engines (Kokoro repos) — always send the installed path.
			params.voice = voice;
			if (caps.supports_speed) params.speed = speed;
			params.model = pathOf(selected);
		}
		await submitJob(TASK, params);
		// First successful load knows the real speaker list — pick it up.
		if (engine === 'qwen3-tts' && liveQwenVoices.length === 0) {
			try {
				const status = await ttsStatus();
				if (status.phase === 'ready') liveQwenVoices = status.voices;
			} catch {
				// keep the fallback list
			}
		}
	}

	function useForChatVoice() {
		if (!selected) return;
		const modelRef = engine === 'qwen3-tts' ? selected.id : pathOf(selected);
		setSpeechVoiceConfig(engine, voice, modelRef);
		addToast('Chat replies will speak with this voice', 'success');
	}

	const isChatVoice = $derived(
		selected !== null &&
			voiceStore.engine === engine &&
			voiceStore.voice === voice &&
			(engine === 'qwen3-tts'
				? voiceStore.qwenModel === selected.id
				: voiceStore.ttsModel === pathOf(selected)),
	);

	function metaOf(r: TaskResult): Record<string, unknown> {
		return (r.meta ?? {}) as Record<string, unknown>;
	}

	/** Restore a recent generation's text + params into the form (model stays). */
	let remixNote = $state<string | null>(null);
	let remixToken = 0;
	let picker = $state<ModelPicker>();

	/**
	 * Restore a recent generation into the form. Voices, speed, style and
	 * cloning inputs belong to the model that spoke it, so they come back only
	 * when that model is still installed, and it is selected again (see
	 * remix.ts). Otherwise just the text returns.
	 */
	async function remix(a: Asset) {
		const token = ++remixToken;
		const installed = (await picker?.refresh()) ?? [];
		if (token !== remixToken) return; // a newer remix won
		const r = speechRemix(a, installed.map(remixModelOf));
		if (r.text !== undefined) text = r.text;
		remixNote = r.note;
		if (r.modelId === null) return;
		modelValue = r.modelId;
		// Swap the caps now, not when the picker reports back, so the voice
		// reset effect checks the restored voice against the right list.
		selected = installed.find((m) => m.id === r.modelId) ?? selected;
		if (r.voice !== undefined) voice = r.voice;
		if (r.speed !== undefined) speed = r.speed;
		instructions = r.instructions ?? '';
		if (r.cfgScale !== undefined) cfgScale = r.cfgScale;
		refText = r.refText ?? '';
		if (r.referenceAudios) {
			referenceAudios = r.referenceAudios;
			numSpeakers = r.referenceAudios.length;
		}
		cloneVoice = r.cloneVoice ?? false;
	}
</script>

<TaskShell task={TASK} assetTasks={['text_to_speech', 'vibevoice']} onremix={remix}>
	{#snippet controls()}
		<RemixNote note={remixNote} ondismiss={() => (remixNote = null)} />
		<ModelPicker
			bind:this={picker}
			hfTask="text-to-speech"
			capsFilter={SPEECH_FILTER}
			valueKind="id"
			emptyHint="speech"
			bind:value={modelValue}
			onmodel={(m) => (selected = m)}
		/>

		{#if selected}
			<!-- Engine line — informational; derived from the model, never flipped by hand -->
			<p class="wb-help -mt-2">
				<span class="wb-dot" style="--tone: {engine === 'qwen3-tts' ? 'var(--info)' : 'var(--success)'}">
					{engine === 'qwen3-tts'
						? qwenClone
							? 'Runs the qwen-tts command-line tool for voice cloning'
							: 'Runs on the Qwen3-TTS engine'
						: cloning
							? 'Runs on the Python runtime with VibeVoice cloning'
							: 'Runs on the Python runtime with Kokoro'}
				</span>
			</p>

			{#if engine === 'qwen3-tts' && !qwenEngineInstalled}
				<p class="wb-banner warn">
					The Qwen3-TTS engine files aren't in engine\qwentts\ yet. Install the tts-server runtime first.
				</p>
			{/if}
			{#if qwenClone && !qwenCliInstalled}
				<p class="wb-banner warn">
					Voice cloning needs qwen-tts.exe. Copy it into engine\qwentts\ next to tts-server.exe.
				</p>
			{/if}
		{/if}

		<div class="st-field">
			<label for="tts-text" class="st-label">
				Text
				{#if text.length > 0}<span class="num">{text.length}</span>{/if}
			</label>
			<textarea
				id="tts-text"
				class="wb-input"
				bind:value={text}
				rows="6"
				placeholder="Type what you want spoken"
			></textarea>
		</div>

		{#if selected}
			{#if cloning && caps.voices.length > 0}
				<!-- Models with presets AND cloning: dropdown + a clone toggle. -->
				<div class="st-field">
					<label for="tts-voice" class="st-label">Voice</label>
					<select id="tts-voice" class="wb-input" bind:value={voice} disabled={cloneVoice}>
						{#each voiceOptions as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
					</select>
				</div>
				<div class="st-row">
					<span>Clone a voice from a sample instead</span>
					<Toggle bind:checked={cloneVoice} label="Clone a voice from a sample instead" />
				</div>
			{:else if !cloning}
				<div class="st-field">
					<label for="tts-voice" class="st-label">Voice</label>
					<select id="tts-voice" class="wb-input" bind:value={voice}>
						{#each voiceOptions as o (o.value)}<option value={o.value}>{o.label}</option>{/each}
					</select>
				</div>
			{/if}

			{#if cloneActive}
				<div class="st-field">
					<label for="tts-reference" class="st-label">
						Reference voice{numSpeakers > 1 ? ', speaker 1' : ''}
					</label>
					<input
						id="tts-reference"
						class="wb-input"
						bind:value={referenceAudios[0]}
						placeholder="C:\path\to\voice-sample.wav"
						spellcheck="false"
					/>
					<p class="wb-help">A few seconds of clean speech as a wav, mp3, m4a, flac or ogg file. Paste the full path.</p>
				</div>
				{#if needsRefText}
					<div class="st-field">
						<label for="tts-ref-text" class="st-label">What the sample says (required)</label>
						<textarea
							id="tts-ref-text"
							class="wb-input"
							bind:value={refText}
							rows="2"
							placeholder="The exact words spoken in the sample"
						></textarea>
						<p class="wb-help">The closer this matches the audio, the better the cloned voice.</p>
					</div>
				{/if}
			{/if}

			<!-- Chat voice supports only Kokoro / Qwen3-TTS (preset-voice engines);
			     VibeVoice isn't a chat-voice engine, so hide the action for it. -->
			{#if !cloneActive && caps.engine !== 'vibevoice'}
				<button type="button" class="wb-btn ghost sm self-start" onclick={useForChatVoice} disabled={isChatVoice}>
					{#if isChatVoice}<Icon name="check" size={13} /> Chat replies use this voice{:else}Use this voice for chat replies{/if}
				</button>
			{/if}

			{#if caps.supports_speed || caps.supports_instructions || vibeClone}
				<AdvancedSection storageKey="speech">
					{#if caps.supports_speed}
						<Slider bind:value={speed} min={0.5} max={2} step={0.05} label="Speed" unit="×" />
					{/if}

					{#if caps.supports_instructions}
						<div class="st-field">
							<label for="tts-instructions" class="st-label">Style instructions (optional)</label>
							<textarea
								id="tts-instructions"
								class="wb-input"
								bind:value={instructions}
								rows="2"
								placeholder="For example: speak slowly and warmly"
							></textarea>
						</div>
					{/if}

					{#if vibeClone}
						<Slider value={numSpeakers} min={1} max={4} step={1} label="Speakers" onchange={setNumSpeakers} />
						{#each { length: Math.max(0, numSpeakers - 1) } as _, i (i)}
							<div class="st-field">
								<label for="tts-reference-{i + 2}" class="st-label">Reference voice, speaker {i + 2}</label>
								<input
									id="tts-reference-{i + 2}"
									class="wb-input"
									bind:value={referenceAudios[i + 1]}
									placeholder="C:\path\to\speaker-{i + 2}.wav"
									spellcheck="false"
								/>
							</div>
						{/each}
						{#if numSpeakers > 1}
							<p class="wb-help">
								For several speakers, start each line with “Speaker 1:”, “Speaker 2:” and so on. Each speaker
								clones its own sample.
							</p>
						{/if}
						<Slider bind:value={cfgScale} min={1} max={2} step={0.05} label="Guidance" />
					{/if}
				</AdvancedSection>
			{/if}
		{/if}
	{/snippet}

	{#snippet action()}
		<button type="button" class="wb-btn primary" disabled={!canSubmit} onclick={generate}>
			<Icon name="speaker" size={15} />
			{active ? 'Speaking…' : 'Speak'}
		</button>
	{/snippet}

	{#snippet resultView(result)}
		<div class="clip">
			<p class="clip-text">“{result.params['text']}”</p>
			{#each result.outputs as out (out)}
				<!-- svelte-ignore a11y_media_has_caption -->
				<audio controls src={fileSrc(out)} class="w-full"></audio>
			{/each}
			<p class="st-cap clip-cap">
				<span>
					{#if result.params['engine'] === 'vibevoice' || result.params['reference_audio'] !== undefined}
						Cloned voice
					{:else}
						{@const vid = String(metaOf(result)['voice'] ?? result.params['voice'] ?? voice)}
						Voice: {voiceOptions.find((o) => o.value === vid)?.label ?? prettyVoice(vid)}
					{/if}
				</span>
				{#if metaOf(result)['duration']}
					<span><span class="num">{Number(metaOf(result)['duration']).toFixed(1)}</span> s</span>
				{/if}
				<span>Saved to the library</span>
			</p>
		</div>
	{/snippet}

	{#snippet empty()}
		<span class="st-idle-ic"><Icon name="speaker" size={24} /></span>
		<b class="st-idle-t">No speech yet</b>
		<p class="st-idle-p">Pick a speech model, type some text and press Speak.</p>
	{/snippet}
</TaskShell>

<style>
	.clip {
		display: flex;
		flex-direction: column;
		gap: 12px;
		width: 100%;
		max-width: 560px;
		padding: 16px;
		border: 1px solid var(--border);
		border-radius: 12px;
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
	}
	.clip-text {
		display: -webkit-box;
		-webkit-line-clamp: 4;
		line-clamp: 4;
		-webkit-box-orient: vertical;
		overflow: hidden;
		font-size: var(--text-sm);
		line-height: 1.6;
		color: var(--foreground);
	}
	.clip-cap {
		justify-content: flex-start;
	}
</style>
