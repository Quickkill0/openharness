<script lang="ts">
	/**
	 * TranscribeTask (M16 W4) — simple by default: audio file + model +
	 * Transcribe. Mode (translate), language hint, and word-level timestamps
	 * live in the Advanced disclosure.
	 */
	import Icon from '$lib/components/shell/Icon.svelte';
	import Toggle from '$lib/components/ui/Toggle.svelte';
	import ModelPicker from './ModelPicker.svelte';
	import TaskShell from './TaskShell.svelte';
	import AdvancedSection from './AdvancedSection.svelte';
	import RemixNote from './RemixNote.svelte';
	import { remixModelOf, transcribeRemix } from './remix';
	import { fileSrc } from '$lib/ipc/backend';
	import { studio, submitJob, type TaskResult } from '$lib/stores/studio.svelte';
	import type { Asset } from '$lib/types';

	const TASK = 'transcribe' as const;

	let audioPath = $state('');
	let model = $state(''); // installed automatic-speech-recognition model dir
	let language = $state('');
	let mode = $state<'transcribe' | 'translate'>('transcribe');
	let wordTimestamps = $state(false);

	const active = $derived(studio.active[TASK]);
	const result = $derived(studio.results[TASK]);
	const canSubmit = $derived(
		audioPath.trim().length > 0 && model.trim().length > 0 && !active,
	);

	const modeOptions = [
		{ value: 'transcribe', label: 'Same language' },
		{ value: 'translate', label: 'Translate to English' },
	];

	// Transcript text, loaded from the job's .txt output via the asset protocol.
	let transcript = $state<string | null>(null);
	let transcriptJob = $state<string | null>(null);

	$effect(() => {
		const r = result;
		if (!r || r.jobId === transcriptJob) return;
		transcriptJob = r.jobId;
		transcript = null;
		const txt = r.outputs.find((o) => o.toLowerCase().endsWith('.txt'));
		if (!txt) return;
		const src = fileSrc(txt);
		if (!src) {
			transcript = 'The transcript is saved. Open it from the Library tab.';
			return;
		}
		fetch(src)
			.then((res) => (res.ok ? res.text() : Promise.reject(res.status)))
			.then((t) => (transcript = t))
			.catch(() => (transcript = 'The transcript is saved. Open it from the Library tab.'));
	});

	async function generate() {
		const params: Record<string, unknown> = {
			audio: audioPath.trim(),
			model,
			task: mode,
			word_timestamps: wordTimestamps,
		};
		if (language.trim()) params.language = language.trim();
		await submitJob(TASK, params);
	}

	function metaOf(r: TaskResult): Record<string, unknown> {
		return (r.meta ?? {}) as Record<string, unknown>;
	}

	/** Restore a recent transcription's source + settings into the form. */
	let remixNote = $state<string | null>(null);
	let remixToken = 0;
	let picker = $state<ModelPicker>();

	/** Restore a recent transcription; the model returns only if it is still installed. */
	async function remix(a: Asset) {
		const token = ++remixToken;
		const installed = (await picker?.refresh()) ?? [];
		if (token !== remixToken) return; // a newer remix won
		const r = transcribeRemix(a, installed.map(remixModelOf));
		if (r.audioPath !== undefined) audioPath = r.audioPath;
		if (r.model !== null) model = r.model;
		language = r.language;
		mode = r.mode;
		wordTimestamps = r.wordTimestamps;
		remixNote = r.note;
	}
</script>

<TaskShell task={TASK} onremix={remix}>
	{#snippet controls()}
		<RemixNote note={remixNote} ondismiss={() => (remixNote = null)} />
		<ModelPicker bind:this={picker} hfTask="automatic-speech-recognition" bind:value={model} />

		<div class="st-field">
			<label for="stt-audio" class="st-label">Audio file</label>
			<input
				id="stt-audio"
				class="wb-input"
				bind:value={audioPath}
				placeholder="C:\path\to\recording.mp3"
				spellcheck="false"
			/>
			<p class="wb-help">Paste the full path to a wav, mp3, m4a, flac or ogg file.</p>
		</div>

		<div class="st-field">
			<span class="st-label">Mode</span>
			<div class="wb-seg w-full" role="radiogroup" aria-label="Mode">
				{#each modeOptions as o (o.value)}
					<button
						type="button"
						role="radio"
						class="flex-1"
						aria-checked={mode === o.value}
						onclick={() => (mode = o.value as typeof mode)}
					>
						{o.label}
					</button>
				{/each}
			</div>
		</div>

		<AdvancedSection storageKey="transcribe">
			<div class="st-field">
				<label for="stt-lang" class="st-label">Language</label>
				<input id="stt-lang" class="wb-input" bind:value={language} placeholder="Detect automatically, or en, de, ja" />
			</div>

			<div class="st-row">
				<span>Word-level timestamps</span>
				<Toggle bind:checked={wordTimestamps} label="Word-level timestamps" />
			</div>
		</AdvancedSection>
	{/snippet}

	{#snippet action()}
		<button type="button" class="wb-btn primary" disabled={!canSubmit} onclick={generate}>
			<Icon name="doc" size={15} />
			{active ? 'Transcribing…' : 'Transcribe'}
		</button>
	{/snippet}

	{#snippet resultView(result)}
		<div class="transcript">
			<div class="t-head">
				<span class="t-title">Transcript</span>
				{#if metaOf(result)['language']}
					<span class="wb-badge">{String(metaOf(result)['language']).toUpperCase()}</span>
				{/if}
				{#if metaOf(result)['duration']}
					<span><span class="num">{Number(metaOf(result)['duration']).toFixed(1)}</span> s of audio</span>
				{/if}
				{#if metaOf(result)['segment_count']}
					<span><span class="num">{metaOf(result)['segment_count']}</span> segments</span>
				{/if}
				<span class="flex-1"></span>
				{#if transcript}
					<button
						type="button"
						class="wb-icon-btn"
						onclick={() => navigator.clipboard?.writeText(transcript ?? '')}
						aria-label="Copy transcript"
						title="Copy transcript"
					>
						<Icon name="copy" size={14} />
					</button>
				{/if}
			</div>
			{#if transcript === null}
				<p class="t-body muted">Loading the transcript…</p>
			{:else}
				<pre class="t-body">{transcript}</pre>
			{/if}
			<p class="t-foot">Saved as .txt and .json in the library.</p>
		</div>
	{/snippet}

	{#snippet empty()}
		<span class="st-idle-ic"><Icon name="mic" size={24} /></span>
		<b class="st-idle-t">No transcript yet</b>
		<p class="st-idle-p">Pick a Whisper model, point at an audio file and press Transcribe.</p>
	{/snippet}
</TaskShell>

<style>
	.transcript {
		display: flex;
		flex-direction: column;
		width: 100%;
		max-width: 720px;
		max-height: 100%;
		border: 1px solid var(--border);
		border-radius: 12px;
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
		overflow: hidden;
	}
	.t-head {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 8px 10px 8px 14px;
		border-bottom: 1px solid var(--border);
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.t-title {
		font-size: var(--text-xs);
		font-weight: 600;
		color: var(--foreground);
	}
	.t-body {
		flex: 1;
		min-height: 0;
		max-height: 52vh;
		overflow-y: auto;
		margin: 0;
		padding: 14px;
		white-space: pre-wrap;
		font-family: inherit;
		font-size: var(--text-sm);
		line-height: 1.65;
		color: var(--foreground);
	}
	.t-body.muted {
		color: var(--muted-foreground);
	}
	.t-foot {
		padding: 8px 14px;
		border-top: 1px solid var(--border);
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
</style>
