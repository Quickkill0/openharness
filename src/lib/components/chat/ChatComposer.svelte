<script lang="ts">
	/**
	 * Chat composer (Workbench) — a ruled card pinned to the bottom of the
	 * transcript: auto-growing textarea, then a bar with the voice-input mic,
	 * a plain-language hint, and Send / Stop. The model picker lives in the
	 * chat header.
	 *
	 * With no engine installed the composer shows an inline install banner
	 * (InstallWizard) instead of a blocking gate page.
	 */
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import Icon from '$lib/components/shell/Icon.svelte';
	import InstallWizard from '$lib/components/engine/InstallWizard.svelte';
	import { engineStore } from '$lib/stores/engine.svelte';
	import { chatStore, cancelStream } from '$lib/stores/chat.svelte';
	import { voiceStore, startRecording, stopAndTranscribe } from '$lib/stores/voice.svelte';
	import { composerHint, composerPlaceholder } from './chatMeta';

	interface Props {
		/** Sends the trimmed message — the page creates a conversation if needed. */
		onsend: (text: string) => void | Promise<void>;
		/** Display name of the loaded chat model, for the placeholder. */
		modelName?: string | null;
		autofocus?: boolean;
	}

	let { onsend, modelName = null, autofocus = false }: Props = $props();

	let text = $state('');
	let isComposing = false;
	let wizardOpen = $state(false);
	let textareaEl = $state<HTMLTextAreaElement | null>(null);

	const phase = $derived(engineStore.phase);
	/** Some text engine can run models here (llama.cpp or the native engine). */
	const installed = $derived(engineStore.hasTextEngine);
	const ready = $derived(engineStore.isReady);
	const hasModel = $derived(engineStore.currentModelId !== null);
	const streaming = $derived(chatStore.streaming);
	const recording = $derived(voiceStore.recording);
	const transcribing = $derived(voiceStore.transcribing);
	const voiceError = $derived(voiceStore.error);
	const sttAvailable = $derived(voiceStore.sttAvailable);

	const canSend = $derived(ready && hasModel && text.trim().length > 0 && !streaming);

	const placeholder = $derived(composerPlaceholder({ installed, phase, hasModel, modelName }));
	const hint = $derived(composerHint({ recording, transcribing, voiceError }));

	// Auto-grow the textarea with content, capped at ~8 lines.
	$effect(() => {
		void text;
		const el = textareaEl;
		if (!el) return;
		el.style.height = 'auto';
		el.style.height = `${Math.min(Math.max(el.scrollHeight, 44), 168)}px`;
	});

	$effect(() => {
		if (autofocus) textareaEl?.focus();
	});

	// The wizard reports progress through engine status — close it once the
	// install kicks off or the engine comes back ready.
	$effect(() => {
		if (phase === 'ready' || phase === 'installing' || phase === 'loading') {
			wizardOpen = false;
		}
	});

	async function handleSend() {
		if (!canSend) return;
		const t = text.trim();
		text = '';
		try {
			await onsend(t);
		} catch {
			// Send never reached the backend (e.g. conversation create failed) —
			// give the user their message back instead of losing it.
			if (!text.trim()) text = t;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
			e.preventDefault();
			void handleSend();
		}
	}

	async function handleMic() {
		if (transcribing) return;
		if (!recording) {
			await startRecording();
			return;
		}
		const transcript = await stopAndTranscribe();
		if (transcript) {
			text = text.trim() ? `${text.trim()} ${transcript}` : transcript;
			textareaEl?.focus();
		}
	}
</script>

<div class="w-full">
	{#if !installed || phase === 'installing'}
		<div class="wb-banner warn mb-2 flex items-center justify-between gap-3">
			{#if phase === 'installing'}
				<span class="flex items-center gap-2">
					<Spinner size="sm" />
					Installing the llama.cpp engine. This is a one-time download.
				</span>
			{:else}
				<span>No engine can run models here yet. Install llama.cpp, or see <a class="link" href="/settings/engines">Settings, Engines</a> for the native engine.</span>
				<button type="button" class="wb-btn sm" onclick={() => (wizardOpen = true)}>Install llama.cpp</button>
			{/if}
		</div>
		<InstallWizard bind:open={wizardOpen} />
	{/if}

	<div class="composer">
		<textarea
			bind:this={textareaEl}
			bind:value={text}
			rows="2"
			{placeholder}
			aria-label="Message"
			onkeydown={handleKeydown}
			oncompositionstart={() => (isComposing = true)}
			oncompositionend={() => (isComposing = false)}
		></textarea>

		<div class="bar">
			<button
				type="button"
				class="wb-icon-btn {recording ? 'rec' : ''}"
				onclick={handleMic}
				disabled={transcribing || !sttAvailable}
				aria-label={recording ? 'Stop and transcribe' : 'Speak your message'}
				title={!sttAvailable ? voiceStore.sttHint : recording ? 'Stop and transcribe' : 'Speak your message'}
			>
				{#if transcribing}
					<Spinner size="sm" />
				{:else}
					<Icon name="mic" size={15} />
				{/if}
			</button>
			<span class="hint {hint.tone}" aria-live="polite">{hint.text}</span>
			<span class="flex-1"></span>
			{#if streaming}
				<button type="button" class="wb-btn sm" onclick={cancelStream}>
					<Icon name="stop" size={12} />Stop
				</button>
			{:else}
				<button type="button" class="wb-btn sm primary" onclick={handleSend} disabled={!canSend}>
					<Icon name="send" size={12} />Send
				</button>
			{/if}
		</div>
	</div>
</div>

<style>
	.composer {
		border-radius: 14px;
		border: 1px solid var(--border);
		background: var(--surface);
		box-shadow:
			0 10px 30px -14px rgb(0 0 0 / 0.35),
			inset 0 1px 0 var(--inset-highlight);
		padding: 10px 10px 8px 14px;
		transition:
			border-color 0.15s ease,
			box-shadow 0.15s ease;
	}
	.composer:focus-within {
		border-color: color-mix(in oklch, var(--primary) 55%, var(--border));
	}
	textarea {
		display: block;
		width: 100%;
		min-height: 44px;
		max-height: 168px;
		border: 0;
		background: transparent;
		resize: none;
		outline: none;
		font-size: var(--text-sm);
		line-height: 1.5;
		color: var(--foreground);
	}
	textarea::placeholder {
		color: var(--muted-foreground);
	}
	.bar {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 4px;
	}
	.hint {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.hint.err {
		color: var(--error);
	}
	.hint.live {
		color: var(--primary);
	}
	.rec {
		color: #fff !important;
		background: var(--error) !important;
		animation: rec-pulse 1.2s ease-in-out infinite;
	}
	@keyframes rec-pulse {
		50% {
			opacity: 0.65;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.rec {
			animation: none;
		}
	}
</style>
