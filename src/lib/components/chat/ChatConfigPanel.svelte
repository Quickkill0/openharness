<script lang="ts">
	/**
	 * Chat settings slide-over (Workbench): one scrolling column of ruled
	 * sections — system prompt, sampling, context, load settings, voice.
	 * System prompt and sampling save as you type (debounced) and apply from
	 * the next message; load settings need an explicit "Apply and reload"
	 * because they restart llama-server.
	 */
	import Slider from '$lib/components/ui/Slider.svelte';
	import Toggle from '$lib/components/ui/Toggle.svelte';
	import Icon from '$lib/components/shell/Icon.svelte';
	import FitBadge from '$lib/components/engine/FitBadge.svelte';
	import ContextUsage from '$lib/components/chat/ContextUsage.svelte';
	import { chatStore, saveSettings } from '$lib/stores/chat.svelte';
	import { engineStore, loadModel } from '$lib/stores/engine.svelte';
	import { voiceStore, setAutoSpeak } from '$lib/stores/voice.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';
	import { listLocalModels, planLoad } from '$lib/ipc/commands';
	import {
		DEFAULT_SAMPLING,
		type ConversationSettings,
		type InstalledModel,
		type LoadConfig,
		type SamplingParams,
	} from '$lib/types';

	interface Props {
		/** Renders a close button in the header. */
		onclose?: () => void;
	}

	let { onclose }: Props = $props();

	// Local draft with non-optional system_prompt + load_config to keep template
	// access straightforward.
	interface Draft {
		system_prompt: string;
		sampling: SamplingParams;
		load_config: LoadConfig;
	}

	const DEFAULT_LOAD: LoadConfig = {
		ctx: 4096,
		ngl: -1,
		flash_attn: false,
		mmap: true,
		mlock: false,
	};

	let draft = $state<Draft | null>(null);
	let lastConv = $state<string | null>(null);
	let stopText = $state('');
	let reloading = $state(false);

	const modelId = $derived(engineStore.currentModelId);

	// Loaded model's catalog record — the context slider tops out at the
	// model's trained context instead of an arbitrary constant.
	let loadedModel = $state<InstalledModel | null>(null);
	$effect(() => {
		const id = modelId;
		if (!id) {
			loadedModel = null;
			return;
		}
		listLocalModels()
			.then((models) => {
				loadedModel = models.find((m) => m.id === id) ?? null;
			})
			.catch(() => {
				loadedModel = null;
			});
	});

	const ctxTrain = $derived(loadedModel?.context_train);
	/** Slider max: the model's trained context rounded down to the 512 grid,
	 *  with 32k as the fallback when the model didn't declare one. */
	const ctxMax = $derived(
		ctxTrain && ctxTrain >= 1024 ? Math.floor(ctxTrain / 512) * 512 : 32768,
	);
	const ctxMaxLabel = $derived(
		ctxTrain
			? ctxTrain >= 1024 && ctxTrain % 1024 === 0
				? `${ctxTrain / 1024}K`
				: String(ctxTrain)
			: null,
	);

	$effect(() => {
		const id = chatStore.activeId;
		const s = chatStore.settings;
		if (id !== lastConv && s) {
			draft = {
				system_prompt: s.system_prompt ?? '',
				sampling: JSON.parse(JSON.stringify(s.sampling)),
				load_config: s.load_config
					? JSON.parse(JSON.stringify(s.load_config))
					: { ...DEFAULT_LOAD },
			};
			stopText = draft.sampling.stop.join(', ');
			lastConv = id;
		}
	});

	function toSettings(d: Draft): ConversationSettings {
		return {
			system_prompt: d.system_prompt,
			sampling: d.sampling,
			load_config: d.load_config,
		};
	}

	let saveTimer: ReturnType<typeof setTimeout> | null = null;
	function scheduleSave() {
		if (!draft) return;
		if (saveTimer) clearTimeout(saveTimer);
		const snapshot = toSettings(draft);
		saveTimer = setTimeout(() => saveSettings(snapshot), 400);
	}

	function intOrUndef(v: string): number | undefined {
		const t = v.trim();
		if (t === '') return undefined;
		const n = Number(t);
		return Number.isFinite(n) ? n : undefined;
	}

	function resetSampling() {
		if (!draft) return;
		draft.sampling = { ...DEFAULT_SAMPLING, stop: [] };
		stopText = '';
		scheduleSave();
	}

	function commitStop() {
		if (!draft) return;
		draft.sampling.stop = stopText.split(',').map((s) => s.trim()).filter(Boolean);
		scheduleSave();
	}

	async function applyLoadConfig() {
		if (!draft || !modelId) return;
		const lc = draft.load_config;
		reloading = true;
		try {
			saveSettings(toSettings(draft));
			// A native model reloads through the planner: an explicit GPU-layer
			// count would send it down the llama.cpp path.
			const kind =
				engineStore.runningKind ??
				(await planLoad(modelId, { ctx: lc.ctx }).then((p) => p.engine_kind, () => null));
			if (kind === 'ssd_stream') {
				await loadModel(modelId, { ctx: lc.ctx, overrides: { engine: 'ssd_stream' } });
				addToast('Model reloaded with the new settings', 'success');
				return;
			}
			await loadModel(modelId, {
				ctx: lc.ctx,
				ngl: lc.ngl,
				batchSize: lc.batch_size,
				threads: lc.threads,
				flashAttn: lc.flash_attn,
				kvCacheType: lc.kv_cache_type,
				mmap: lc.mmap,
				mlock: lc.mlock,
			});
			addToast('Model reloaded with the new settings', 'success');
		} catch (e) {
			// Also lands here when the user backs out of the eviction dialog.
			addToast(`Reload failed: ${e instanceof Error ? e.message : e}`, 'error');
		} finally {
			reloading = false;
		}
	}

	const SYSTEM_PRESETS = [
		{ label: 'None', value: '' },
		{ label: 'Helpful assistant', value: 'You are a helpful, honest assistant. Answer concisely and accurately.' },
		{ label: 'Concise', value: 'Be extremely concise. Prefer short, direct answers with no preamble.' },
		{ label: 'Code expert', value: 'You are an expert software engineer. Provide correct, idiomatic code with brief explanations. Use fenced code blocks.' },
		{ label: 'Socratic tutor', value: 'You are a patient tutor. Guide the user with questions and step-by-step reasoning rather than giving the answer outright.' },
	];
	const presetValue = $derived(
		draft && SYSTEM_PRESETS.some((p) => p.value === draft?.system_prompt) ? draft.system_prompt : '__custom',
	);

	const KV_CACHE_TYPES = [
		{ value: '', label: 'Default (f16)' },
		{ value: 'q8_0', label: 'q8_0, half the size' },
		{ value: 'q4_0', label: 'q4_0, a quarter of the size' },
	];

	const ttsLabel = $derived(
		voiceStore.engine === 'qwen3-tts' ? `Qwen3-TTS, ${voiceStore.voice}` : `Kokoro, ${voiceStore.voice}`,
	);
</script>

<div class="panel">
	<div class="head">
		<div class="min-w-0">
			<h2>Chat settings</h2>
			<p class="wb-help">For this conversation only</p>
		</div>
		{#if onclose}
			<button type="button" class="wb-icon-btn" onclick={onclose} aria-label="Close settings" title="Close (Esc)">
				<Icon name="x" size={15} />
			</button>
		{/if}
	</div>

	{#if !draft}
		<div class="wb-empty flex-1">
			<b>No conversation open</b>
			Pick a chat or send a message to configure it.
		</div>
	{:else}
		<div class="scroll">
			<!-- System prompt -->
			<section class="wb-sec">
				<div class="wb-sec-h">
					<label class="wb-sec-label" for="sysprompt">System prompt</label>
					<select
						class="wb-input preset"
						aria-label="System prompt preset"
						value={presetValue}
						onchange={(e) => {
							const v = e.currentTarget.value;
							if (draft && v !== '__custom') {
								draft.system_prompt = v;
								scheduleSave();
							}
						}}
					>
						{#each SYSTEM_PRESETS as p (p.label)}
							<option value={p.value}>{p.label}</option>
						{/each}
						{#if presetValue === '__custom'}
							<option value="__custom">Custom</option>
						{/if}
					</select>
				</div>
				<textarea
					id="sysprompt"
					rows="5"
					class="wb-input w-full"
					bind:value={draft.system_prompt}
					oninput={scheduleSave}
					placeholder="Instructions the model reads before every message in this chat"
				></textarea>
				<p class="wb-help mt-1.5">Applies from the next message. No reload needed.</p>
			</section>

			<!-- Sampling -->
			<section class="wb-sec">
				<div class="wb-sec-h">
					<span class="wb-sec-label">Sampling</span>
					<button type="button" class="wb-btn ghost sm" onclick={resetSampling}>Reset</button>
				</div>
				<div class="stack">
					<Slider label="Temperature" min={0} max={2} step={0.01} bind:value={draft.sampling.temperature} onchange={scheduleSave} />
					<Slider label="Top-p" min={0} max={1} step={0.01} bind:value={draft.sampling.top_p} onchange={scheduleSave} />
					<Slider label="Top-k" min={0} max={100} step={1} bind:value={draft.sampling.top_k} onchange={scheduleSave} />
					<Slider label="Min-p" min={0} max={1} step={0.01} bind:value={draft.sampling.min_p} onchange={scheduleSave} />
					<div class="row2">
						<label for="maxtok">Max reply tokens</label>
						<input
							id="maxtok"
							type="number"
							class="wb-input num w-24 text-right"
							placeholder="No limit"
							value={draft.sampling.max_tokens ?? ''}
							oninput={(e) => {
								if (draft) {
									draft.sampling.max_tokens = intOrUndef(e.currentTarget.value);
									scheduleSave();
								}
							}}
						/>
					</div>
				</div>
				<details class="more">
					<summary><Icon name="chevron" size={12} class="chev" />Penalties, seed and stop words</summary>
					<div class="stack mt-3">
						<Slider label="Repeat penalty" min={0.8} max={2} step={0.01} bind:value={draft.sampling.repeat_penalty} onchange={scheduleSave} />
						<Slider label="Presence penalty" min={-2} max={2} step={0.01} bind:value={draft.sampling.presence_penalty} onchange={scheduleSave} />
						<Slider label="Frequency penalty" min={-2} max={2} step={0.01} bind:value={draft.sampling.frequency_penalty} onchange={scheduleSave} />
						<div class="row2">
							<label for="seed">Seed</label>
							<input
								id="seed"
								type="number"
								class="wb-input num w-24 text-right"
								placeholder="Random"
								value={draft.sampling.seed ?? ''}
								oninput={(e) => {
									if (draft) {
										draft.sampling.seed = intOrUndef(e.currentTarget.value);
										scheduleSave();
									}
								}}
							/>
						</div>
						<div class="wb-field">
							<div class="wb-field-top"><label for="stop">Stop words, comma separated</label></div>
							<input
								id="stop"
								class="wb-input"
								bind:value={stopText}
								onchange={commitStop}
								placeholder="For example ###"
							/>
						</div>
					</div>
				</details>
			</section>

			<!-- Context -->
			{#if engineStore.loadedCtx}
				<section class="wb-sec">
					<div class="wb-sec-h"><span class="wb-sec-label">Context window</span></div>
					<ContextUsage variant="panel" />
				</section>
			{/if}

			<!-- Load settings -->
			<section class="wb-sec">
				<div class="wb-sec-h"><span class="wb-sec-label">Load settings</span></div>
				{#if !modelId}
					<p class="wb-help">
						Load a model from the picker or the <a href="/models" class="text-primary underline">Models</a> page
						to tune how it loads.
					</p>
				{:else}
					<dl class="wb-props">
						<dt>Model</dt>
						<dd class="truncate" title={modelId}>{loadedModel?.display_name ?? modelId}</dd>
						<dt>KV cache</dt>
						<dd>{draft.load_config.kv_cache_type || 'f16'}</dd>
						<dt>Fit</dt>
						<dd>
							{#key draft.load_config.ctx}
								<FitBadge {modelId} ctx={draft.load_config.ctx} />
							{/key}
						</dd>
					</dl>
					<div class="mt-3">
						<Slider label="Context length" unit="tok" min={512} max={ctxMax} step={512} bind:value={draft.load_config.ctx} />
						<p class="wb-help mt-1">
							{#if ctxMaxLabel}
								The model was trained on up to <span class="num">{ctxMaxLabel}</span> tokens.
							{:else}
								The model doesn't say how much it was trained on, so the slider stops at 32K.
							{/if}
						</p>
					</div>
					<details class="more">
						<summary><Icon name="chevron" size={12} class="chev" />GPU layers, threads and memory</summary>
						<div class="stack mt-3">
							<div class="row2">
								<label for="ngl">GPU layers</label>
								<input
									id="ngl"
									type="number"
									class="wb-input num w-24 text-right"
									value={draft.load_config.ngl}
									oninput={(e) => {
										if (draft) draft.load_config.ngl = intOrUndef(e.currentTarget.value) ?? -1;
									}}
								/>
							</div>
							<p class="wb-help -mt-2">-1 puts every layer on the GPU, 0 runs on the CPU only.</p>
							{#if engineStore.runningKind === 'ssd_stream'}
								<p class="wb-help -mt-2">This model runs on the native engine, so only the context length applies when you reload.</p>
							{/if}
							<div class="row2">
								<label for="batch">Batch size</label>
								<input
									id="batch"
									type="number"
									class="wb-input num w-24 text-right"
									placeholder="Auto"
									value={draft.load_config.batch_size ?? ''}
									oninput={(e) => {
										if (draft) draft.load_config.batch_size = intOrUndef(e.currentTarget.value);
									}}
								/>
							</div>
							<div class="row2">
								<label for="threads">Threads</label>
								<input
									id="threads"
									type="number"
									class="wb-input num w-24 text-right"
									placeholder="Auto"
									value={draft.load_config.threads ?? ''}
									oninput={(e) => {
										if (draft) draft.load_config.threads = intOrUndef(e.currentTarget.value);
									}}
								/>
							</div>
							<div class="row2">
								<label for="kv">KV cache type</label>
								<select
									id="kv"
									class="wb-input"
									value={draft.load_config.kv_cache_type ?? ''}
									onchange={(e) => {
										if (draft) draft.load_config.kv_cache_type = e.currentTarget.value || undefined;
									}}
								>
									{#each KV_CACHE_TYPES as k (k.value)}
										<option value={k.value}>{k.label}</option>
									{/each}
								</select>
							</div>
							<div class="row2">
								<span>Flash attention</span>
								<Toggle label="Flash attention" bind:checked={draft.load_config.flash_attn} />
							</div>
							<div class="row2">
								<span>Memory-map the file</span>
								<Toggle label="Memory-map the file" bind:checked={draft.load_config.mmap} />
							</div>
							<div class="row2">
								<span>Lock in system memory</span>
								<Toggle label="Lock in system memory" bind:checked={draft.load_config.mlock} />
							</div>
						</div>
					</details>
					<button type="button" class="wb-btn primary block mt-3" onclick={applyLoadConfig} disabled={reloading}>
						{reloading ? 'Reloading…' : 'Apply and reload model'}
					</button>
					<p class="wb-help mt-1.5">Reloading restarts the engine, so chat pauses for a moment.</p>
				{/if}
			</section>

			<!-- Voice -->
			<section class="wb-sec">
				<div class="wb-sec-h"><span class="wb-sec-label">Voice</span></div>
				<dl class="wb-props">
					<dt>Speech to text</dt>
					<dd>
						{#if voiceStore.sttAvailable}
							<span class="wb-dot" style="--tone: var(--success)">Ready</span>
						{:else}
							<span class="text-muted-foreground">No transcription model</span>
						{/if}
					</dd>
					<dt>Speak replies with</dt>
					<dd>
						{#if voiceStore.ttsAvailable}
							{ttsLabel}
						{:else}
							<span class="text-muted-foreground">No speech model</span>
						{/if}
					</dd>
				</dl>
				<div class="row2 mt-3">
					<span>Read replies aloud</span>
					<Toggle
						label="Read replies aloud"
						checked={voiceStore.autoSpeak}
						disabled={!voiceStore.ttsAvailable}
						onchange={(v) => setAutoSpeak(v)}
					/>
				</div>
				<p class="wb-help mt-2">
					{#if !voiceStore.sttAvailable || !voiceStore.ttsAvailable}
						Install a Whisper model and a speech model from the <a href="/models/browse" class="text-primary underline">Store</a>
						to talk to the model.
					{:else}
						Change the reply voice in <a href="/studio?task=speech" class="text-primary underline">Studio, Speech</a>.
					{/if}
				</p>
			</section>
		</div>
	{/if}
</div>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		height: 100%;
		width: 100%;
		background: var(--surface);
	}
	.head {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 8px;
		padding: 14px 12px 12px 16px;
		border-bottom: 1px solid var(--border);
	}
	h2 {
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--foreground);
	}
	.scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
	}
	.preset {
		height: 26px;
		max-width: 150px;
		font-size: var(--text-xs);
	}
	textarea {
		font-size: var(--text-xs);
	}
	.stack {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.row2 {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--muted-foreground);
	}
	.row2 .wb-input {
		height: 28px;
		font-size: var(--text-xs);
	}
	.more {
		margin-top: 14px;
	}
	.more summary {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		list-style: none;
		cursor: pointer;
		font-size: var(--text-xs);
		color: var(--muted-foreground);
	}
	.more summary::-webkit-details-marker {
		display: none;
	}
	.more summary:hover {
		color: var(--foreground);
	}
	.more summary:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 2px;
		border-radius: 4px;
	}
	.more summary :global(.chev) {
		transition: transform 0.15s ease;
	}
	.more[open] summary :global(.chev) {
		transform: rotate(90deg);
	}
	@media (prefers-reduced-motion: reduce) {
		.more summary :global(.chev) {
			transition: none;
		}
	}
</style>
