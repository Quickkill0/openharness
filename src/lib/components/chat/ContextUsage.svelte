<script lang="ts">
	/**
	 * ContextUsage — how full the loaded model's context window is for the
	 * active conversation. `variant="meter"` is the compact header gauge;
	 * `variant="panel"` adds the breakdown for the settings slide-over.
	 *
	 * The gauge prefers the server-reported figure from the last completed
	 * turn (prompt + completion tokens — exactly what the model saw and
	 * produced). Between turns the estimate (~4 chars/token + per-message
	 * overhead) covers anything typed since; the larger of the two wins.
	 */
	import { chatStore } from '$lib/stores/chat.svelte';
	import { engineStore } from '$lib/stores/engine.svelte';
	import { kTokens } from './chatMeta';

	interface Props {
		variant?: 'meter' | 'panel';
	}

	let { variant = 'panel' }: Props = $props();

	/** ~4 chars per token plus a little per-message chat-template overhead. */
	function estTokens(text: string): number {
		return Math.ceil(text.length / 4) + 8;
	}

	const messages = $derived(chatStore.messages);
	const systemPrompt = $derived(chatStore.settings?.system_prompt ?? '');
	const ctxWindow = $derived(engineStore.loadedCtx);

	const sysEst = $derived(systemPrompt ? estTokens(systemPrompt) : 0);
	const historyEst = $derived(messages.reduce((sum, m) => sum + estTokens(m.content), 0));

	/** Last turn's server-reported context consumption (exact). */
	const lastTurn = $derived.by(() => {
		for (let i = messages.length - 1; i >= 0; i--) {
			const m = messages[i];
			if (m.role === 'assistant' && m.prompt_tokens != null) {
				return (m.prompt_tokens ?? 0) + (m.completion_tokens ?? 0);
			}
		}
		return null;
	});

	const used = $derived(Math.max(lastTurn ?? 0, sysEst + historyEst));
	const pct = $derived(ctxWindow ? Math.min(100, (used / ctxWindow) * 100) : 0);
	const remaining = $derived(ctxWindow ? Math.max(0, ctxWindow - used) : null);
	const tone = $derived(pct >= 90 ? 'var(--error)' : pct >= 75 ? 'var(--warning)' : 'var(--tier-vram)');
</script>

{#if ctxWindow}
	{#if variant === 'meter'}
		<span class="meter" title="Context used: about {used} of {ctxWindow} tokens">
			<span>Context</span>
			<span class="tier-bar"><i style="width: {pct}%; background: {tone}"></i></span>
			<span class="num">{kTokens(used)} / {kTokens(ctxWindow)}</span>
		</span>
	{:else}
		<div>
			<div class="flex items-baseline justify-between text-xs">
				<span class="text-muted-foreground">Used</span>
				<span class="num text-foreground">
					{kTokens(used)} / {kTokens(ctxWindow)}
					<span class="text-muted-foreground">({Math.round(pct)}%)</span>
				</span>
			</div>
			<div class="tier-bar mt-2"><i style="width: {pct}%; background: {tone}"></i></div>
			<dl class="wb-props mt-3">
				<dt>System prompt</dt>
				<dd class="num">~{kTokens(sysEst)}</dd>
				<dt>Messages ({messages.length})</dt>
				<dd class="num">~{kTokens(historyEst)}</dd>
				{#if lastTurn != null}
					<dt>Last turn</dt>
					<dd class="num">{kTokens(lastTurn)}</dd>
				{/if}
				{#if remaining != null}
					<dt>Left</dt>
					<dd class="num">~{kTokens(remaining)}</dd>
				{/if}
			</dl>
			{#if pct >= 75}
				<p class="wb-help mt-2 {pct >= 90 ? 'text-error' : 'text-warning'}">
					{pct >= 90 ? 'The context window is nearly full.' : 'The context window is filling up.'}
					Older turns may be cut off. Raise the context length or start a new chat.
				</p>
			{/if}
		</div>
	{/if}
{/if}

<style>
	.meter {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
		white-space: nowrap;
	}
	.meter .tier-bar {
		width: 90px;
		height: 6px;
	}
	@media (max-width: 1199px) {
		.meter .tier-bar {
			width: 56px;
		}
	}
</style>
