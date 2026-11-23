<script lang="ts">
	/**
	 * Chat (Workbench) — a 240px conversation pane (search, new chat, rows
	 * grouped Today / Yesterday / Earlier) beside the chat column. The column
	 * header carries the model picker (load-from-picker), the context meter,
	 * voice mode and the settings toggle; the transcript is a centred 720px
	 * column with the composer pinned over its bottom edge; conversation
	 * settings open as a slide-over inside the chat column.
	 */
	import { engineStore, refreshNativeEngine } from '$lib/stores/engine.svelte';
	import {
		chatStore,
		createConversation,
		selectConversation,
		sendMessage,
		deleteConversation,
		deleteFromHere,
		regenerate,
		searchMessages,
		clearSearch,
		clearError,
		setAssistantDoneListener,
	} from '$lib/stores/chat.svelte';
	import {
		voiceStore,
		setAutoSpeak,
		speak,
		stopSpeaking,
		clearVoiceError,
		refreshVoiceModels,
	} from '$lib/stores/voice.svelte';
	import { onMount } from 'svelte';
	import Icon from '$lib/components/shell/Icon.svelte';
	import MarkdownMessage from '$lib/components/chat/MarkdownMessage.svelte';
	import StreamingMarkdown from '$lib/components/chat/StreamingMarkdown.svelte';
	import ChatConfigPanel from '$lib/components/chat/ChatConfigPanel.svelte';
	import ChatComposer from '$lib/components/chat/ChatComposer.svelte';
	import ChatModelPicker from '$lib/components/chat/ChatModelPicker.svelte';
	import ContextUsage from '$lib/components/chat/ContextUsage.svelte';
	import ThinkingBlock from '$lib/components/chat/ThinkingBlock.svelte';
	import { groupConversations, relTime } from '$lib/components/chat/chatMeta';

	let configOpen = $state(false);
	let searchQuery = $state('');
	let scrollEl = $state<HTMLDivElement | null>(null);
	let modelName = $state<string | null>(null);
	let copiedSeq = $state<number | null>(null);

	const ready = $derived(engineStore.isReady);
	const hasModel = $derived(engineStore.currentModelId !== null);
	const conversations = $derived(chatStore.conversations);
	const groups = $derived(groupConversations(conversations));
	const activeId = $derived(chatStore.activeId);
	const activeTitle = $derived(chatStore.activeConversation?.title ?? null);
	const messages = $derived(chatStore.messages);
	const streaming = $derived(chatStore.streaming);
	const error = $derived(chatStore.error);
	const searchHits = $derived(chatStore.searchHits);
	const autoSpeak = $derived(voiceStore.autoSpeak);
	const speakingSeq = $derived(voiceStore.speakingSeq);
	const voiceError = $derived(voiceStore.error);
	const ttsAvailable = $derived(voiceStore.ttsAvailable);

	// Probe installed STT/TTS models so voice buttons enable/disable honestly.
	onMount(() => {
		void refreshVoiceModels();
		void refreshNativeEngine();
	});

	/** Empty state: no conversation, or a fresh one with nothing in it yet. */
	const showEmpty = $derived(!activeId || (messages.length === 0 && !streaming));

	const greeting = $derived.by(() => {
		const h = new Date().getHours();
		if (h < 5) return 'Up late?';
		if (h < 12) return 'Good morning';
		if (h < 18) return 'Good afternoon';
		return 'Good evening';
	});

	// Voice mode: speak finished assistant replies when auto-speak is on.
	$effect(() => {
		setAssistantDoneListener((content, seq) => {
			if (voiceStore.autoSpeak) speak(content, seq);
		});
		return () => setAssistantDoneListener(null);
	});

	// Escape closes the settings slide-over.
	$effect(() => {
		if (!configOpen) return;
		const onKey = (e: KeyboardEvent) => {
			if (e.key === 'Escape') configOpen = false;
		};
		window.addEventListener('keydown', onKey);
		return () => window.removeEventListener('keydown', onKey);
	});

	function toggleSpeak(content: string, seq: number) {
		if (speakingSeq === seq) stopSpeaking();
		else speak(content, seq);
	}

	function handleRegenerate(seq: number) {
		if (streaming) return;
		void regenerate(seq);
	}

	function handleDeleteFrom(seq: number) {
		if (streaming) return;
		void deleteFromHere(seq);
	}

	/** Composer send — creates a conversation on the fly from the empty state. */
	async function handleSend(text: string) {
		if (!chatStore.activeId) {
			const id = await createConversation();
			// Throw so the composer restores the typed message.
			if (!id) throw new Error('could not create a conversation');
		}
		await sendMessage(text);
	}

	function handleNewChat() {
		void createConversation();
	}

	function handleDelete(id: string, e: Event) {
		e.stopPropagation();
		void deleteConversation(id);
	}

	function copyMessage(content: string, seq: number) {
		navigator.clipboard?.writeText(content);
		copiedSeq = seq;
		setTimeout(() => {
			if (copiedSeq === seq) copiedSeq = null;
		}, 1500);
	}

	function formatTime(ts: number) {
		return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}

	function onSearchInput() {
		if (searchQuery.trim()) searchMessages(searchQuery);
		else clearSearch();
	}

	// Auto-scroll to the newest content as messages stream in.
	$effect(() => {
		void messages.length;
		void streaming?.partial;
		void (streaming && chatStore.thinkingFor(streaming.seq));
		if (scrollEl) {
			queueMicrotask(() => {
				scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: 'smooth' });
			});
		}
	});

	$effect(() => {
		if (error) {
			const t = setTimeout(() => clearError(), 5000);
			return () => clearTimeout(t);
		}
	});

	$effect(() => {
		if (voiceError) {
			const t = setTimeout(() => clearVoiceError(), 6000);
			return () => clearTimeout(t);
		}
	});
</script>

<div class="chat">
	<!-- Conversation pane -->
	<aside class="convs" aria-label="Conversations">
		<div class="convs-head">
			<label class="wb-search min-w-0 flex-1">
				<Icon name="search" size={13} />
				<input
					type="search"
					placeholder="Search chats"
					aria-label="Search chats"
					bind:value={searchQuery}
					oninput={onSearchInput}
				/>
			</label>
			<button type="button" class="wb-icon-btn" onclick={handleNewChat} aria-label="New chat" title="New chat">
				<Icon name="plus" size={15} />
			</button>
		</div>

		<div class="convs-list">
			{#if searchQuery.trim()}
				{#if searchHits.length === 0}
					<p class="convs-none">No messages match “{searchQuery.trim()}”</p>
				{:else}
					<p class="group-l">Matching messages</p>
					{#each searchHits as hit, i (i)}
						<button
							type="button"
							class="conv hit"
							onclick={() => {
								selectConversation(hit.conversation_id);
								searchQuery = '';
								clearSearch();
							}}
						>
							<!-- eslint-disable-next-line svelte/no-at-html-tags — snippet markup is FTS-generated <mark> only -->
							<span class="cm">{@html hit.snippet}</span>
						</button>
					{/each}
				{/if}
			{:else if conversations.length === 0}
				<div class="convs-none">
					No chats yet. Type below to start one.
				</div>
			{:else}
				{#each groups as g (g.label)}
					<p class="group-l">{g.label}</p>
					{#each g.items as conv (conv.id)}
						<div class="conv-row {activeId === conv.id ? 'on' : ''}">
							<button
								type="button"
								class="conv"
								aria-current={activeId === conv.id ? 'true' : undefined}
								onclick={() => selectConversation(conv.id)}
							>
								<span class="ct">{conv.title || 'Untitled'}</span>
								<span class="cm">
									<span class="num">{conv.message_count}</span>
									{conv.message_count === 1 ? 'message' : 'messages'}, {relTime(conv.updated_at)}
								</span>
							</button>
							<button
								type="button"
								class="conv-del"
								onclick={(e) => handleDelete(conv.id, e)}
								aria-label="Delete chat {conv.title || 'Untitled'}"
								title="Delete chat"
							>
								<Icon name="trash" size={13} />
							</button>
						</div>
					{/each}
				{/each}
			{/if}
		</div>
	</aside>

	<!-- Chat column -->
	<section class="chat-main {configOpen ? 'with-panel' : ''}">
		<header class="chat-head">
			<ChatModelPicker oncurrent={(n) => (modelName = n)} />
			<span class="title" title={activeTitle ?? undefined}>
				{activeTitle ?? (activeId ? 'New chat' : '')}
			</span>
			<span class="flex-1"></span>
			<ContextUsage variant="meter" />
			<button
				type="button"
				class="wb-icon-btn {autoSpeak ? 'on' : ''}"
				onclick={() => setAutoSpeak(!autoSpeak)}
				disabled={!ttsAvailable}
				aria-label="Read replies aloud"
				aria-pressed={autoSpeak}
				title={!ttsAvailable
					? voiceStore.ttsHint
					: autoSpeak
						? 'Replies are read aloud. Click to turn off'
						: 'Read replies aloud'}
			>
				<Icon name="speaker" size={15} />
			</button>
			<button
				type="button"
				class="wb-icon-btn {configOpen ? 'on' : ''}"
				onclick={() => (configOpen = !configOpen)}
				aria-label="Chat settings"
				aria-pressed={configOpen}
				title="Chat settings"
			>
				<Icon name="sliders" size={15} />
			</button>
		</header>

		{#if showEmpty}
			<div class="empty">
				<div class="empty-inner">
					<h2>{greeting}</h2>
					<p>
						{#if ready && hasModel}
							What can I help with?
						{:else}
							Pick a model at the top left to get started.
						{/if}
					</p>
					<div class="mt-6 w-full">
						<ChatComposer onsend={handleSend} {modelName} autofocus />
					</div>
					{#if error}
						<p class="wb-banner err mt-3 w-full">{error}</p>
					{/if}
					{#if conversations.length > 0}
						<div class="recents">
							<p class="group-l">Recent chats</p>
							{#each conversations.slice(0, 5) as conv (conv.id)}
								<button type="button" class="recent" onclick={() => selectConversation(conv.id)}>
									<Icon name="chat" size={14} class="text-muted-foreground" />
									<span class="rt">{conv.title || 'Untitled'}</span>
									<span class="ra">{relTime(conv.updated_at)}</span>
								</button>
							{/each}
						</div>
					{/if}
				</div>
			</div>
		{:else}
			<div bind:this={scrollEl} class="transcript">
				{#each messages as msg (msg.seq)}
					{#if msg.role === 'user'}
						<div class="msg user">
							<div class="bubble">{msg.content}</div>
							<div class="msg-foot end">
								<button
									type="button"
									class="foot-btn"
									onclick={() => copyMessage(msg.content, msg.seq)}
									aria-label="Copy message"
									title="Copy"
								>
									<Icon name={copiedSeq === msg.seq ? 'check' : 'copy'} size={13} />
								</button>
								<button
									type="button"
									class="foot-btn danger"
									onclick={() => handleDeleteFrom(msg.seq)}
									disabled={streaming !== null}
									aria-label="Delete this message and everything after it"
									title="Delete from here"
								>
									<Icon name="trash" size={13} />
								</button>
							</div>
						</div>
					{:else}
						<div class="msg assistant">
							{#if chatStore.thinkingFor(msg.seq)}
								<ThinkingBlock text={chatStore.thinkingFor(msg.seq) ?? ''} />
							{/if}
							<div class="body"><MarkdownMessage content={msg.content} /></div>
							<div class="msg-foot">
								<span class="meta">
									{#if msg.ts}{formatTime(msg.ts)}{/if}
									{#if msg.completion_tokens}
										<span class="num">{msg.completion_tokens}</span> tokens
									{/if}
									{#if msg.finish_reason && msg.finish_reason !== 'stop'}
										<span class="wb-badge warn">{msg.finish_reason === 'length' ? 'Cut off at the token limit' : msg.finish_reason}</span>
									{/if}
								</span>
								<button
									type="button"
									class="foot-btn {speakingSeq === msg.seq ? 'live' : ''}"
									onclick={() => toggleSpeak(msg.content, msg.seq)}
									disabled={!ttsAvailable && speakingSeq !== msg.seq}
									aria-label={speakingSeq === msg.seq ? 'Stop reading aloud' : 'Read aloud'}
									title={!ttsAvailable ? voiceStore.ttsHint : speakingSeq === msg.seq ? 'Stop' : 'Read aloud'}
								>
									<Icon name={speakingSeq === msg.seq ? 'stop' : 'speaker'} size={13} />
								</button>
								<button
									type="button"
									class="foot-btn"
									onclick={() => copyMessage(msg.content, msg.seq)}
									aria-label="Copy reply"
									title="Copy"
								>
									<Icon name={copiedSeq === msg.seq ? 'check' : 'copy'} size={13} />
								</button>
								<button
									type="button"
									class="foot-btn"
									onclick={() => handleRegenerate(msg.seq)}
									disabled={streaming !== null || !ready || !hasModel}
									aria-label="Write a new reply"
									title="Discard this reply and write a new one"
								>
									<Icon name="refresh" size={13} />
								</button>
								<button
									type="button"
									class="foot-btn danger"
									onclick={() => handleDeleteFrom(msg.seq)}
									disabled={streaming !== null}
									aria-label="Delete this reply and everything after it"
									title="Delete from here"
								>
									<Icon name="trash" size={13} />
								</button>
							</div>
						</div>
					{/if}
				{/each}

				{#if streaming}
					<div class="msg assistant" aria-live="polite" aria-busy="true">
						{#if chatStore.thinkingFor(streaming.seq)}
							<ThinkingBlock text={chatStore.thinkingFor(streaming.seq) ?? ''} live={!streaming.partial} />
						{/if}
						{#if streaming.partial}
							<div class="body"><StreamingMarkdown content={streaming.partial} /></div>
						{:else if !chatStore.thinkingFor(streaming.seq)}
							<span class="caret" aria-label="Writing a reply"></span>
						{/if}
					</div>
				{/if}

				{#if error}
					<div class="msg"><p class="wb-banner err">{error}</p></div>
				{/if}
			</div>

			<div class="composer-wrap">
				<div class="composer-col">
					<ChatComposer onsend={handleSend} {modelName} />
				</div>
			</div>
		{/if}

		{#if configOpen}
			<aside class="slideover" aria-label="Chat settings">
				<ChatConfigPanel onclose={() => (configOpen = false)} />
			</aside>
		{/if}
	</section>
</div>

<style>
	.chat {
		display: grid;
		grid-template-columns: 240px minmax(0, 1fr);
		height: 100%;
		overflow: hidden;
	}

	/* Conversation pane */
	.convs {
		display: flex;
		flex-direction: column;
		min-height: 0;
		border-right: 1px solid var(--border);
		background: var(--surface);
	}
	.convs-head {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 10px;
		border-bottom: 1px solid var(--border);
	}
	.convs-list {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		padding: 4px 8px 12px;
	}
	.convs-none {
		padding: 28px 12px;
		text-align: center;
		font-size: var(--text-xs);
		color: var(--muted-foreground);
	}
	.group-l {
		margin: 10px 4px 4px;
		font-size: var(--text-2xs);
		font-weight: 500;
		color: var(--muted-foreground);
	}
	.conv-row {
		position: relative;
		border-radius: 8px;
	}
	.conv-row:hover {
		background: color-mix(in oklch, var(--elevated) 60%, transparent);
	}
	.conv-row.on {
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
	}
	.conv {
		display: flex;
		flex-direction: column;
		gap: 2px;
		width: 100%;
		padding: 8px 30px 8px 10px;
		border: 0;
		border-radius: 8px;
		background: transparent;
		text-align: left;
		cursor: pointer;
		color: var(--foreground);
	}
	.conv.hit {
		padding-right: 10px;
	}
	.conv.hit:hover {
		background: color-mix(in oklch, var(--elevated) 60%, transparent);
	}
	.conv:focus-visible,
	.conv-del:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: -2px;
	}
	.ct {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--text-sm);
		font-weight: 500;
	}
	.cm {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.conv.hit .cm {
		white-space: normal;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		line-height: 1.5;
	}
	.cm :global(mark) {
		background: color-mix(in oklch, var(--primary) 30%, transparent);
		color: var(--foreground);
		border-radius: 2px;
	}
	.conv-del {
		position: absolute;
		top: 8px;
		right: 6px;
		display: grid;
		place-items: center;
		width: 22px;
		height: 22px;
		border: 0;
		border-radius: 6px;
		background: transparent;
		color: var(--muted-foreground);
		opacity: 0;
		cursor: pointer;
		transition: opacity 0.15s ease;
	}
	.conv-row:hover .conv-del,
	.conv-del:focus-visible {
		opacity: 1;
	}
	.conv-del:hover {
		color: var(--error);
		background: color-mix(in oklch, var(--error) 14%, transparent);
	}

	/* Chat column */
	.chat-main {
		position: relative;
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
		overflow: hidden;
	}
	.chat-head {
		display: flex;
		align-items: center;
		gap: 10px;
		height: 52px;
		flex: none;
		padding: 0 16px;
		border-bottom: 1px solid var(--border);
	}
	/* The picker and icon buttons keep their size; the title gives way first. */
	.chat-head > :global(*) {
		flex: none;
	}
	.chat-head > .flex-1 {
		flex: 1 1 0;
	}
	.chat-head > .title {
		flex: 0 1 auto;
		min-width: 0;
	}
	.title {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--text-sm);
		color: var(--muted-foreground);
	}
	.wb-icon-btn.on {
		color: var(--primary);
		background: color-mix(in oklch, var(--primary) 14%, transparent);
	}

	.transcript {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		padding: 24px 24px 168px;
	}
	.msg {
		max-width: 720px;
		margin: 0 auto 22px;
	}
	.msg.user {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
	}
	.bubble {
		width: fit-content;
		max-width: 85%;
		padding: 10px 14px;
		border-radius: 14px 14px 4px 14px;
		border: 1px solid var(--border);
		background: var(--elevated);
		box-shadow: inset 0 1px 0 var(--inset-highlight);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		font-size: var(--text-sm);
		line-height: 1.6;
		color: var(--foreground);
	}
	.body {
		font-size: 14px;
		line-height: 1.65;
	}
	.msg-foot {
		display: flex;
		align-items: center;
		gap: 2px;
		margin-top: 6px;
		min-height: 24px;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}
	.msg-foot.end {
		justify-content: flex-end;
	}
	.meta {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		margin-right: 6px;
	}
	.foot-btn {
		display: grid;
		place-items: center;
		width: 24px;
		height: 24px;
		border: 0;
		border-radius: 6px;
		background: transparent;
		color: var(--muted-foreground);
		cursor: pointer;
		opacity: 0;
		transition:
			opacity 0.15s ease,
			color 0.15s ease,
			background 0.15s ease;
	}
	.msg:hover .foot-btn,
	.foot-btn:focus-visible,
	.foot-btn.live {
		opacity: 1;
	}
	/* The newest reply keeps its actions visible. */
	.msg.assistant:last-of-type .foot-btn {
		opacity: 1;
	}
	.foot-btn:hover:not(:disabled) {
		color: var(--foreground);
		background: var(--muted);
	}
	.foot-btn.danger:hover:not(:disabled) {
		color: var(--error);
		background: color-mix(in oklch, var(--error) 14%, transparent);
	}
	.foot-btn.live {
		color: var(--primary);
	}
	.foot-btn:disabled {
		cursor: not-allowed;
		opacity: 0.35 !important;
	}
	.foot-btn:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: 1px;
	}
	.caret {
		display: inline-block;
		width: 7px;
		height: 15px;
		border-radius: 1px;
		background: var(--primary);
		vertical-align: text-bottom;
		animation: caret-pulse 1s ease-in-out infinite;
	}
	@keyframes caret-pulse {
		50% {
			opacity: 0.25;
		}
	}

	.composer-wrap {
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		padding: 28px 24px 18px;
		background: linear-gradient(to top, var(--background) 62%, transparent);
		pointer-events: none;
	}
	.composer-col {
		max-width: 720px;
		margin: 0 auto;
		pointer-events: auto;
	}

	/* Empty state */
	.empty {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		display: flex;
		justify-content: center;
		padding: 0 24px;
	}
	.empty-inner {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		width: 100%;
		max-width: 720px;
		padding: 48px 0;
	}
	.empty h2 {
		font-size: var(--text-xl);
		font-weight: 600;
		letter-spacing: -0.01em;
		color: var(--foreground);
	}
	.empty p {
		margin-top: 4px;
		font-size: var(--text-sm);
		color: var(--muted-foreground);
	}
	.recents {
		width: 100%;
		margin-top: 32px;
		border-top: 1px solid var(--border);
		padding-top: 4px;
	}
	.recent {
		display: flex;
		align-items: center;
		gap: 10px;
		width: 100%;
		padding: 8px 6px;
		border: 0;
		border-bottom: 1px solid color-mix(in oklch, var(--border) 60%, transparent);
		background: transparent;
		text-align: left;
		cursor: pointer;
		color: var(--foreground);
	}
	.recent:hover {
		background: color-mix(in oklch, var(--elevated) 60%, transparent);
	}
	.recent:focus-visible {
		outline: 2px solid var(--ring);
		outline-offset: -2px;
	}
	.rt {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: var(--text-sm);
	}
	.ra {
		flex: none;
		font-size: var(--text-2xs);
		color: var(--muted-foreground);
	}

	/* Settings slide-over */
	.slideover {
		position: absolute;
		top: 52px;
		right: 0;
		bottom: 0;
		z-index: 20;
		width: 340px;
		border-left: 1px solid var(--border);
		background: var(--surface);
		box-shadow: -18px 0 40px -20px rgb(0 0 0 / 0.45);
		animation: slide-in 0.2s ease-out;
	}
	@keyframes slide-in {
		from {
			transform: translateX(16px);
			opacity: 0;
		}
	}

	/* Wide windows make room for the slide-over instead of covering the transcript. */
	@media (min-width: 1280px) {
		.with-panel .transcript,
		.with-panel .composer-wrap,
		.with-panel .empty {
			padding-right: 364px;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.caret,
		.slideover {
			animation: none;
		}
	}
</style>
