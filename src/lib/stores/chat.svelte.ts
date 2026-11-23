/**
 * Svelte 5 rune-based chat store.
 *
 * Owns all mutable chat state: conversation list, active conversation messages,
 * streaming partial, search hits. Call initChatStore() once from the root
 * layout onMount. All event writes from the WS/Channel handler flow through
 * handleEvent(); no other path touches _messages directly during streaming.
 */
import {
	listConversations as fetchList,
	createConversation as createConv,
	getConversation as fetchConv,
	deleteConversation as deleteConv,
	sendMessage as send,
	cancelStream as cancel,
	chatTruncateFrom,
	chatRegenerate,
	searchMessages as searchMsgs,
	getConversationSettings,
	setConversationSettings,
} from '$lib/ipc/commands';
import { subscribeChatEvents } from '$lib/ipc/channels';
import { trackChatStream, type ChatStreams } from '$lib/activity';
import { engineStore } from '$lib/stores/engine.svelte';
import type {
	ConversationSummaryDto,
	MessageDto,
	SearchHitDto,
	ChatEvent,
	ConversationSettings,
} from '$lib/types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _conversations = $state<ConversationSummaryDto[]>([]);
let _activeId = $state<string | null>(null);
let _messages = $state<MessageDto[]>([]);
let _streaming = $state<{ seq: number; partial: string } | null>(null);
/**
 * Replies in flight in ANY conversation, keyed by conversation, until each
 * one's real end event. `_streaming` is only what the open conversation
 * shows and resets on selection; the unload guards read this instead.
 */
let _inflight = $state.raw<ChatStreams>(new Map());
let _searchHits = $state<SearchHitDto[]>([]);
let _searchQuery = $state<string>('');
let _error = $state<string | null>(null);
let _initialized = $state(false);
let _settings = $state<ConversationSettings | null>(null);
/**
 * Reasoning-model thinking, keyed by assistant seq. Session-scoped: thinking
 * never enters the persisted assistant message body (it streams via the
 * `thinking` ChatEvent), so the collapsible block survives only until the
 * conversation is reselected.
 */
let _thinking = $state<Record<number, string>>({});

/** Called with each finished assistant reply (voice auto-speak hook, M11). */
let assistantDoneListener: ((content: string, seq: number) => void) | null = null;

// ---------------------------------------------------------------------------
// Read-only store object
// ---------------------------------------------------------------------------

export const chatStore = {
	get conversations() { return _conversations; },
	get activeId() { return _activeId; },
	get messages() { return _messages; },
	get streaming() { return _streaming; },
	/** Conversation id → text model, for every reply still generating. */
	get inflight() { return _inflight; },
	get searchHits() { return _searchHits; },
	get searchQuery() { return _searchQuery; },
	get error() { return _error; },
	get settings() { return _settings; },
	/** Thinking text for an assistant turn (live or finished), if any. */
	thinkingFor(seq: number): string | null { return _thinking[seq] ?? null; },
	get activeConversation() {
		return _conversations.find(c => c.id === _activeId) ?? null;
	},
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

export async function initChatStore(): Promise<void> {
	if (_initialized) return;
	try {
		await subscribeChatEvents(handleEvent);
		await loadConversations();
		_initialized = true;
	} catch (e) {
		_error = String(e);
		console.error('chat store init:', e);
	}
}

// ---------------------------------------------------------------------------
// Internal event handler (single writer for streaming state)
// ---------------------------------------------------------------------------

function handleEvent(event: ChatEvent) {
	// Activity first, for every conversation, before the selection filter.
	const status = engineStore.status;
	_inflight = trackChatStream(_inflight, event, status.phase === 'ready' ? status.model_id : null);
	if (event.conv_id !== _activeId) {
		// Event for an inactive conversation; just refresh sidebar metadata.
		if (event.type === 'assistant_done') {
			loadConversations();
		}
		return;
	}
	switch (event.type) {
		case 'user_message_stored':
			// Optimistic insert already happened in sendMessage; no-op here.
			break;
		case 'assistant_start':
			_streaming = { seq: event.seq, partial: '' };
			break;
		case 'token':
			if (_streaming && _streaming.seq === event.seq) {
				_streaming = { seq: event.seq, partial: _streaming.partial + event.delta };
			}
			break;
		case 'thinking':
			_thinking = { ..._thinking, [event.seq]: (_thinking[event.seq] ?? '') + event.delta };
			break;
		case 'assistant_done': {
			const finalContent = _streaming?.partial ?? '';
			_messages = [..._messages, {
				seq: event.seq,
				role: 'assistant',
				content: finalContent,
				ts: Math.floor(Date.now() / 1000),
				prompt_tokens: event.prompt_tokens,
				completion_tokens: event.completion_tokens,
				finish_reason: event.finish_reason,
			}];
			_streaming = null;
			loadConversations(); // title may have updated
			assistantDoneListener?.(finalContent, event.seq);
			break;
		}
		case 'truncated': {
			_messages = _messages.filter((m) => m.seq < event.from_seq);
			_streaming = null;
			const pruned: Record<number, string> = {};
			for (const [k, v] of Object.entries(_thinking)) {
				if (Number(k) < event.from_seq) pruned[Number(k)] = v;
			}
			_thinking = pruned;
			break;
		}
		case 'error':
			_error = event.error;
			_streaming = null;
			break;
	}
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export async function loadConversations(): Promise<void> {
	try {
		_conversations = await fetchList();
	} catch (e) {
		_error = String(e);
	}
}

export async function createConversation(): Promise<string | null> {
	_error = null;
	try {
		const conv = await createConv();
		_conversations = [conv, ..._conversations];
		await selectConversation(conv.id);
		return conv.id;
	} catch (e) {
		_error = String(e);
		return null;
	}
}

export async function selectConversation(id: string): Promise<void> {
	_error = null;
	_activeId = id;
	_streaming = null;
	_settings = null;
	_thinking = {};
	try {
		_messages = await fetchConv(id);
	} catch (e) {
		_error = String(e);
		_messages = [];
	}
	try {
		_settings = await getConversationSettings(id);
	} catch (e) {
		console.error('load conversation settings:', e);
	}
}

/** Persist the active conversation's settings (system prompt, sampling, load config). */
export async function saveSettings(settings: ConversationSettings): Promise<void> {
	if (!_activeId) return;
	_settings = settings; // optimistic
	try {
		await setConversationSettings(_activeId, settings);
	} catch (e) {
		_error = String(e);
	}
}

export async function sendMessage(content: string): Promise<void> {
	if (!_activeId) return;
	_error = null;
	const userMsg: MessageDto = {
		seq: _messages.length === 0 ? 0 : _messages[_messages.length - 1].seq + 1,
		role: 'user',
		content,
		ts: Math.floor(Date.now() / 1000),
	};
	_messages = [..._messages, userMsg]; // optimistic
	try {
		await send(_activeId, content);
	} catch (e) {
		_error = String(e);
		// Roll back optimistic insert
		_messages = _messages.slice(0, -1);
	}
}

export async function cancelStream(): Promise<void> {
	if (!_activeId) return;
	try { await cancel(_activeId); }
	catch (e) { _error = String(e); }
}

/** Delete a message and everything after it in the active conversation. */
export async function deleteFromHere(seq: number): Promise<void> {
	if (!_activeId) return;
	_error = null;
	try {
		await chatTruncateFrom(_activeId, seq);
		// The `truncated` event prunes _messages; local prune covers the mock path too.
		_messages = _messages.filter((m) => m.seq < seq);
	} catch (e) {
		_error = String(e);
	}
}

/** Regenerate an assistant message: drop it (and anything after) and re-stream. */
export async function regenerate(seq: number): Promise<void> {
	if (!_activeId) return;
	_error = null;
	try {
		await chatRegenerate(_activeId, seq);
	} catch (e) {
		_error = String(e);
	}
}

export async function deleteConversation(id: string): Promise<void> {
	try {
		await deleteConv(id);
		_conversations = _conversations.filter(c => c.id !== id);
		if (_activeId === id) {
			_activeId = null;
			_messages = [];
			_streaming = null;
		}
	} catch (e) {
		_error = String(e);
	}
}

export async function searchMessages(query: string): Promise<void> {
	_searchQuery = query;
	if (!query.trim()) {
		_searchHits = [];
		return;
	}
	try {
		_searchHits = await searchMsgs(query);
	} catch (e) {
		_error = String(e);
		_searchHits = [];
	}
}

export function clearSearch(): void {
	_searchQuery = '';
	_searchHits = [];
}

export function clearError(): void {
	_error = null;
}

/**
 * Register the voice auto-speak hook — called with each finished assistant
 * reply in the active conversation. Pass null to unregister.
 */
export function setAssistantDoneListener(
	cb: ((content: string, seq: number) => void) | null,
): void {
	assistantDoneListener = cb;
}
