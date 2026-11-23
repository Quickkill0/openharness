/**
 * Work in flight on loaded models, tracked from backend events rather than
 * from what the UI happens to show. The unload guards read this so a model
 * that is still generating can't be unloaded out from under its request.
 *
 * Pure reducers; the chat and Studio stores own the state and feed events in.
 */
import type { ChatEvent, GenerationEvent, GenerationTask } from '$lib/types';

/**
 * Conversation id → the text model its reply is streaming from (null when
 * the model wasn't known at the time). One entry per conversation; the chat
 * actor streams one reply per conversation at a time.
 */
export type ChatStreams = ReadonlyMap<string, string | null>;

/**
 * Track a conversation from the moment the backend stores its user message
 * until the reply's real end. Every turn ends in `assistant_done` or `error`
 * (the chat actor emits one on every path). `truncated` only arrives between
 * turns, so it clears too. Selecting another conversation never touches this.
 */
export function trackChatStream(prev: ChatStreams, ev: ChatEvent, textModel: string | null): ChatStreams {
	switch (ev.type) {
		case 'user_message_stored':
		case 'assistant_start': {
			// Keep the model recorded at the start; only fill in a missing one.
			if (prev.has(ev.conv_id) && (prev.get(ev.conv_id) != null || textModel === null)) return prev;
			const next = new Map(prev);
			next.set(ev.conv_id, textModel);
			return next;
		}
		case 'assistant_done':
		case 'error':
		case 'truncated': {
			if (!prev.has(ev.conv_id)) return prev;
			const next = new Map(prev);
			next.delete(ev.conv_id);
			return next;
		}
		default:
			return prev;
	}
}

/**
 * Headless generation jobs (chat voice: transcription and spoken replies) by
 * job id, or by a `pending:` token while the submit call is still out. They
 * never appear in the Studio task pages' state, so they are tracked here.
 */
export type HeadlessJobs = ReadonlyMap<string, GenerationTask>;

/** Drop a headless job once its real `job_done` or `job_error` arrives. */
export function settleHeadlessJob(prev: HeadlessJobs, ev: GenerationEvent): HeadlessJobs {
	if (ev.event !== 'job_done' && ev.event !== 'job_error') return prev;
	if (!prev.has(ev.job_id)) return prev;
	const next = new Map(prev);
	next.delete(ev.job_id);
	return next;
}

export function withJob(prev: HeadlessJobs, id: string, task: GenerationTask): HeadlessJobs {
	const next = new Map(prev);
	next.set(id, task);
	return next;
}

export function withoutJob(prev: HeadlessJobs, id: string): HeadlessJobs {
	if (!prev.has(id)) return prev;
	const next = new Map(prev);
	next.delete(id);
	return next;
}

/**
 * Resident slots a running task may be using. Speech can run on the Qwen3-TTS
 * sidecar, the Kokoro pipeline, the Qwen voice-clone CLI or VibeVoice, so a
 * speech job conservatively locks all of them.
 */
export function slotsForTask(task: string): string[] {
	if (task === 'text_to_speech') {
		return ['tts', 'pipeline:text_to_speech', 'pipeline:qwen_tts_clone', 'pipeline:vibevoice'];
	}
	return [`pipeline:${task}`];
}
