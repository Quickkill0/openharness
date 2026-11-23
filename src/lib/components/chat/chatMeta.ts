/**
 * Pure display rules for the Chat page (Workbench): composer copy, the
 * conversation-list grouping and the context meter. Kept out of the
 * components so the wording and edge cases are unit-tested.
 */

export interface PlaceholderInput {
	installed: boolean;
	phase: string;
	hasModel: boolean;
	modelName: string | null;
}

export function composerPlaceholder({ installed, phase, hasModel, modelName }: PlaceholderInput): string {
	if (!installed) return 'Install the engine to start chatting';
	if (phase === 'installing') return 'Installing the engine…';
	if (phase === 'loading') return 'Loading the model…';
	if (!hasModel) return 'Load a model to chat';
	return modelName ? `Message ${modelName}` : 'Send a message';
}

export interface HintInput {
	recording: boolean;
	transcribing: boolean;
	voiceError: string | null;
}

export function composerHint({ recording, transcribing, voiceError }: HintInput): {
	text: string;
	tone: '' | 'err' | 'live';
} {
	if (voiceError) return { text: voiceError, tone: 'err' };
	if (recording) return { text: 'Listening. Click the mic again to transcribe', tone: 'err' };
	if (transcribing) return { text: 'Transcribing…', tone: 'live' };
	return { text: 'Enter to send, Shift+Enter for a new line', tone: '' };
}

export interface ConvLike {
	id: string;
	updated_at: number;
}

export interface ConvGroup<T> {
	label: string;
	items: T[];
}

/**
 * Split conversations (newest first, `updated_at` in unix seconds) into
 * Today / Yesterday / Earlier by the local calendar day. Empty groups drop.
 */
export function groupConversations<T extends ConvLike>(convs: T[], now: Date = new Date()): ConvGroup<T>[] {
	const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
	const startOfYesterday = startOfToday - 86400;
	const groups: ConvGroup<T>[] = [
		{ label: 'Today', items: [] },
		{ label: 'Yesterday', items: [] },
		{ label: 'Earlier', items: [] },
	];
	for (const c of convs) {
		if (c.updated_at >= startOfToday) groups[0].items.push(c);
		else if (c.updated_at >= startOfYesterday) groups[1].items.push(c);
		else groups[2].items.push(c);
	}
	return groups.filter((g) => g.items.length > 0);
}

/** Unix seconds → "just now" / "5 min ago" / "3 h ago" / "4 days ago". */
export function relTime(ts: number, nowMs: number = Date.now()): string {
	const s = Math.max(0, Math.floor(nowMs / 1000) - ts);
	if (s < 60) return 'just now';
	if (s < 3600) return `${Math.floor(s / 60)} min ago`;
	if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
	const d = Math.floor(s / 86400);
	return d === 1 ? 'yesterday' : `${d} days ago`;
}

/** Tokens → "6.2K" / "512" for the context meter. */
export function kTokens(n: number): string {
	if (!(n > 0)) return '0';
	if (n < 1000) return String(Math.round(n));
	const k = n / 1024;
	return `${k >= 10 ? Math.round(k) : k.toFixed(1)}K`;
}
