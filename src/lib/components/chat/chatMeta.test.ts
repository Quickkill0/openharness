import { describe, expect, it } from 'vitest';
import { composerHint, composerPlaceholder, groupConversations, kTokens, relTime } from './chatMeta';

describe('composerPlaceholder', () => {
	const base = { installed: true, phase: 'ready', hasModel: true, modelName: 'Qwen3 8B' };

	it('asks for the engine first', () => {
		expect(composerPlaceholder({ ...base, installed: false })).toBe('Install the engine to start chatting');
	});
	it('reports install and load phases', () => {
		expect(composerPlaceholder({ ...base, phase: 'installing' })).toBe('Installing the engine…');
		expect(composerPlaceholder({ ...base, phase: 'loading' })).toBe('Loading the model…');
	});
	it('asks for a model when none is loaded', () => {
		expect(composerPlaceholder({ ...base, hasModel: false })).toBe('Load a model to chat');
	});
	it('names the loaded model when known', () => {
		expect(composerPlaceholder(base)).toBe('Message Qwen3 8B');
		expect(composerPlaceholder({ ...base, modelName: null })).toBe('Send a message');
	});
});

describe('composerHint', () => {
	const idle = { recording: false, transcribing: false, voiceError: null };
	it('defaults to the keyboard hint', () => {
		expect(composerHint(idle)).toEqual({ text: 'Enter to send, Shift+Enter for a new line', tone: '' });
	});
	it('shows voice errors above everything else', () => {
		expect(composerHint({ ...idle, recording: true, voiceError: 'Mic blocked' })).toEqual({
			text: 'Mic blocked',
			tone: 'err',
		});
	});
	it('flags recording and transcribing', () => {
		expect(composerHint({ ...idle, recording: true }).tone).toBe('err');
		expect(composerHint({ ...idle, transcribing: true })).toEqual({ text: 'Transcribing…', tone: 'live' });
	});
});

describe('groupConversations', () => {
	const now = new Date(2026, 10, 7, 15, 0, 0);
	const sec = (d: Date) => Math.floor(d.getTime() / 1000);

	it('buckets by local calendar day and drops empty groups', () => {
		const convs = [
			{ id: 'a', updated_at: sec(new Date(2026, 10, 7, 9, 0)) },
			{ id: 'b', updated_at: sec(new Date(2026, 10, 7, 0, 0, 1)) },
			{ id: 'c', updated_at: sec(new Date(2026, 10, 6, 23, 59)) },
			{ id: 'd', updated_at: sec(new Date(2026, 10, 1, 12, 0)) },
		];
		const groups = groupConversations(convs, now);
		expect(groups.map((g) => g.label)).toEqual(['Today', 'Yesterday', 'Earlier']);
		expect(groups[0].items.map((c) => c.id)).toEqual(['a', 'b']);
		expect(groups[1].items.map((c) => c.id)).toEqual(['c']);
		expect(groups[2].items.map((c) => c.id)).toEqual(['d']);
	});

	it('returns no groups for no conversations', () => {
		expect(groupConversations([], now)).toEqual([]);
	});

	it('omits the Yesterday group when nothing falls there', () => {
		const groups = groupConversations([{ id: 'a', updated_at: sec(now) }, { id: 'b', updated_at: 0 }], now);
		expect(groups.map((g) => g.label)).toEqual(['Today', 'Earlier']);
	});
});

describe('relTime', () => {
	const nowMs = 1_800_000_000_000;
	const nowS = nowMs / 1000;
	it('formats recent times in plain words', () => {
		expect(relTime(nowS - 5, nowMs)).toBe('just now');
		expect(relTime(nowS - 5 * 60, nowMs)).toBe('5 min ago');
		expect(relTime(nowS - 3 * 3600, nowMs)).toBe('3 h ago');
		expect(relTime(nowS - 30 * 3600, nowMs)).toBe('yesterday');
		expect(relTime(nowS - 4 * 86400, nowMs)).toBe('4 days ago');
	});
	it('treats future timestamps as just now', () => {
		expect(relTime(nowS + 100, nowMs)).toBe('just now');
	});
});

describe('kTokens', () => {
	it('keeps small counts exact', () => {
		expect(kTokens(0)).toBe('0');
		expect(kTokens(512)).toBe('512');
	});
	it('uses one decimal under 10K and whole numbers above', () => {
		expect(kTokens(6349)).toBe('6.2K');
		expect(kTokens(32768)).toBe('32K');
	});
	it('guards against bad input', () => {
		expect(kTokens(Number.NaN)).toBe('0');
		expect(kTokens(-4)).toBe('0');
	});
});
