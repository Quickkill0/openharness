import { describe, expect, it } from 'vitest';
import { compactCount, formatBytes, gbFromMb, pct, timeAgo } from './format';

describe('gbFromMb', () => {
	it('renders one decimal gigabytes', () => {
		expect(gbFromMb(24576)).toBe('24.0');
		expect(gbFromMb(9011)).toBe('8.8');
	});
	it('keeps tiny non-zero values visible', () => {
		expect(gbFromMb(40)).toBe('0.04');
		expect(gbFromMb(0)).toBe('0.0');
	});
	it('never renders negatives', () => {
		expect(gbFromMb(-300)).toBe('0.0');
	});
});

describe('formatBytes', () => {
	it('picks GB, MB or KB', () => {
		expect(formatBytes(4.9 * 1024 ** 3)).toBe('4.9 GB');
		expect(formatBytes(820 * 1024 ** 2)).toBe('820 MB');
		expect(formatBytes(12 * 1024)).toBe('12 KB');
	});
	it('shows a dash for unknown sizes', () => {
		expect(formatBytes(null)).toBe('—');
		expect(formatBytes(0)).toBe('—');
	});
});

describe('pct', () => {
	it('clamps to 0–100', () => {
		expect(pct(5, 10)).toBe('50.00%');
		expect(pct(20, 10)).toBe('100.00%');
		expect(pct(1, 0)).toBe('0%');
	});
});

describe('compactCount', () => {
	it('shortens thousands and millions without a trailing .0', () => {
		expect(compactCount(412)).toBe('412');
		expect(compactCount(1_300)).toBe('1.3K');
		expect(compactCount(2_000)).toBe('2K');
		expect(compactCount(2_400_000)).toBe('2.4M');
	});
	it('dashes missing or invalid values', () => {
		expect(compactCount(undefined)).toBe('—');
		expect(compactCount(Number.NaN)).toBe('—');
	});
});

describe('timeAgo', () => {
	const now = Date.parse('2026-11-08T12:00:00Z');
	it('uses plain relative wording', () => {
		expect(timeAgo('2026-11-08T08:00:00Z', now)).toBe('today');
		expect(timeAgo('2026-11-07T08:00:00Z', now)).toBe('yesterday');
		expect(timeAgo('2026-11-05T12:00:00Z', now)).toBe('3 days ago');
		expect(timeAgo('2026-10-25T12:00:00Z', now)).toBe('2 weeks ago');
		expect(timeAgo('2026-09-01T12:00:00Z', now)).toBe('2 months ago');
		expect(timeAgo('2024-10-01T12:00:00Z', now)).toBe('2 years ago');
	});
	it('returns null for missing or bad input', () => {
		expect(timeAgo(undefined, now)).toBeNull();
		expect(timeAgo('not a date', now)).toBeNull();
	});
});
