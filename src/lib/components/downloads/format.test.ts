import { describe, expect, it } from 'vitest';
import { groupStatus, humanEta, humanSpeed, jobStatusBadge, unfinishedCount } from './format';

describe('humanSpeed', () => {
	it('formats bytes per second', () => {
		expect(humanSpeed(41 * 1024 * 1024)).toBe('41 MB/s');
	});
	it('is empty when stalled or unknown', () => {
		expect(humanSpeed(0)).toBe('');
		expect(humanSpeed(undefined)).toBe('');
		expect(humanSpeed(NaN)).toBe('');
	});
});

describe('humanEta', () => {
	it('uses seconds under a minute', () => {
		expect(humanEta(40)).toBe('40 s left');
		expect(humanEta(0)).toBe('1 s left');
	});
	it('rounds minutes up', () => {
		expect(humanEta(61)).toBe('2 min left');
		expect(humanEta(180)).toBe('3 min left');
	});
	it('uses hours past an hour', () => {
		expect(humanEta(3600)).toBe('1 h left');
		expect(humanEta(3900)).toBe('1 h 5 min left');
	});
	it('is empty when unknown', () => {
		expect(humanEta(undefined)).toBe('');
		expect(humanEta(null)).toBe('');
		expect(humanEta(-1)).toBe('');
	});
});

describe('jobStatusBadge', () => {
	it('hides the badge while running', () => {
		expect(jobStatusBadge('running')).toBeNull();
	});
	it('maps terminal and waiting states to tones', () => {
		expect(jobStatusBadge('done')).toEqual({ label: 'Done', tone: 'ok' });
		expect(jobStatusBadge('failed')).toEqual({ label: 'Failed', tone: 'err' });
		expect(jobStatusBadge('paused')).toEqual({ label: 'Paused', tone: 'warn' });
		expect(jobStatusBadge('queued')?.label).toBe('Queued');
		expect(jobStatusBadge('cancelled')?.label).toBe('Cancelled');
	});
});

describe('groupStatus', () => {
	it('lets a failure win', () => {
		expect(groupStatus(['done', 'running', 'failed'])).toBe('failed');
	});
	it('prefers running over paused and queued', () => {
		expect(groupStatus(['queued', 'paused', 'running'])).toBe('running');
		expect(groupStatus(['queued', 'paused', 'done'])).toBe('paused');
		expect(groupStatus(['queued', 'done'])).toBe('queued');
	});
	it('is done only when every member is done', () => {
		expect(groupStatus(['done', 'done'])).toBe('done');
		expect(groupStatus(['done', 'cancelled'])).toBe('cancelled');
	});
});

describe('unfinishedCount', () => {
	const jobs = (...statuses: string[]) => statuses.map((status) => ({ status }) as { status: 'queued' });

	it('counts paused files as still left', () => {
		// Review repro: 1 running + 2 paused is 3 files left, not 1.
		expect(unfinishedCount(jobs('running', 'paused', 'paused'))).toBe(3);
		expect(unfinishedCount(jobs('paused', 'paused'))).toBe(2);
	});

	it('counts failed files, which can be resumed', () => {
		// Review repro: 1 paused + 1 failed is 2 files left, not 1.
		expect(unfinishedCount(jobs('paused', 'failed'))).toBe(2);
		expect(unfinishedCount(jobs('failed'))).toBe(1);
	});

	it('leaves out finished and cancelled files', () => {
		expect(unfinishedCount(jobs('queued', 'done', 'cancelled'))).toBe(1);
		expect(unfinishedCount([])).toBe(0);
	});
});
