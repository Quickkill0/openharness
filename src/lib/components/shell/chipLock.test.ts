import { describe, expect, it } from 'vitest';
import { chipLock, type ChipLockInputs } from './chipLock';
import { residentSignature } from '$lib/ledgerSignature';
import type { ResidentInfo } from '$lib/types';

const text = (model_id: string | null, mb = 7000): ResidentInfo => ({
	id: 'text',
	kind: 'text_engine',
	label: model_id ?? 'Unknown',
	model_id,
	est_mb: mb,
	actual_mb: null,
});

const idle: ChipLockInputs = {
	textLoadingModelId: null,
	pendingSlots: new Map(),
	chatStreams: new Map(),
	busyTasks: new Set(),
};

describe('chipLock', () => {
	it('leaves an idle resident unlockable', () => {
		expect(chipLock(text('a'), idle)).toBeNull();
	});

	it('marks the old chip as being replaced while another model loads into its slot', () => {
		expect(chipLock(text('a'), { ...idle, textLoadingModelId: 'b' })?.kind).toBe('replacing');
		expect(chipLock(text('a'), { ...idle, pendingSlots: new Map([['text', 'b']]) })?.kind).toBe('replacing');
	});

	it('does not lock the chip of the model that is itself loading', () => {
		expect(chipLock(text('b'), { ...idle, textLoadingModelId: 'b' })).toBeNull();
	});

	it('blocks unloading the chat model while a reply streams from it', () => {
		const lock = chipLock(text('a'), { ...idle, chatStreams: new Map([['conv-1', 'a']]) });
		expect(lock?.kind).toBe('in_use');
		expect(lock?.reason).toMatch(/chat reply/);
	});

	it('keys chat activity on the model: a stream from another model does not lock this one', () => {
		expect(chipLock(text('b'), { ...idle, chatStreams: new Map([['conv-1', 'a']]) })).toBeNull();
		// Unknown model on either side can't be ruled out.
		expect(chipLock(text('b'), { ...idle, chatStreams: new Map([['conv-1', null]]) })?.kind).toBe('in_use');
		expect(chipLock(text(null), { ...idle, chatStreams: new Map([['conv-1', 'a']]) })?.kind).toBe('in_use');
	});

	it('blocks unloading a pipeline or TTS resident while its job runs', () => {
		const image: ResidentInfo = { ...text('img'), id: 'pipeline:text_to_image', kind: 'pipeline' };
		const tts: ResidentInfo = { ...text('voice'), id: 'tts', kind: 'tts_engine' };
		const busy = { ...idle, busyTasks: new Set(['text_to_image', 'text_to_speech']) };
		expect(chipLock(image, busy)?.kind).toBe('in_use');
		expect(chipLock(tts, busy)?.kind).toBe('in_use');
		expect(chipLock(image, idle)).toBeNull();
		// A streaming chat doesn't lock unrelated slots.
		expect(chipLock(image, { ...idle, chatStreams: new Map([['conv-1', 'img']]) })).toBeNull();
	});

	it('locks every speech slot during a speech job, since the engine is not known here', () => {
		const busy = { ...idle, busyTasks: new Set(['text_to_speech']) };
		for (const id of ['tts', 'pipeline:text_to_speech', 'pipeline:qwen_tts_clone', 'pipeline:vibevoice']) {
			expect(chipLock({ ...text('v'), id, kind: 'pipeline' }, busy)?.kind).toBe('in_use');
		}
		expect(chipLock({ ...text('w'), id: 'pipeline:transcribe', kind: 'pipeline' }, busy)).toBeNull();
	});
});

describe('residentSignature', () => {
	it('changes when the text slot swaps to another model', () => {
		expect(residentSignature([text('a')])).not.toBe(residentSignature([text('b')]));
	});

	it('changes when a resident settles to its measured size', () => {
		const est = text('a', 7000);
		expect(residentSignature([est])).not.toBe(residentSignature([{ ...est, actual_mb: 7400 }]));
	});

	it('ignores snapshot order', () => {
		const img: ResidentInfo = { ...text('img'), id: 'pipeline:text_to_image', kind: 'pipeline' };
		expect(residentSignature([text('a'), img])).toBe(residentSignature([img, text('a')]));
	});
});
