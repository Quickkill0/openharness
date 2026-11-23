import { describe, expect, it } from 'vitest';
import { requestRemix, studio, takeRemix } from '$lib/stores/studio.svelte';
import type { Asset } from '$lib/types';

const asset: Asset = {
	id: 'a1',
	job_id: 'j1',
	kind: 'image',
	path: 'C:/assets/a1.png',
	task: 'text_to_image',
	prompt: 'a lighthouse at dusk',
	model: 'SDXL Turbo',
	params: { prompt: 'a lighthouse at dusk', steps: 4 },
	meta: null,
	size: 1024,
	created_at: 0,
};

describe('library remix hand-off', () => {
	it('only the matching task page can claim the asset, and only once', () => {
		requestRemix('text_to_image', asset);
		expect(studio.remix?.task).toBe('text_to_image');
		expect(takeRemix('text_to_video')).toBeNull();
		expect(takeRemix('text_to_image')).toEqual(asset);
		expect(studio.remix).toBeNull();
		expect(takeRemix('text_to_image')).toBeNull();
	});

	it('a newer request replaces one nobody claimed', () => {
		requestRemix('text_to_image', asset);
		requestRemix('transcribe', { ...asset, id: 'a2', task: 'transcribe' });
		expect(takeRemix('text_to_image')).toBeNull();
		expect(takeRemix('transcribe')?.id).toBe('a2');
	});
});
