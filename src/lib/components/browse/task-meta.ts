/**
 * task-meta — plain-language store task copy (M16 W5).
 *
 * One source of truth for how the store talks about HF `pipeline_tag`s:
 *  - `TASK_TABS` drives the browse tab strip (label + one-line description).
 *  - `taskLabel()` turns an arbitrary pipeline_tag into "what this model does"
 *    copy for rows/badges, falling back to a prettified tag for the long tail.
 */

export interface TaskTabMeta {
	/** Raw HF pipeline_tag, or 'any' for the All tab. */
	value: string;
	label: string;
	/** One-line plain-language description shown next to the tabs. */
	description: string;
}

export const TASK_TABS: TaskTabMeta[] = [
	{ value: 'any', label: 'All', description: 'Every kind of model' },
	{ value: 'text-generation', label: 'Text', description: 'Models you chat with' },
	{ value: 'text-to-image', label: 'Image', description: 'Turn a prompt into pictures' },
	{ value: 'text-to-video', label: 'Video', description: 'Turn a prompt into short clips' },
	{ value: 'text-to-speech', label: 'Speech', description: 'Text to voice' },
	{ value: 'automatic-speech-recognition', label: 'Transcribe', description: 'Audio to text' },
	{
		value: 'feature-extraction',
		label: 'Embeddings',
		description: 'Power search & similarity (advanced)',
	},
];

/** Plain-language "what it does" label for a pipeline_tag. */
export function taskLabel(tag?: string | null): string | null {
	switch (tag) {
		case 'text-generation':
			return 'Chat & writing';
		case 'text-to-image':
			return 'Makes images';
		case 'text-to-video':
			return 'Makes video';
		case 'text-to-speech':
			return 'Text to voice';
		case 'automatic-speech-recognition':
			return 'Audio to text';
		case 'feature-extraction':
		case 'sentence-similarity':
			return 'Embeddings';
		case 'image-text-to-text':
			return 'Vision chat';
		case undefined:
		case null:
			return null;
		default:
			// Long tail: prettify the raw tag ("image-classification" → "Image classification").
			return tag.charAt(0).toUpperCase() + tag.slice(1).replaceAll('-', ' ');
	}
}
