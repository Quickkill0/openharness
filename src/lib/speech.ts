/**
 * Built-in speech capability descriptors (M15).
 *
 * Kokoro 82M is the worker's bundled default TTS — it needs no catalog row,
 * so its descriptor lives here rather than coming from the backend resolver.
 * Mirrors `kokoro_speech_caps()` in src-tauri/src/models/capabilities.rs.
 */
import type { SpeechCaps } from '$lib/types';

export const KOKORO_VOICES = [
	{ id: 'af_heart', label: 'Heart — US female' },
	{ id: 'af_bella', label: 'Bella — US female' },
	{ id: 'af_nicole', label: 'Nicole — US female (whisper)' },
	{ id: 'af_sarah', label: 'Sarah — US female' },
	{ id: 'af_sky', label: 'Sky — US female' },
	{ id: 'am_adam', label: 'Adam — US male' },
	{ id: 'am_michael', label: 'Michael — US male' },
	{ id: 'bf_emma', label: 'Emma — UK female' },
	{ id: 'bf_isabella', label: 'Isabella — UK female' },
	{ id: 'bm_george', label: 'George — UK male' },
	{ id: 'bm_lewis', label: 'Lewis — UK male' },
];

export const BUILTIN_KOKORO_CAPS: SpeechCaps = {
	engine: 'kokoro',
	voices: KOKORO_VOICES,
	supports_speed: true,
	supports_instructions: false,
	supports_voice_cloning: false,
	cloning_requires_ref_text: false,
};
