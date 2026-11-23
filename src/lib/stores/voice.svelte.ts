/**
 * Svelte 5 rune-based voice store (M11 §11.3) — chat voice mode.
 *
 * Voice in: MediaRecorder capture → save_voice_capture → transcribe job →
 * transcript text handed to the composer. Voice out: text_to_speech job →
 * <audio> playback of the result. Jobs run headless through
 * studio.submitJobAndWait so they never disturb the Studio task pages.
 */
import { saveVoiceCapture, listLocalModels } from '$lib/ipc/commands';
import { fileSrc } from '$lib/ipc/backend';
import { submitJobAndWait } from '$lib/stores/studio.svelte';
import type { InstalledModel } from '$lib/types';

const SETTINGS_KEY = 'openharness.voice';

export type SpeechEngine = 'kokoro' | 'qwen3-tts';

interface VoiceSettings {
	/** Speak assistant replies aloud automatically. */
	autoSpeak: boolean;
	/** Which TTS engine speaks replies. */
	engine: SpeechEngine;
	/** Voice id — a Kokoro voice or a Qwen3-TTS speaker, depending on engine. */
	voice: string;
	speed: number;
	/**
	 * Catalog id of the Qwen3-TTS talker model (engine === 'qwen3-tts').
	 * For engine === 'kokoro', `ttsModel` carries the installed model path.
	 */
	qwenModel: string;
	/** Installed Whisper/STT model dir for voice input. '' = none configured. */
	sttModel: string;
	/** Installed Kokoro TTS model path for voice output. '' = none configured. */
	ttsModel: string;
}

const DEFAULTS: VoiceSettings = {
	autoSpeak: false,
	engine: 'kokoro',
	voice: 'af_heart',
	speed: 1,
	qwenModel: '',
	sttModel: '',
	ttsModel: '',
};

function loadSettings(): VoiceSettings {
	if (typeof localStorage !== 'undefined') {
		try {
			const raw = localStorage.getItem(SETTINGS_KEY);
			if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
		} catch {
			// fall through to defaults
		}
	}
	return { ...DEFAULTS };
}

let _settings = $state<VoiceSettings>(loadSettings());
let _recording = $state(false);
let _transcribing = $state(false);
let _speakingSeq = $state<number | null>(null);
let _error = $state<string | null>(null);
/** Installed-model availability (auto-download is gone — these gate the UI). */
let _sttAvailable = $state(false);
let _ttsAvailable = $state(false);

let recorder: MediaRecorder | null = null;
let chunks: Blob[] = [];
let player: HTMLAudioElement | null = null;

const STT_HINT = 'Install a transcription model in the Store to use voice input.';
const TTS_HINT = 'Install a speech model in the Store to use voice output.';

export const voiceStore = {
	get autoSpeak() { return _settings.autoSpeak; },
	get engine() { return _settings.engine; },
	get voice() { return _settings.voice; },
	get speed() { return _settings.speed; },
	get qwenModel() { return _settings.qwenModel; },
	/** Installed Kokoro TTS model path (engine === 'kokoro'). */
	get ttsModel() { return _settings.ttsModel; },
	get recording() { return _recording; },
	get transcribing() { return _transcribing; },
	/** Message seq currently being synthesized/played, or null. */
	get speakingSeq() { return _speakingSeq; },
	get error() { return _error; },
	/** True once an installed Whisper/STT model is available for voice input. */
	get sttAvailable() { return _sttAvailable; },
	/** True once an installed TTS model is available for voice output. */
	get ttsAvailable() { return _ttsAvailable; },
	get sttHint() { return STT_HINT; },
	get ttsHint() { return TTS_HINT; },
};

/** Dir-layout models load from local_path; single-blob from the file itself. */
function pathOf(m: InstalledModel): string {
	return m.local_path ?? m.primary_file_path;
}

/**
 * Probe installed models and auto-configure the chat voice STT/TTS picks.
 * Voice features stay disabled (with a store hint) until suitable installed
 * models exist — nothing is ever auto-downloaded. Call from the chat UI on
 * mount; re-runs are cheap and keep availability fresh after installs.
 */
export async function refreshVoiceModels(): Promise<void> {
	try {
		const models = await listLocalModels();
		const stt = models.filter((m) => m.task === 'automatic-speech-recognition');
		// Chat voice out auto-speaks; cloning-only engines (VibeVoice) need a
		// reference sample, so only preset-voice Kokoro / Qwen3-TTS qualify.
		const tts = models.filter((m) => {
			const s = m.caps.speech;
			return (
				s !== null &&
				(s.engine === 'kokoro' || s.engine === 'qwen3-tts') &&
				s.voices.length > 0
			);
		});

		// STT: keep a still-installed configured pick, else auto-pick the first.
		const sttPick =
			stt.find((m) => pathOf(m) === _settings.sttModel) ?? stt[0] ?? null;
		_sttAvailable = sttPick !== null;
		if (sttPick && pathOf(sttPick) !== _settings.sttModel) {
			_settings = { ..._settings, sttModel: pathOf(sttPick) };
			persist();
		} else if (!sttPick && _settings.sttModel) {
			_settings = { ..._settings, sttModel: '' };
			persist();
		}

		// TTS: honor a configured engine/model when it's still installed, else
		// auto-pick the first preset-voice speech model.
		const configured =
			_settings.engine === 'qwen3-tts'
				? tts.find((m) => m.id === _settings.qwenModel)
				: tts.find((m) => pathOf(m) === _settings.ttsModel);
		const ttsPick = configured ?? tts[0] ?? null;
		_ttsAvailable = ttsPick !== null;
		if (ttsPick && ttsPick !== configured) {
			const eng: SpeechEngine =
				ttsPick.caps.speech?.engine === 'qwen3-tts' ? 'qwen3-tts' : 'kokoro';
			_settings = {
				..._settings,
				engine: eng,
				qwenModel: eng === 'qwen3-tts' ? ttsPick.id : '',
				ttsModel: eng === 'qwen3-tts' ? '' : pathOf(ttsPick),
			};
			persist();
		} else if (!ttsPick) {
			// Clear a stale configured ref so speak() honestly refuses.
			if (_settings.qwenModel || _settings.ttsModel) {
				_settings = { ..._settings, qwenModel: '', ttsModel: '' };
				persist();
			}
		}
	} catch {
		// Probe failure leaves availability false — the buttons stay disabled
		// with the install hint rather than risk an auto-download.
		_sttAvailable = false;
		_ttsAvailable = false;
	}
}

function persist(): void {
	if (typeof localStorage !== 'undefined') {
		localStorage.setItem(SETTINGS_KEY, JSON.stringify(_settings));
	}
}

export function setAutoSpeak(on: boolean): void {
	_settings = { ..._settings, autoSpeak: on };
	persist();
	if (!on) stopSpeaking();
}

export function setVoice(voice: string): void {
	_settings = { ..._settings, voice };
	persist();
}

/**
 * Point chat voice output at an engine + voice + installed model. Studio's
 * Speech tab calls this from its "Use for chat replies" action. For Qwen the
 * id is the talker catalog id; for Kokoro it's the installed model path.
 * Either way an installed model is required — there is no built-in fallback.
 */
export function setSpeechVoiceConfig(
	engine: SpeechEngine,
	voice: string,
	modelRef = '',
): void {
	_settings = {
		..._settings,
		engine,
		voice,
		qwenModel: engine === 'qwen3-tts' ? modelRef : '',
		ttsModel: engine === 'qwen3-tts' ? '' : modelRef,
	};
	_ttsAvailable = modelRef.trim().length > 0;
	persist();
}

export function clearVoiceError(): void {
	_error = null;
}

// ---------------------------------------------------------------------------
// Voice in — record + transcribe
// ---------------------------------------------------------------------------

export async function startRecording(): Promise<void> {
	if (_recording) return;
	_error = null;
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
		chunks = [];
		recorder = new MediaRecorder(stream, pickMime());
		recorder.ondataavailable = (e) => {
			if (e.data.size > 0) chunks.push(e.data);
		};
		recorder.start();
		_recording = true;
	} catch (e) {
		_error = `Microphone unavailable: ${e instanceof Error ? e.message : e}`;
	}
}

function pickMime(): MediaRecorderOptions {
	for (const mimeType of ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']) {
		if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(mimeType)) {
			return { mimeType };
		}
	}
	return {};
}

/**
 * Stop recording, transcribe the capture, and return the transcript text
 * (null on failure — the error lands in voiceStore.error).
 */
export async function stopAndTranscribe(): Promise<string | null> {
	const rec = recorder;
	if (!rec || !_recording) return null;
	_recording = false;

	// No auto-download: voice input needs an explicitly-installed STT model.
	if (!_settings.sttModel) {
		try {
			rec.stop();
			rec.stream.getTracks().forEach((t) => t.stop());
		} catch {
			// already stopped
		}
		recorder = null;
		chunks = [];
		_error = STT_HINT;
		return null;
	}

	const blob = await new Promise<Blob>((resolve) => {
		rec.onstop = () => resolve(new Blob(chunks, { type: rec.mimeType || 'audio/webm' }));
		rec.stop();
		rec.stream.getTracks().forEach((t) => t.stop());
	});
	recorder = null;
	chunks = [];

	if (blob.size < 1000) {
		_error = 'Recording too short — hold the mic button a little longer.';
		return null;
	}

	_transcribing = true;
	try {
		const b64 = await blobToBase64(blob);
		const ext = blob.type.includes('mp4') ? 'm4a' : 'webm';
		const path = await saveVoiceCapture(b64, ext);
		const result = await submitJobAndWait('transcribe', {
			audio: path,
			model: _settings.sttModel,
		});
		const txt = result.outputs.find((o) => o.toLowerCase().endsWith('.txt'));
		if (!txt) throw new Error('transcription produced no text output');
		const src = fileSrc(txt);
		if (!src) throw new Error('transcript file is not readable');
		const res = await fetch(src);
		if (!res.ok) throw new Error(`could not read transcript (HTTP ${res.status})`);
		const text = (await res.text()).trim();
		if (!text) {
			_error = 'No speech detected in the recording.';
			return null;
		}
		return text;
	} catch (e) {
		_error = `Transcription failed: ${e instanceof Error ? e.message : e}`;
		return null;
	} finally {
		_transcribing = false;
	}
}

/** Abandon an in-progress recording without transcribing. */
export function cancelRecording(): void {
	if (recorder) {
		try {
			recorder.stop();
			recorder.stream.getTracks().forEach((t) => t.stop());
		} catch {
			// already stopped
		}
	}
	recorder = null;
	chunks = [];
	_recording = false;
}

async function blobToBase64(blob: Blob): Promise<string> {
	const buf = await blob.arrayBuffer();
	const bytes = new Uint8Array(buf);
	let bin = '';
	const CHUNK = 0x8000;
	for (let i = 0; i < bytes.length; i += CHUNK) {
		bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
	}
	return btoa(bin);
}

// ---------------------------------------------------------------------------
// Voice out — synthesize + play
// ---------------------------------------------------------------------------

/**
 * Synthesize `text` with the configured voice and play it. `seq` tags which
 * message is speaking so the UI can highlight it. Strips markdown noise first.
 */
export async function speak(text: string, seq: number): Promise<void> {
	stopSpeaking();
	const clean = stripForSpeech(text);
	if (!clean) return;

	// No auto-download: voice output needs an explicitly-installed TTS model
	// (a Qwen3-TTS talker or a Kokoro model path). Degrade gracefully.
	const hasTts =
		_settings.engine === 'qwen3-tts' ? !!_settings.qwenModel : !!_settings.ttsModel;
	if (!hasTts) {
		_error = TTS_HINT;
		return;
	}

	_error = null;
	_speakingSeq = seq;
	try {
		const params: Record<string, unknown> =
			_settings.engine === 'qwen3-tts'
				? { text: clean, voice: _settings.voice, model: _settings.qwenModel, engine: 'qwen3-tts' }
				: { text: clean, voice: _settings.voice, speed: _settings.speed, model: _settings.ttsModel };
		const result = await submitJobAndWait('text_to_speech', params);
		// User may have hit stop while the job ran.
		if (_speakingSeq !== seq) return;
		const out = result.outputs[0];
		const src = out ? fileSrc(out) : '';
		if (!src) {
			_speakingSeq = null;
			return; // browser mock — no playable audio
		}
		player = new Audio(src);
		player.onended = () => {
			if (_speakingSeq === seq) _speakingSeq = null;
		};
		player.onerror = () => {
			if (_speakingSeq === seq) _speakingSeq = null;
		};
		await player.play();
	} catch (e) {
		_error = `Speech failed: ${e instanceof Error ? e.message : e}`;
		if (_speakingSeq === seq) _speakingSeq = null;
	}
}

export function stopSpeaking(): void {
	if (player) {
		player.pause();
		player.src = '';
		player = null;
	}
	_speakingSeq = null;
}

/** TTS reads prose, not markup — drop code blocks and markdown decorations. */
function stripForSpeech(md: string): string {
	return md
		.replace(/```[\s\S]*?```/g, ' (code omitted) ')
		.replace(/`([^`]+)`/g, '$1')
		.replace(/!\[[^\]]*\]\([^)]*\)/g, '')
		.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
		.replace(/^#{1,6}\s+/gm, '')
		.replace(/[*_~]{1,3}([^*_~]+)[*_~]{1,3}/g, '$1')
		.replace(/^\s*[-+*]\s+/gm, '')
		.replace(/\s+/g, ' ')
		.trim()
		.slice(0, 4000);
}
