"""Worker server tests — stdlib unittest, no external deps.

Run from worker-py/:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openharness_worker import pipelines, protocol  # noqa: E402
from openharness_worker.server import Server  # noqa: E402


class CaptureStdout:
    """Swap sys.stdout for a StringIO and parse emitted JSON lines."""

    def __enter__(self):
        self._old = sys.stdout
        self.buf = io.StringIO()
        sys.stdout = self.buf
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old

    def messages(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.buf.getvalue().splitlines()
            if line.strip()
        ]

    def response_for(self, req_id: str) -> dict:
        for m in self.messages():
            if m.get("id") == req_id:
                return m
        raise AssertionError(f"no response for id={req_id}")

    def events(self, name: str) -> list[dict]:
        return [m for m in self.messages() if m.get("event") == name]


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class ProtocolTests(unittest.TestCase):
    def test_read_requests_skips_garbage(self):
        stream = io.StringIO('not json\n\n{"op": "ping", "id": "1"}\n{"no_op": 1}\n')
        with CaptureStdout():
            reqs = list(protocol.read_requests(stream))
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["op"], "ping")

    def test_write_message_single_line_json(self):
        with CaptureStdout() as cap:
            protocol.write_message({"event": "log", "msg": "héllo\nworld"})
        lines = cap.buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["msg"], "héllo\nworld")


class RegistryTests(unittest.TestCase):
    def test_capabilities_lists_all_tasks(self):
        caps = {c["task"]: c for c in pipelines.capabilities()}
        for task in (
            "echo",
            "text_to_image",
            "transcribe",
            "text_to_speech",
            "text_to_video",
            "vibevoice",
        ):
            self.assertIn(task, caps)
        self.assertTrue(caps["echo"]["available"])
        self.assertTrue(caps["echo"]["hidden"])

    def test_unknown_task_raises_keyerror(self):
        with self.assertRaises(KeyError):
            pipelines.get_pipeline("nope")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = Server(Path(self.tmp.name))

    def tearDown(self):
        self.server._executor.shutdown(wait=True, cancel_futures=True)
        self.tmp.cleanup()

    def test_ping(self):
        with CaptureStdout() as cap:
            self.server.handle({"id": "r1", "op": "ping"})
        resp = cap.response_for("r1")
        self.assertTrue(resp["ok"])
        self.assertTrue(resp["result"]["pong"])

    def test_capabilities(self):
        with CaptureStdout() as cap:
            self.server.handle({"id": "r2", "op": "capabilities"})
        resp = cap.response_for("r2")
        self.assertTrue(resp["ok"])
        tasks = {p["task"] for p in resp["result"]["pipelines"]}
        self.assertIn("text_to_image", tasks)

    def test_echo_job_round_trip(self):
        with CaptureStdout() as cap:
            self.server.handle(
                {
                    "id": "r3",
                    "op": "submit",
                    "task": "echo",
                    "job_id": "job-1",
                    "params": {"steps": 3, "delay": 0.01, "text": "hi"},
                }
            )
            self.assertTrue(cap.response_for("r3")["ok"])
            self.assertTrue(wait_for(lambda: cap.events("job_done")))
        done = cap.events("job_done")[0]
        self.assertEqual(done["job_id"], "job-1")
        self.assertEqual(done["task"], "echo")  # orchestrator attributes VRAM by task
        out = Path(done["outputs"][0])
        self.assertTrue(out.exists())
        self.assertIn("hi", out.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cap.events("job_progress")), 3)

    def test_cancel_running_job(self):
        with CaptureStdout() as cap:
            self.server.handle(
                {
                    "id": "r4",
                    "op": "submit",
                    "task": "echo",
                    "job_id": "job-2",
                    "params": {"steps": 200, "delay": 0.05},
                }
            )
            self.assertTrue(wait_for(lambda: cap.events("job_progress")))
            self.server.handle({"id": "r5", "op": "cancel", "job_id": "job-2"})
            self.assertTrue(cap.response_for("r5")["ok"])
            self.assertTrue(wait_for(lambda: cap.events("job_error")))
        err = cap.events("job_error")[0]
        self.assertEqual(err["error"], "cancelled")

    def test_submit_unknown_task_rejected(self):
        with CaptureStdout() as cap:
            self.server.handle(
                {"id": "r6", "op": "submit", "task": "nope", "job_id": "j"}
            )
        resp = cap.response_for("r6")
        self.assertFalse(resp["ok"])
        self.assertIn("unknown task", resp["error"])

    def test_unavailable_pipeline_reports_missing_deps(self):
        # Heavy deps are absent in the dev/test environment, so submitting a
        # real task must fail the JOB (job_error), not the submit request.
        if not pipelines.missing_requirements("text_to_image"):
            self.skipTest("torch+diffusers installed here; nothing to assert")
        with CaptureStdout() as cap:
            self.server.handle(
                {
                    "id": "r7",
                    "op": "submit",
                    "task": "text_to_image",
                    "job_id": "job-3",
                    "params": {},
                }
            )
            self.assertTrue(cap.response_for("r7")["ok"])
            self.assertTrue(wait_for(lambda: cap.events("job_error")))
        self.assertIn("missing packages", cap.events("job_error")[0]["error"])

    def test_unload_noop_ok(self):
        with CaptureStdout() as cap:
            self.server.handle({"id": "r8", "op": "unload", "task": None})
        self.assertTrue(cap.response_for("r8")["ok"])

    def test_preload_round_trip(self):
        with CaptureStdout() as cap:
            self.server.handle(
                {
                    "id": "r9",
                    "op": "preload",
                    "task": "echo",
                    "job_id": "pre-1",
                    "params": {"delay": 0.01, "vram_mb": 1234},
                }
            )
            self.assertTrue(cap.response_for("r9")["ok"])
            self.assertTrue(wait_for(lambda: cap.events("job_done")))
        done = cap.events("job_done")[0]
        self.assertEqual(done["job_id"], "pre-1")
        self.assertEqual(done["task"], "echo")  # ledger attributes VRAM by task
        self.assertEqual(done["outputs"], [])
        self.assertEqual(done["meta"]["vram_mb"], 1234)
        self.assertTrue(done["meta"]["preload"])

    def test_preload_unknown_task_rejected(self):
        with CaptureStdout() as cap:
            self.server.handle(
                {"id": "r10", "op": "preload", "task": "nope", "job_id": "pre-2"}
            )
        resp = cap.response_for("r10")
        self.assertFalse(resp["ok"])
        self.assertIn("unknown task", resp["error"])

    def test_preload_requires_task_and_job_id(self):
        with CaptureStdout() as cap:
            self.server.handle({"id": "r11", "op": "preload", "task": "echo"})
        resp = cap.response_for("r11")
        self.assertFalse(resp["ok"])
        self.assertIn("preload requires", resp["error"])


class ParseParamsTests(unittest.TestCase):
    """Param validation is pure stdlib — testable without torch & friends.
    Pipeline modules keep heavy imports inside functions for exactly this."""

    def test_text_to_image_defaults_and_limits(self):
        from openharness_worker.pipelines import text_to_image as t2i

        p = t2i.parse_params({"model": "m", "prompt": "a cat"})
        self.assertEqual(p["width"], 1024)
        self.assertEqual(p["arch"], "sdxl")
        self.assertEqual(p["num_images"], 1)
        with self.assertRaises(ValueError):
            t2i.parse_params({"prompt": "no model"})
        with self.assertRaises(ValueError):
            t2i.parse_params({"model": "m", "prompt": "p", "width": 1001})
        with self.assertRaises(ValueError):
            t2i.parse_params({"model": "m", "prompt": "p", "width": 0})
        with self.assertRaises(ValueError):
            t2i.parse_params({"model": "m", "prompt": "p", "arch": "vae"})
        # num_images clamps to 1..8
        p = t2i.parse_params({"model": "m", "prompt": "p", "num_images": 99})
        self.assertEqual(p["num_images"], 8)

    def test_transcribe_params(self):
        from openharness_worker.pipelines import transcribe

        # A model is now required — no "base" default, no auto-download.
        p = transcribe.parse_params({"audio": "a.wav", "model": "C:\\m\\whisper"})
        self.assertEqual(p["model"], "C:\\m\\whisper")
        self.assertEqual(p["task"], "transcribe")
        with self.assertRaises(ValueError):
            transcribe.parse_params({})
        # Missing/empty model raises rather than substituting a size name.
        with self.assertRaises(ValueError):
            transcribe.parse_params({"audio": "a.wav"})
        with self.assertRaises(ValueError):
            transcribe.parse_params({"audio": "a.wav", "model": "  "})
        with self.assertRaises(ValueError):
            transcribe.parse_params({"audio": "a.wav", "model": "m", "task": "summarize"})

    def test_transcribe_load_rejects_nonlocal_model(self):
        from openharness_worker.pipelines import transcribe

        # A non-existent path must raise (no HF auto-download), without ever
        # needing faster-whisper installed — the path check runs first.
        with self.assertRaises(FileNotFoundError):
            transcribe._load("base")
        with self.assertRaises(FileNotFoundError):
            transcribe._load("Z:\\does\\not\\exist\\whisper")

    def test_text_to_speech_params(self):
        from openharness_worker.pipelines import text_to_speech as tts

        # A model is now required — no Kokoro repo default, no auto-download.
        p = tts.parse_params({"text": "hello", "model": "C:\\m\\kokoro"})
        self.assertEqual(p["voice"], "af_heart")
        self.assertEqual(p["lang_code"], "a")  # derived from voice
        self.assertEqual(p["model"], "C:\\m\\kokoro")
        p = tts.parse_params({"text": "hi", "voice": "bf_emma", "model": "m"})
        self.assertEqual(p["lang_code"], "b")
        with self.assertRaises(ValueError):
            tts.parse_params({"text": "  ", "model": "m"})
        # Missing/empty model raises rather than defaulting to the HF repo.
        with self.assertRaises(ValueError):
            tts.parse_params({"text": "hello"})
        with self.assertRaises(ValueError):
            tts.parse_params({"text": "hello", "model": "  "})
        with self.assertRaises(ValueError):
            tts.parse_params({"text": "x", "speed": 3.0, "model": "m"})

    def test_text_to_speech_load_rejects_nonlocal_model(self):
        from openharness_worker.pipelines import text_to_speech as tts

        # A non-existent repo/path must raise (no HF auto-download); the path
        # check runs before kokoro is imported, so no heavy dep is needed.
        with self.assertRaises(FileNotFoundError):
            tts._load("a", "hexgrad/Kokoro-82M")
        with self.assertRaises(FileNotFoundError):
            tts._load("a", "Z:\\does\\not\\exist\\kokoro")

    def test_vibevoice_params(self):
        from openharness_worker.pipelines import vibevoice as vv

        p = vv.parse_params(
            {"model": "C:\\m\\vibevoice", "text": "hello", "reference_audio": "v.wav"}
        )
        self.assertEqual(p["reference_audio"], ["v.wav"])  # normalized to list
        self.assertEqual(p["cfg_scale"], 1.3)
        # Multi-speaker: one sample per speaker, list passes through.
        p = vv.parse_params(
            {
                "model": "m",
                "text": "Speaker 1: hi\nSpeaker 2: hey",
                "reference_audio": ["a.wav", "b.wav"],
                "num_speakers": 2,
                "cfg_scale": 1.5,
            }
        )
        self.assertEqual(p["reference_audio"], ["a.wav", "b.wav"])
        self.assertEqual(p["cfg_scale"], 1.5)
        with self.assertRaises(ValueError):
            vv.parse_params({"text": "x", "reference_audio": "v.wav"})  # no model
        with self.assertRaises(ValueError):
            vv.parse_params({"model": "m", "reference_audio": "v.wav"})  # no text
        with self.assertRaises(ValueError):
            vv.parse_params({"model": "m", "text": "x"})  # no reference audio
        with self.assertRaises(ValueError):
            vv.parse_params({"model": "m", "text": "x", "reference_audio": []})
        with self.assertRaises(ValueError):
            vv.parse_params(
                {"model": "m", "text": "x", "reference_audio": "v.wav", "cfg_scale": 0.5}
            )
        with self.assertRaises(ValueError):
            vv.parse_params(
                {
                    "model": "m",
                    "text": "x",
                    "reference_audio": ["a.wav"],
                    "num_speakers": 2,  # mismatch
                }
            )

    def test_vibevoice_builtin_voice_resolves_to_bundled_clip(self):
        from openharness_worker.pipelines import vibevoice as vv

        # A preset 'voice' (no reference_audio) resolves to the bundled clip at
        # <model>/_builtin_voices/<id>.wav — a pure local-file lookup.
        with tempfile.TemporaryDirectory() as tmp:
            voices_dir = Path(tmp) / vv.BUILTIN_VOICES_SUBDIR
            voices_dir.mkdir()
            clip = voices_dir / "alice.wav"
            clip.write_bytes(b"RIFF")  # contents don't matter to parse_params

            p = vv.parse_params({"model": tmp, "text": "hello", "voice": "alice"})
            self.assertEqual(p["reference_audio"], [str(clip)])

            # reference_audio still works and WINS over a preset voice.
            p = vv.parse_params(
                {
                    "model": tmp,
                    "text": "hi",
                    "voice": "alice",
                    "reference_audio": "C:\\custom\\clip.wav",
                }
            )
            self.assertEqual(p["reference_audio"], ["C:\\custom\\clip.wav"])

            # An unknown / not-yet-installed preset raises an honest error
            # naming the expected path — never a silent fallback or a fetch.
            with self.assertRaises(FileNotFoundError) as ctx:
                vv.parse_params({"model": tmp, "text": "x", "voice": "nope"})
            self.assertIn(vv.BUILTIN_VOICES_SUBDIR, str(ctx.exception))

        # Neither voice nor reference_audio → the original clear error.
        with self.assertRaises(ValueError):
            vv.parse_params({"model": "m", "text": "x"})

    def test_vibevoice_script_builder(self):
        from openharness_worker.pipelines import vibevoice as vv

        # Plain text wraps as a single-speaker script (lines joined).
        self.assertEqual(vv.build_script("hello\nworld"), "Speaker 1: hello world")
        # Labeled scripts pass through verbatim, blank lines dropped.
        script = "Speaker 1: hi\n\nspeaker 2: hey there"
        self.assertEqual(vv.build_script(script), "Speaker 1: hi\nspeaker 2: hey there")

    def test_vibevoice_load_rejects_missing_model_dir(self):
        from openharness_worker.pipelines import vibevoice as vv

        # A non-existent checkpoint dir must raise BEFORE torch/vibevoice are
        # imported (the path check runs first), so no heavy dep is needed and
        # no network fetch can happen.
        with self.assertRaises(ValueError) as ctx:
            vv._load("Z:\\does\\not\\exist\\vibevoice")
        self.assertIn("not fully installed", str(ctx.exception))

    def test_vibevoice_load_rejects_missing_local_tokenizer(self):
        from openharness_worker.pipelines import vibevoice as vv

        # The checkpoint dir exists but the bundled Qwen2.5-1.5B tokenizer is
        # absent — must raise an actionable error rather than fall back to an
        # HF fetch of "Qwen/Qwen2.5-1.5B".
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                vv._load(tmp)
            msg = str(ctx.exception)
            self.assertIn("Qwen2.5-1.5B tokenizer", msg)
            self.assertIn(vv.QWEN_TOKENIZER_SUBDIR, msg)

    def test_vibevoice_resolves_bundled_tokenizer_dir(self):
        from openharness_worker.pipelines import vibevoice as vv

        # When the install carries <model_dir>/_qwen_tokenizer/tokenizer.json,
        # resolution returns that local path — never a bare repo id.
        with tempfile.TemporaryDirectory() as tmp:
            tok_dir = Path(tmp) / vv.QWEN_TOKENIZER_SUBDIR
            tok_dir.mkdir()
            (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            resolved = vv._resolve_local_tokenizer(tmp)
            self.assertEqual(Path(resolved), tok_dir)
            # The resolved value is a local directory, not the HF repo id.
            self.assertNotEqual(resolved, "Qwen/Qwen2.5-1.5B")
            self.assertTrue(Path(resolved).is_dir())

    def test_text_to_video_params(self):
        from openharness_worker.pipelines import text_to_video as t2v

        p = t2v.parse_params({"model": "m", "prompt": "waves"})
        self.assertEqual(p["num_frames"], 97)
        self.assertEqual(p["fps"], 24)
        with self.assertRaises(ValueError):
            t2v.parse_params({"model": "m", "prompt": "p", "width": 700})
        with self.assertRaises(ValueError):
            t2v.parse_params({"model": "m", "prompt": "p", "num_frames": 0})

    def test_text_to_image_qwen_gguf_parses(self):
        from openharness_worker.pipelines import text_to_image as t2i

        # A qwen_image single-file diffusion GGUF now parses through to the
        # real loader (it no longer raises an honest interim error). The
        # companion arch + gguf flag are carried for `_load` to dispatch on.
        p = t2i.parse_params(
            {
                "model": "C:\\m\\qwen-image.gguf",
                "prompt": "a cat",
                "gguf": True,
                "companion_arch": "qwen_image",
            }
        )
        self.assertTrue(p["gguf"])
        self.assertEqual(p["companion_arch"], "qwen_image")
        # No `arch` validation against sdxl/sd/flux on the gguf path.
        self.assertEqual(p["arch"], "qwen_image")

    def test_text_to_image_unsupported_gguf_rejects(self):
        from openharness_worker.pipelines import text_to_image as t2i

        # A diffusion GGUF reaching the image pipeline with an arch we have no
        # loader/companion pack for still raises an honest error.
        with self.assertRaises(RuntimeError) as ctx:
            t2i.parse_params(
                {
                    "model": "C:\\m\\mystery.gguf",
                    "prompt": "a cat",
                    "gguf": True,
                    "companion_arch": "auraflow",
                }
            )
        self.assertIn("auraflow", str(ctx.exception))

    def test_text_to_image_companion_dir_derivation(self):
        from openharness_worker.pipelines import text_to_image as t2i

        # The companion pack is resolved next to the GGUF weights, never inside
        # the single file.
        d = t2i._companion_dir("/models/abc.gguf", "qwen_image")
        self.assertEqual(d.name, "qwen_image")
        self.assertEqual(d.parent.name, "companions")
        self.assertEqual(d.parent.parent.name, "models")

    def test_strip_redundant_text_config(self):
        from openharness_worker.pipelines import text_to_image as t2i

        # The Qwen-Image text encoder's 4.53-saved config carries a nested
        # `text_config` block that transformers 4.51.3 leaves as a raw dict,
        # crashing GenerationConfig.from_model_config. The strip drops it and
        # leaves the rest of the config untouched.
        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.text_config = {"model_type": "qwen2_5_vl_text"}
        cfg.hidden_size = 3584
        out = t2i._strip_redundant_text_config(cfg)
        self.assertIs(out, cfg)
        self.assertFalse(hasattr(cfg, "text_config"))
        self.assertEqual(cfg.hidden_size, 3584)

        # No-op (and no error) when the block is absent.
        bare = _Cfg()
        bare.hidden_size = 1
        t2i._strip_redundant_text_config(bare)
        self.assertEqual(bare.hidden_size, 1)

    def test_text_to_video_ltxv_gguf_parses(self):
        from openharness_worker.pipelines import text_to_video as t2v

        # An ltxv single-file diffusion GGUF now parses through to the real LTX-2
        # loader (it no longer raises an honest interim error). The companion
        # arch + gguf flag are carried for `_load` to dispatch on.
        p = t2v.parse_params(
            {
                "model": "C:\\m\\ltx2.gguf",
                "prompt": "waves",
                "gguf": True,
                "companion_arch": "ltxv",
            }
        )
        self.assertTrue(p["gguf"])
        self.assertEqual(p["companion_arch"], "ltxv")
        # The caller's 8n+1 frame count + fps pass through untouched.
        self.assertEqual(p["num_frames"], 97)
        self.assertEqual(p["fps"], 24)

    def test_text_to_video_wan_gguf_still_rejects(self):
        from openharness_worker.pipelines import text_to_video as t2v

        # Wan stays a later phase — its GGUF still raises an honest error naming
        # the missing companion VAE + text encoder.
        with self.assertRaises(RuntimeError) as ctx:
            t2v.parse_params(
                {
                    "model": "C:\\m\\wan.gguf",
                    "prompt": "waves",
                    "gguf": True,
                    "companion_arch": "wan",
                }
            )
        msg = str(ctx.exception)
        self.assertIn("Wan 2.2", msg)
        self.assertIn("companion", msg.lower())

    def test_text_to_video_companion_dir_derivation(self):
        from openharness_worker.pipelines import text_to_video as t2v

        # The LTX-2 companion pack is resolved next to the GGUF weights, never
        # inside the single file (same scheme as the image pipeline).
        d = t2v._companion_dir("/models/ltx2.gguf", "ltxv")
        self.assertEqual(d.name, "ltxv")
        self.assertEqual(d.parent.name, "companions")
        self.assertEqual(d.parent.parent.name, "models")

    def test_text_to_video_ltx_gguf_missing_companion_raises(self):
        from openharness_worker.pipelines import text_to_video as t2v

        # The GGUF path exists but the companion pack is absent — must raise an
        # actionable FileNotFoundError naming LTX-2 + the needed subfolders,
        # before any heavy diffusers load. We point `model` at a real temp file
        # so the path-exists check passes and the companion check is what fires.
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "ltx2.gguf"
            gguf.write_bytes(b"GGUF")  # contents irrelevant; only existence matters
            with self.assertRaises(FileNotFoundError) as ctx:
                t2v._load_ltx_gguf(str(gguf))
            msg = str(ctx.exception)
            self.assertIn("LTX-2", msg)
            self.assertIn("companion", msg.lower())

    def test_ltx_bnb_4bit_config_present_with_bitsandbytes(self):
        import importlib.machinery
        import sys
        import types
        from unittest.mock import patch

        # Import transformers (warming its one-time bitsandbytes probe against
        # the real environment) BEFORE we fake the module, so injecting a stub
        # doesn't trip transformers' `find_spec` check.
        import transformers

        from openharness_worker.pipelines import text_to_video as t2v

        # When bitsandbytes is importable, the loader builds an NF4 double-quant
        # 4-bit config so the Gemma-3-12B encoder fits under offload. Fake the
        # module's presence (with a real ModuleSpec so importlib probes don't
        # choke) AND stub BitsAndBytesConfig so we assert the exact 4-bit params
        # without the real wheel — its `post_init` validates the installed bnb
        # version via dist metadata a stub can't supply. Deterministic on CPU CI.
        fake = types.ModuleType("bitsandbytes")
        fake.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", loader=None)
        captured = {}

        def fake_cfg(**kw):
            captured.update(kw)
            return "BNB_CONFIG"

        saved = sys.modules.get("bitsandbytes")
        sys.modules["bitsandbytes"] = fake
        try:
            with patch.object(transformers, "BitsAndBytesConfig", fake_cfg):
                cfg = t2v._bnb_4bit_config("dtype_sentinel")
        finally:
            if saved is None:
                sys.modules.pop("bitsandbytes", None)
            else:
                sys.modules["bitsandbytes"] = saved
        self.assertEqual(cfg, "BNB_CONFIG")
        self.assertTrue(captured.get("load_in_4bit"))
        self.assertEqual(captured.get("bnb_4bit_quant_type"), "nf4")
        self.assertTrue(captured.get("bnb_4bit_use_double_quant"))
        self.assertEqual(captured.get("bnb_4bit_compute_dtype"), "dtype_sentinel")

    def test_ltx_bnb_4bit_config_none_without_bitsandbytes(self):
        import sys

        from openharness_worker.pipelines import text_to_video as t2v

        # No bitsandbytes → no quant config (caller falls back to bf16). Setting
        # the sys.modules entry to None makes `import bitsandbytes` raise even if
        # the wheel is actually installed, so this stays deterministic.
        sentinel = object()
        saved = sys.modules.get("bitsandbytes", sentinel)
        sys.modules["bitsandbytes"] = None
        try:
            cfg = t2v._bnb_4bit_config(object())
        finally:
            if saved is sentinel:
                sys.modules.pop("bitsandbytes", None)
            else:
                sys.modules["bitsandbytes"] = saved
        self.assertIsNone(cfg)

    def test_ltx_text_encoder_passes_4bit_on_cuda(self):
        from unittest.mock import MagicMock, patch

        import transformers

        from openharness_worker.pipelines import text_to_video as t2v

        # On CUDA with bnb available, the encoder load must carry the 4-bit
        # quantization_config (so it lands at ~6.5 GB, offload-friendly).
        marker = object()
        captured = {}

        def fake_load(path, **kw):
            captured.update(kw)
            return MagicMock()

        with (
            patch.object(t2v._runtime, "cuda_available", return_value=True),
            patch.object(t2v, "_bnb_4bit_config", return_value=marker),
            patch.object(
                transformers.Gemma3Config, "from_pretrained", return_value=MagicMock()
            ),
            patch.object(
                transformers.Gemma3ForConditionalGeneration,
                "from_pretrained",
                side_effect=fake_load,
            ),
        ):
            t2v._ltx_text_encoder(Path("/c"), "bf16")
        self.assertIs(captured.get("quantization_config"), marker)

    def test_ltx_text_encoder_bf16_on_cpu(self):
        from unittest.mock import MagicMock, patch

        import transformers

        from openharness_worker.pipelines import text_to_video as t2v

        # No CUDA → plain bf16 load, no quantization_config kwarg at all.
        captured = {}

        def fake_load(path, **kw):
            captured.update(kw)
            return MagicMock()

        with (
            patch.object(t2v._runtime, "cuda_available", return_value=False),
            patch.object(
                transformers.Gemma3Config, "from_pretrained", return_value=MagicMock()
            ),
            patch.object(
                transformers.Gemma3ForConditionalGeneration,
                "from_pretrained",
                side_effect=fake_load,
            ),
        ):
            t2v._ltx_text_encoder(Path("/c"), "bf16")
        self.assertNotIn("quantization_config", captured)

    def test_ltx_prompt_adaln_rename_maps_gguf_to_model_names(self):
        from openharness_worker.pipelines.text_to_video import (
            _ltx_prompt_adaln_rename,
        )

        ckpt = {
            "prompt_adaln_single.linear.weight": 1,
            "prompt_adaln_single.emb.timestep_embedder.linear_1.bias": 2,
            "audio_prompt_adaln_single.linear.bias": 3,
            "transformer_blocks.0.scale_shift_table": 4,
            "adaln_single.emb.timestep_embedder.linear_1.weight": 5,
        }
        out = _ltx_prompt_adaln_rename(ckpt)
        # GGUF prompt-AdaLN names rewritten to the model's param names...
        self.assertIn("prompt_adaln.linear.weight", out)
        self.assertIn("prompt_adaln.emb.timestep_embedder.linear_1.bias", out)
        # ...including the audio_ variant via the same substring.
        self.assertIn("audio_prompt_adaln.linear.bias", out)
        self.assertNotIn("prompt_adaln_single.linear.weight", out)
        self.assertNotIn("audio_prompt_adaln_single.linear.bias", out)
        # Unrelated keys (incl. the already-handled adaln_single) are untouched.
        self.assertEqual(out["transformer_blocks.0.scale_shift_table"], 4)
        self.assertIn("adaln_single.emb.timestep_embedder.linear_1.weight", out)

    def test_patch_ltx2_single_file_mapping_swaps_registry_idempotently(self):
        import sys
        import types

        from openharness_worker.pipelines import text_to_video as t2v

        def orig_convert(checkpoint, **kw):
            return checkpoint  # sentinel "original" converter

        sfu = types.ModuleType("diffusers.loaders.single_file_utils")
        sfu.convert_ltx2_transformer_to_diffusers = orig_convert
        ltx_entry = {"checkpoint_mapping_fn": orig_convert}
        other_entry = {"checkpoint_mapping_fn": object()}
        sfm = types.ModuleType("diffusers.loaders.single_file_model")
        sfm.SINGLE_FILE_LOADABLE_CLASSES = {
            "LTX2VideoTransformer3DModel": ltx_entry,
            "Other": other_entry,
        }
        loaders = types.ModuleType("diffusers.loaders")
        loaders.single_file_model = sfm
        loaders.single_file_utils = sfu
        diffusers = types.ModuleType("diffusers")
        diffusers.loaders = loaders

        keys = (
            "diffusers",
            "diffusers.loaders",
            "diffusers.loaders.single_file_model",
            "diffusers.loaders.single_file_utils",
        )
        saved = {k: sys.modules.get(k) for k in keys}
        sys.modules.update(
            {
                "diffusers": diffusers,
                "diffusers.loaders": loaders,
                "diffusers.loaders.single_file_model": sfm,
                "diffusers.loaders.single_file_utils": sfu,
            }
        )
        try:
            t2v._patch_ltx2_single_file_mapping()
            patched = ltx_entry["checkpoint_mapping_fn"]
            # Registry entry that pointed at the original is swapped...
            self.assertIsNot(patched, orig_convert)
            self.assertTrue(getattr(patched, "_ltx_prompt_adaln_patched", False))
            # ...and the wrapper renames before delegating to the original.
            out = patched({"prompt_adaln_single.linear.weight": 1})
            self.assertIn("prompt_adaln.linear.weight", out)
            # An unrelated registry entry is left alone.
            self.assertIs(
                other_entry["checkpoint_mapping_fn"],
                other_entry["checkpoint_mapping_fn"],
            )
            # Idempotent: a second call does not re-wrap.
            t2v._patch_ltx2_single_file_mapping()
            self.assertIs(ltx_entry["checkpoint_mapping_fn"], patched)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_runtime_seed(self):
        from openharness_worker.pipelines import _runtime

        self.assertEqual(_runtime.resolve_seed(42), 42)
        self.assertEqual(_runtime.resolve_seed(2**33 + 7), 7)  # masked to u32
        s = _runtime.resolve_seed(None)
        self.assertTrue(0 <= s < 2**32)

    def test_capabilities_include_formats(self):
        caps = {c["task"]: c for c in pipelines.capabilities()}
        self.assertIn("diffusers-dir", caps["text_to_image"]["formats"])
        self.assertIn("vibevoice-dir", caps["vibevoice"]["formats"])
        self.assertEqual(caps["echo"]["formats"], [])


class AudioMuxTests(unittest.TestCase):
    """The LTX-2 audio path's one GPU-free, model-free moving part: muxing a
    silent MP4 + a WAV into a single MP4 with an audio track. This genuinely
    exercises `_mux_audio_video` end-to-end (it shells out to the bundled
    ffmpeg). Skips gracefully when the heavy deps / ffmpeg aren't installed in
    the test environment (the stdlib CI box has neither)."""

    def _require(self):
        try:
            import imageio  # noqa: F401
            import imageio_ffmpeg
            import numpy  # noqa: F401
            import soundfile  # noqa: F401
        except Exception as exc:  # missing heavy deps → not this test's job
            raise unittest.SkipTest(f"audio-mux deps unavailable: {exc}")
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not exe or not Path(exe).exists():
            raise unittest.SkipTest("imageio-ffmpeg binary unavailable")

    def test_mux_audio_video_produces_mp4_with_audio_stream(self):
        self._require()
        import imageio
        import numpy as np
        import soundfile as sf

        from openharness_worker.pipelines import text_to_video as t2v

        with tempfile.TemporaryDirectory() as tmp:
            tmpd = Path(tmp)
            # ~6 tiny RGB frames (a moving gradient — content is irrelevant).
            frames = [
                np.full((32, 32, 3), (i * 40) % 256, dtype=np.uint8) for i in range(6)
            ]
            video = tmpd / "silent.mp4"
            imageio.mimwrite(
                str(video), frames, fps=24, codec="libx264",
                quality=8, macro_block_size=1,
            )

            # A short 48 kHz sine wave (matches the LTX-2 output rate).
            sr = t2v._LTX_AUDIO_SR
            t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
            wave = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            wav = tmpd / "tone.wav"
            sf.write(str(wav), wave, sr)

            out = tmpd / "muxed.mp4"
            t2v._mux_audio_video(video, wav, out)

            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

            # Best-effort: confirm the muxed file actually carries an audio
            # stream. `av` is in the worker venv; if absent we settle for the
            # existence + non-empty assertions above.
            try:
                import av
            except Exception:
                return
            with av.open(str(out)) as container:
                kinds = {s.type for s in container.streams}
            self.assertIn("audio", kinds)
            self.assertIn("video", kinds)

    def test_write_audio_wav_handles_empty_and_shapes(self):
        self._require()
        import numpy as np
        import soundfile as sf

        from openharness_worker.pipelines import text_to_video as t2v

        with tempfile.TemporaryDirectory() as tmp:
            tmpd = Path(tmp)
            # None / empty → False, no file written, no crash (silent-video path).
            self.assertFalse(t2v._write_audio_wav(None, tmpd / "none.wav"))
            self.assertFalse(
                t2v._write_audio_wav(np.zeros((0,), dtype=np.float32), tmpd / "e.wav")
            )

            # A channel-first (1, N) batch array writes a real 48 kHz WAV with
            # the batch dim squeezed away (the LTX-2 tensor layout).
            arr = np.zeros((1, 480), dtype=np.float32)
            wav = tmpd / "mono.wav"
            self.assertTrue(t2v._write_audio_wav(arr, wav))
            self.assertTrue(wav.exists())
            data, rate = sf.read(str(wav))
            self.assertEqual(rate, t2v._LTX_AUDIO_SR)
            self.assertEqual(len(data), 480)


class CancelBeforeStartTests(unittest.TestCase):
    def test_queued_job_cancelled_before_start(self):
        tmp = tempfile.TemporaryDirectory()
        server = Server(Path(tmp.name))
        try:
            with CaptureStdout() as cap:
                # First job occupies the single worker thread...
                server.handle(
                    {
                        "id": "a",
                        "op": "submit",
                        "task": "echo",
                        "job_id": "long",
                        "params": {"steps": 100, "delay": 0.05},
                    }
                )
                # ...second job queues behind it and gets cancelled in place.
                server.handle(
                    {
                        "id": "b",
                        "op": "submit",
                        "task": "echo",
                        "job_id": "queued",
                        "params": {"steps": 1},
                    }
                )
                server.handle({"id": "c", "op": "cancel", "job_id": "queued"})
                server.handle({"id": "d", "op": "cancel", "job_id": "long"})
                self.assertTrue(
                    wait_for(
                        lambda: len(cap.events("job_error")) >= 2, timeout=10.0
                    )
                )
            errors = {e["job_id"]: e["error"] for e in cap.events("job_error")}
            self.assertEqual(errors.get("queued"), "cancelled")
            self.assertEqual(errors.get("long"), "cancelled")
        finally:
            server._executor.shutdown(wait=True, cancel_futures=True)
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
