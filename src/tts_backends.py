"""
TTS Backend Registry
====================
Modular adapter layer for text-to-speech engines, mirroring the design of
`avatar_backends.py`.

Each backend implements three operations:
  * initialize(...)        — load models, prepare voice-clone state from ref audio
  * synthesize(text, out)  — produce one .wav file for one segment of script
  * cleanup()              — release GPU memory

Adding a new TTS engine:
  1. Subclass TTSBackend, implement the three methods.
  2. Register an instance in TTS_BACKEND_SPECS below.
  3. Done. The UI dropdown picks it up automatically; no pipeline edits required.

Why an ABC instead of a flag-spec dataclass (like avatar_backends.py)?
  Avatar backends are subprocess-based — they obey a uniform CLI contract
  (image, audio, output flags). TTS engines are in-process Python APIs with
  meaningfully different lifecycles (single-call vs two-step voice conversion,
  different state requirements). An ABC fits that better than a flag map.
"""
from __future__ import annotations
import gc
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Base contract
# ──────────────────────────────────────────────────────────────────────────────
class TTSBackend(ABC):
    """Contract every TTS engine must implement."""

    #: Short registry key (lowercase, no spaces). Used as the lookup id.
    key: str = ""
    #: User-facing label shown in the Gradio dropdown.
    name: str = ""
    #: One-line description shown in the UI.
    description: str = ""
    #: Whether this backend supports zero-shot voice cloning from a reference clip.
    supports_voice_clone: bool = False

    @abstractmethod
    def initialize(self, *, device: str, ref_audio: str, melotts_ckpt_root: str = "checkpoints_v2") -> None:
        """Load models and prepare any voice-clone state. Called once per render run."""

    @abstractmethod
    def synthesize(self, *, text: str, output_path: str, speed: float = 0.85,
                   temp_dir: Optional[str] = None) -> None:
        """Produce a .wav at `output_path` containing `text` in the cloned voice."""

    @abstractmethod
    def cleanup(self) -> None:
        """Drop references and free GPU memory."""


# ──────────────────────────────────────────────────────────────────────────────
# Backend 1 — MeloTTS (base) + OpenVoice V2 (tone-color voice cloning)
# ──────────────────────────────────────────────────────────────────────────────
class MeloOpenVoiceBackend(TTSBackend):
    key = "melotts-openvoice"
    name = "MeloTTS + OpenVoice (Default)"
    description = "Two-step: MeloTTS base synthesis → OpenVoice V2 tone-color voice clone."
    supports_voice_clone = True

    def __init__(self) -> None:
        self._device: Optional[str] = None
        self._base_tts = None
        self._converter = None
        self._source_se = None
        self._target_se = None
        self._default_speaker_id = None

    def initialize(self, *, device: str, ref_audio: str, melotts_ckpt_root: str = "checkpoints_v2") -> None:
        # Lazy imports keep the module loadable without TTS deps installed.
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS  # OpenVoice V2's official base.
        import torch

        self._device = device

        # Base speaker
        self._base_tts = TTS(language='EN', device=device)
        speaker_ids = self._base_tts.hps.data.spk2id
        self._default_speaker_id = (
            speaker_ids['EN-Default']
            if 'EN-Default' in speaker_ids
            else list(speaker_ids.values())[0]
        )

        # Tone-color converter checkpoint
        ckpt_root = (melotts_ckpt_root or "checkpoints_v2").rstrip("/\\ ")
        ckpt_converter = f"{ckpt_root}/converter"
        if not os.path.exists(ckpt_converter):
            raise FileNotFoundError(
                f"Could not find '{ckpt_converter}'. "
                "Ensure the OpenVoice V2 checkpoints are extracted in your project root."
            )
        self._converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
        self._converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

        # Source speaker embedding (the base voice MeloTTS produces)
        self._source_se = torch.load(
            f"{ckpt_root}/base_speakers/ses/en-default.pth",
            map_location=device,
        )

        # Target speaker embedding (the cloned voice from the user-uploaded ref audio)
        self._target_se, _ = se_extractor.get_se(ref_audio, self._converter, vad=True)

    def synthesize(self, *, text: str, output_path: str, speed: float = 0.85,
                   temp_dir: Optional[str] = None) -> None:
        if self._base_tts is None or self._converter is None:
            raise RuntimeError("MeloOpenVoiceBackend.initialize(...) must be called first.")

        # Two-step: base TTS → voice conversion
        out_dir = temp_dir or os.path.dirname(output_path) or "."
        temp_base = os.path.join(out_dir, f"_melo_temp_{os.path.basename(output_path)}")
        try:
            self._base_tts.tts_to_file(
                text=text,
                speaker_id=self._default_speaker_id,
                output_path=temp_base,
                speed=speed,
            )
            self._converter.convert(
                audio_src_path=temp_base,
                src_se=self._source_se,
                tgt_se=self._target_se,
                output_path=output_path,
                message="@MyShell",
            )
        finally:
            if os.path.exists(temp_base):
                try:
                    os.remove(temp_base)
                except OSError:
                    pass

    def cleanup(self) -> None:
        self._base_tts = None
        self._converter = None
        self._source_se = None
        self._target_se = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Backend 2 — F5-TTS (native zero-shot voice cloning, single call)
# ──────────────────────────────────────────────────────────────────────────────
class F5TTSBackend(TTSBackend):
    key = "f5-tts"
    name = "F5-TTS (Voice Cloning)"
    description = "Single-call zero-shot voice cloning. Auto-transcribes ref audio via built-in ASR."
    supports_voice_clone = True

    def __init__(self) -> None:
        self._device: Optional[str] = None
        self._model = None
        self._ref_audio: Optional[str] = None

    def initialize(self, *, device: str, ref_audio: str, melotts_ckpt_root: str = "checkpoints_v2") -> None:
        from f5_tts.api import F5TTS
        # Patch F5-TTS to use sequential (not parallel) batch inference.
        # Its ThreadPoolExecutor fires concurrent model.sample() calls on shared
        # CUDA tensors, causing "Sizes of tensors must match" crashes.
        import f5_tts.infer.utils_infer as _f5u
        from concurrent.futures import ThreadPoolExecutor as _OrigTPE

        class _SeqTPE(_OrigTPE):
            def __init__(self, *a, **kw):
                kw["max_workers"] = 1
                super().__init__(*a, **kw)
        _f5u.ThreadPoolExecutor = _SeqTPE

        self._device = device
        self._model = F5TTS(device=device)
        self._ref_audio = ref_audio

    def synthesize(self, *, text: str, output_path: str, speed: float = 0.85,
                   temp_dir: Optional[str] = None) -> None:
        if self._model is None:
            raise RuntimeError("F5TTSBackend.initialize(...) must be called first.")

        self._model.infer(
            ref_file=self._ref_audio,
            ref_text="",          # empty → auto-transcribe via built-in ASR
            gen_text=text,
            file_wave=output_path,
            speed=speed,
        )

        # Sanitize PYTHONHASHSEED — F5-TTS's seed_everything sometimes sets it
        # > 4294967295, which crashes downstream subprocesses on Windows.
        _hs = os.environ.get("PYTHONHASHSEED", "")
        if _hs and _hs != "random":
            try:
                if int(_hs) > 4294967295:
                    os.environ["PYTHONHASHSEED"] = str(int(_hs) % 4294967296)
            except ValueError:
                os.environ.pop("PYTHONHASHSEED", None)

    def cleanup(self) -> None:
        self._model = None
        self._ref_audio = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────
TTS_BACKEND_SPECS: Dict[str, TTSBackend] = {
    MeloOpenVoiceBackend.key: MeloOpenVoiceBackend(),
    F5TTSBackend.key:         F5TTSBackend(),
}

DEFAULT_TTS_BACKEND = MeloOpenVoiceBackend.key


def list_tts_backend_names() -> List[str]:
    """User-facing labels for the UI dropdown."""
    return [spec.name for spec in TTS_BACKEND_SPECS.values()]


def get_tts_backend_key_by_label(label: str) -> str:
    """Reverse-lookup: dropdown label → registry key. Defaults to the default backend."""
    if not label:
        return DEFAULT_TTS_BACKEND
    for key, spec in TTS_BACKEND_SPECS.items():
        if spec.name == label or spec.key == label:
            return key
    # Loose match: any backend whose UI name appears as a substring of the label.
    label_lower = label.lower()
    for key, spec in TTS_BACKEND_SPECS.items():
        if spec.key.split("-")[0] in label_lower or spec.name.split()[0].lower() in label_lower:
            return key
    return DEFAULT_TTS_BACKEND


def get_tts_backend(key_or_label: str) -> TTSBackend:
    """Resolve a backend by either its registry key or its UI label.

    Note: returns the *singleton* instance from TTS_BACKEND_SPECS. Callers must
    invoke .initialize(...) before .synthesize(...), and .cleanup() afterwards.
    A new render run can re-initialize the same instance safely.
    """
    if key_or_label in TTS_BACKEND_SPECS:
        return TTS_BACKEND_SPECS[key_or_label]
    return TTS_BACKEND_SPECS[get_tts_backend_key_by_label(key_or_label)]
