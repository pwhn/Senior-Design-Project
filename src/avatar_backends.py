"""
Avatar Backend Registry
=======================
Modular adapter layer for talking-head generation models.

Each backend is described by an `AvatarBackend` spec: the entry-point script
name inside the model repo, the CLI flag mapping, and (optionally) any extra
fixed flags the model needs. The pipeline calls `build_command(...)` with the
chosen backend name plus the runtime paths; nothing in the rest of the codebase
needs to change when a new model is added — just append a new spec below.

To add a new talking-head model:
  1. Make sure its inference script accepts (image, audio, output) on the CLI.
     If it doesn't, write a thin shim inside the model repo that does.
  2. Add a new entry to BACKEND_SPECS with the entry script + flag names.
  3. Done. The UI dropdown picks it up automatically.

All backends share the same subprocess plumbing in pipeline.py
(PYTHONHASHSEED sanitisation, TORCH_COMPILE_DISABLE, progress streaming, etc.)
because that infrastructure is backend-agnostic.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AvatarBackend:
    """Describes how to invoke a talking-head model from a subprocess."""
    name: str                                    # User-facing dropdown label
    entry_script: str                            # Script name inside the model repo (e.g. "generate_video.py")
    image_flag: str                              # CLI flag for the source image
    audio_flag: str                              # CLI flag for the input audio
    output_flag: str                             # CLI flag for the output video path
    ckpt_flag: Optional[str] = None              # CLI flag for the main checkpoint dir (None = not needed)
    audio_encoder_flag: Optional[str] = None     # CLI flag for audio encoder dir (e.g. wav2vec). None = not needed.
    audio_encoder_dir_hint: Optional[str] = None # Substring used to auto-detect the audio encoder subdir
    extra_args: List[str] = field(default_factory=list)  # Fixed extra CLI args (e.g. ["--model_type", "lite"])
    description: str = ""                        # Short human-readable description shown in the UI


# ──────────────────────────────────────────────────────────────────────────────
# Registry of supported backends.
# ──────────────────────────────────────────────────────────────────────────────
BACKEND_SPECS: Dict[str, AvatarBackend] = {

    "soulx-flashhead": AvatarBackend(
        name="SoulX-FlashHead (default)",
        entry_script="generate_video.py",
        image_flag="--cond_image",
        audio_flag="--audio_path",
        output_flag="--save_file",
        ckpt_flag="--ckpt_dir",
        audio_encoder_flag="--wav2vec_dir",
        audio_encoder_dir_hint="wav2vec",
        extra_args=["--model_type", "lite"],
        description="1.3B audio-driven facial animation. Deterministic, identity locked to input photo.",
    ),

    # ── Example backends (drop-in if the user installs the corresponding repo) ──
    # These show how trivially other talking-head models can be added.
    # Each entry took ~6 lines of config — no pipeline changes required.

    "sadtalker": AvatarBackend(
        name="SadTalker",
        entry_script="inference.py",
        image_flag="--source_image",
        audio_flag="--driven_audio",
        output_flag="--result_dir",
        ckpt_flag="--checkpoint_dir",
        audio_encoder_flag=None,
        extra_args=["--still", "--preprocess", "full"],
        description="3DMM-based talking head. Mature, lighter weight than SoulX. Output written into result_dir.",
    ),

    "wav2lip": AvatarBackend(
        name="Wav2Lip",
        entry_script="inference.py",
        image_flag="--face",
        audio_flag="--audio",
        output_flag="--outfile",
        ckpt_flag="--checkpoint_path",
        audio_encoder_flag=None,
        extra_args=["--nosmooth"],
        description="Lip-sync only model. Lower expressive range but extremely fast and well-tested.",
    ),
}


def list_backend_names() -> List[str]:
    """User-facing labels for the UI dropdown."""
    return [spec.name for spec in BACKEND_SPECS.values()]


def get_backend_key_by_label(label: str) -> str:
    """Reverse-lookup: dropdown label → registry key. Defaults to soulx-flashhead."""
    for key, spec in BACKEND_SPECS.items():
        if spec.name == label:
            return key
    return "soulx-flashhead"


def get_backend(key_or_label: str) -> AvatarBackend:
    """Resolve a backend by either its registry key or its UI label."""
    if key_or_label in BACKEND_SPECS:
        return BACKEND_SPECS[key_or_label]
    return BACKEND_SPECS[get_backend_key_by_label(key_or_label)]


def build_command(
    backend: AvatarBackend,
    python_exe: str,
    repo_root: str,
    inference_script_path: str,
    image_path: str,
    audio_path: str,
    output_path: str,
    ckpt_dir: Optional[str] = None,
    audio_encoder_dir: Optional[str] = None,
) -> List[str]:
    """Build the subprocess argv list for the given backend."""
    cmd = [python_exe, inference_script_path,
           backend.image_flag, image_path,
           backend.audio_flag, audio_path,
           backend.output_flag, output_path]

    if backend.ckpt_flag and ckpt_dir:
        cmd += [backend.ckpt_flag, ckpt_dir]

    if backend.audio_encoder_flag and audio_encoder_dir:
        cmd += [backend.audio_encoder_flag, audio_encoder_dir]

    cmd += list(backend.extra_args)
    return cmd
