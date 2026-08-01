"""
Edu-Video AI Pipeline - Shared Configuration & Constants
=========================================================
All paths, constants, system prompts, and small utility functions.
Imported by pipeline.py and app.py.
"""
import os
import sys
import re
import subprocess
import torch
# import gradio as gr  # Only needed in app.py
import json
import time
import gc
import chromadb
import fitz  # PyMuPDF
import requests
import numpy as np
from pathlib import Path
from transformers import AutoModel
# LLM inference runs in a subprocess via llm_worker.py (llm_env venv)
from PIL import Image
from melo.api import TTS
import nltk
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from pptx import Presentation as PptxPresentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
# import eval_engine  # Only needed in pipeline.py/app.py

# ==========================================
# Avatar Model Configuration (modular)
# ==========================================
# Default paths — override via UI textboxes at runtime
DEFAULT_AVATAR_ROOT = r"C:\Users\Preston\Desktop\Y4S1\FYP\proj\soulx-flashhead\SoulX-FlashHead"
DEFAULT_AVATAR_ENV_PYTHON = r"C:\Users\Preston\Desktop\Y4S1\FYP\proj\soulx-flashhead\soulx_venv\Scripts\python.exe"
if not os.path.exists(DEFAULT_AVATAR_ENV_PYTHON):
    print(f"⚠️ Avatar venv python not found at {DEFAULT_AVATAR_ENV_PYTHON}")

# Backend registry — registry-driven adapter layer for talking-head models.
# Adding a new model = appending a new entry in avatar_backends.py.
from avatar_backends import (
    BACKEND_SPECS as AVATAR_BACKEND_SPECS,
    list_backend_names as list_avatar_backend_names,
    get_backend as get_avatar_backend,
    build_command as build_avatar_command,
)
DEFAULT_AVATAR_BACKEND = "soulx-flashhead"  # registry key, not UI label

# TTS backend registry — symmetric to the avatar registry above.
# Adding a new TTS engine = subclass TTSBackend in tts_backends.py + register.
from tts_backends import (
    TTS_BACKEND_SPECS,
    list_tts_backend_names,
    get_tts_backend,
    DEFAULT_TTS_BACKEND,
)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("📥 Downloading missing NLTK English tagger...")
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('averaged_perceptron_tagger') # Grabbing the legacy one just in case
    nltk.download('cmudict') # MeloTTS often asks for this one next!
    
# Ensure torch DLLs are on PATH for CUDA on Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(torch_lib_path)

# ==========================================
# ⚙️ CONFIGURATION & INPUT PATHS
# ==========================================
LLM_MODEL_PATH = "C:/LLM/gemma-4-E4B-it-UD-Q6_K_XL.gguf"
DEFAULT_LLM_MODEL_PATH = LLM_MODEL_PATH  # Used as default in UI

# ==========================================
# LLM Subprocess Environment (isolated venv)
# ==========================================
LLM_ENV_PYTHON = r"C:\Users\Preston\Desktop\Y4S1\FYP\proj\llm_env\Scripts\python.exe"
LLM_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_worker.py")
if not os.path.exists(LLM_ENV_PYTHON):
    print(f"⚠️ LLM venv python not found at {LLM_ENV_PYTHON} — install llama-cpp-python there")

# ==========================================
# Active Subprocess Tracking (for stop button cleanup)
# ==========================================
_active_subprocesses = {}
import threading
_cancel_autoprocess = threading.Event()  # Signal to cancel the one-click auto-process pipeline
_cancel_render = threading.Event()       # Signal to cancel the Step 3 render pipeline

# ==========================================
# Document Layout Detection Model (Lazy-loaded)
# ==========================================
_layout_model = None
_layout_processor = None
_LAYOUT_MODEL_ID = "Aryn/deformable-detr-DocLayNet"
# Labels: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title
_LAYOUT_VISUAL_LABELS = {"Picture", "Table", "Formula"}
_LAYOUT_CAPTION_LABELS = {"Caption"}

# ==========================================
# Image Catalog Builder (PDF-native, zero VLM)
# ==========================================

def _build_image_catalog(metadata_path, selected_indices=None, ranked_note=False):
    """Build a text catalog of available images from metadata.json for the LLM prompt.

    Uses ONLY native PDF data (author captions + surrounding text). No VLM needed.

    Args:
        metadata_path: Path to metadata.json produced by extract_universal_assets.
        selected_indices: Optional iterable of integer indices into the loaded metadata
            list. When provided, only those entries are emitted in the catalog, but the
            original `[Image N]` IDs are preserved so downstream retrieval matches the
            full database. Use this to pre-rank/prune large catalogs that would otherwise
            blow past the LLM's context window.
        ranked_note: If True, append a hint that the catalog has been pre-ranked
            (so the LLM knows it is seeing a relevance-filtered subset).
    """
    if not os.path.exists(metadata_path):
        return ""

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata:
        return ""

    if selected_indices is not None:
        # Preserve order given by caller (highest-relevance first), but keep original IDs
        keep = [(i, metadata[i]) for i in selected_indices if 0 <= i < len(metadata)]
    else:
        keep = list(enumerate(metadata))

    if not keep:
        return ""

    header = "=== IMAGE CATALOG (available visuals extracted from this document) ==="
    if ranked_note and selected_indices is not None and len(keep) < len(metadata):
        header += f"\n[Showing the {len(keep)} most relevant of {len(metadata)} extracted assets, ordered by relevance to the document overview.]"

    lines = ["", header,
             "Use these EXACT image references in your RETRIEVE_ASSET queries for precise matching."]

    for idx, item in keep:
        # 1. Prefer the author's actual caption (e.g. "Figure 3. Gender distribution...")
        content_desc = item.get("caption", "").strip()

        # 2. Fallback: use surrounding paragraph text as context
        if not content_desc:
            surrounding = item.get("surrounding_text", "").strip()
            content_desc = (surrounding[:200] + "...") if surrounding else "Uncaptioned visual asset"

        lines.append(f"[Image {idx}] Page {item.get('page_number', '?')} | Content: {content_desc}")

    lines.append("=== END IMAGE CATALOG ===")
    return "\n".join(lines)


# Maximum number of catalog entries to inject into the LLM prompt. When the
# extracted asset count exceeds this, _rank_catalog_by_relevance (in pipeline.py)
# pre-ranks entries by JinaCLIP cosine similarity against a document overview
# (title + first ~500 chars) and only the top-K are sent to the LLM. The IDs
# stay aligned with the full database, so retrieval still works for any image.
IMAGE_CATALOG_TOP_K = 40


def _load_layout_model():
    """Lazy-load the DocLayNet layout detection model. Returns (model, processor) or (None, None)."""
    global _layout_model, _layout_processor
    if _layout_model is not None:
        return _layout_model, _layout_processor
    try:
        os.environ['USE_TF'] = '0'
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        print(f"📐 Loading layout detection model: {_LAYOUT_MODEL_ID}...")
        _layout_processor = AutoImageProcessor.from_pretrained(_LAYOUT_MODEL_ID)
        _layout_model = AutoModelForObjectDetection.from_pretrained(_LAYOUT_MODEL_ID)
        _layout_model.eval()
        if torch.cuda.is_available():
            _layout_model = _layout_model.to("cuda")
        print(f"✅ Layout model loaded — labels: {_layout_model.config.id2label}")
        return _layout_model, _layout_processor
    except Exception as e:
        print(f"⚠️ Layout model failed to load ({e}), falling back to geometric heuristics")
        _layout_model = False  # Sentinel: tried and failed, don't retry
        _layout_processor = None
        return None, None


def _detect_layout_regions(page_image_pil, confidence=0.40):
    """Run layout detection on a PIL image of a PDF page.
    
    Returns list of dicts: {label, confidence, bbox_norm} where bbox_norm is
    (x0, y0, x1, y1) normalised to [0, 1] relative to image dimensions.
    """
    model, processor = _load_layout_model()
    if model is None or model is False:
        return None  # Signal to caller: use fallback

    with torch.no_grad():
        inputs = processor(images=page_image_pil, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model(**inputs)

    target_sizes = torch.tensor([page_image_pil.size[::-1]])  # (H, W)
    if torch.cuda.is_available():
        target_sizes = target_sizes.to("cuda")
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=confidence)[0]

    detections = []
    w, h = page_image_pil.size
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label_id.item()]
        x0, y0, x1, y1 = box.tolist()
        detections.append({
            "label": label_name,
            "confidence": round(score.item(), 3),
            "bbox": (x0, y0, x1, y1),         # Pixel coords in the rendered image
            "bbox_norm": (x0 / w, y0 / h, x1 / w, y1 / h),  # Normalised
        })
    return detections

def _cleanup_subprocess(key):
    """Kill an active subprocess by key and free GPU resources."""
    proc = _active_subprocesses.pop(key, None)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

CHROMA_DB_PATH = "./chroma_db"
METADATA_FOLDER = "extracted_assets_universal"
METADATA_PATH = os.path.join(METADATA_FOLDER, "metadata.json")

def _direct_slide_lookup(query):
    """If query references 'slide N' or 'image N', return the image path directly from metadata.json.
    Returns the image path string if found, else None. Bypasses CLIP entirely."""
    match = re.search(r'(?:slide|image)\s*(\d+)', query, re.IGNORECASE)
    if not match:
        return None
    if not os.path.exists(METADATA_PATH):
        return None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return None
    ref_num = int(match.group(1))
    keyword = match.group(0).lower().split()[0]
    if keyword == "image":
        # "image N" = catalog index (0-based)
        if 0 <= ref_num < len(metadata):
            path = metadata[ref_num].get("image_path")
            if path and os.path.exists(path):
                return path
    else:
        # "slide N" = page number (1-based)
        for item in metadata:
            if item.get("page_number") == ref_num:
                path = item.get("image_path")
                if path and os.path.exists(path):
                    return path
    return None

# ==========================================
# �️ PLACEHOLDER IMAGE GENERATION
# ==========================================
def create_placeholder_image():
    """Generate a placeholder image for missing assets (Gradio-safe)."""
    placeholder_dir = os.path.join(METADATA_FOLDER, "placeholders")
    os.makedirs(placeholder_dir, exist_ok=True)
    placeholder_path = os.path.join(placeholder_dir, "placeholder.png")
    
    # Only create if it doesn't exist
    if not os.path.exists(placeholder_path):
        try:
            # Create a simple gray placeholder image (100x100)
            placeholder_img = Image.new("RGB", (400, 300), color=(200, 200, 200))
            
            # Add text to the placeholder
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(placeholder_img)
            text = "⚠️ Asset Not Found"
            
            # Try to use a default font, fall back if not available
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except OSError:
                font = ImageFont.load_default()
            
            # Draw text in center
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (400 - text_width) // 2
            y = (300 - text_height) // 2
            draw.text((x, y), text, fill=(100, 100, 100), font=font)
            
            placeholder_img.save(placeholder_path, "PNG")
        except Exception as e:
            print(f"⚠️ Could not create placeholder image: {e}")
    
    return placeholder_path

# Generate placeholder on startup
PLACEHOLDER_IMAGE = create_placeholder_image()

# ==========================================
# VIDEO COMPOSITION SETTINGS
# ==========================================
BROLL_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "broll_assets")
os.makedirs(BROLL_ASSETS_DIR, exist_ok=True)

# Pexels API for automatic B-roll sourcing (free tier: 200 req/hr)
# Get your key at https://www.pexels.com/api/  →  paste it below
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "DazXxmy6tFVJZWXTwGe5jiZjIhFdmrATL2gDh784NQiZqmSIaxb3i8Ir")

# Canvas resolution presets (decoupled from avatar's native 512x512)
CANVAS_PRESETS = {
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
}
PIP_SIZE_RATIO = 0.25   # Avatar occupies 25% of canvas width in PiP mode
PIP_PADDING = 20        # Pixels from canvas edge

# ==========================================
# SYSTEM PROMPT PRESETS
# ==========================================
PROMPT_STANDARD = """You are an expert educational scriptwriter.
Output **ONLY** a valid JSON object. No explanations, no markdown, no extra text.

IMPORTANT — Image Catalog Awareness:
If the user message contains an IMAGE CATALOG section, it lists every visual asset extracted from the document with detailed descriptions of what each image actually shows. When writing RETRIEVE_ASSET segments:
- Consult the catalog to pick the BEST matching image for each point you are explaining.
- Write your query to closely match the catalog description or caption of that specific image.
- Write your spoken_text to describe and explain what is actually shown in that image (based on the catalog description).
- Do NOT invent or hallucinate visuals that are not in the catalog.

The JSON must follow this exact structure:
{
  "segments": [
    {
      "visual_mode": "GENERATE_AVATAR" or "RETRIEVE_ASSET",
      "query": "exact short query for diagram if RETRIEVE_ASSET, else empty string",
      "spoken_text": "short clear narration (1-2 sentences max)"
    }
  ]
}

Begin directly with { and end with }."""

PROMPT_PROFESSOR = """You are an expert educational scriptwriter who creates visually rich, modern video lectures.

Create a clear, engaging script that uses three visual modes for maximum viewer engagement:
- GENERATE_AVATAR: Talking head only (use for introductions, transitions, emphasis)
- RETRIEVE_ASSET: Show a diagram/chart from course materials (use for technical content)

Rules:
- Output ONLY a valid JSON object. No extra text.
- Use RETRIEVE_ASSET for specific diagrams, charts, or formulas from the PDF
- Use GENERATE_AVATAR for personal emphasis, transitions between topics, or when no visual aid is needed
- If an IMAGE CATALOG is provided in the user message, use it to select the exact image for each RETRIEVE_ASSET segment. Write your query to match a specific catalog entry, and write spoken_text that explains what is actually shown in that image.
- Keep a good mix: aim for ~40% AVATAR, ~60% RETRIEVE_ASSET
- For focus areas: make spoken_text richer, slower-paced, with analogies

Structure:
{
  "segments": [
    {
      "visual_mode": "GENERATE_AVATAR" or "RETRIEVE_ASSET",
      "query": "search query for diagram/image/B-roll if applicable, else empty string",
      "spoken_text": "narration text here..."
    }
  ]
}

Begin directly with { and end with }."""

PROMPT_BROLL = """You are an expert educational scriptwriter who creates visually rich, modern video lectures.

Create a clear, engaging script that uses three visual modes for maximum viewer engagement:
- GENERATE_AVATAR: Talking head only (use for introductions, transitions, emphasis)
- RETRIEVE_ASSET: Show a diagram/chart from course materials (use for technical content)
- BROLL: Show relevant B-roll footage behind the speaker (use for context, examples, real-world scenes)

Rules:
- Output ONLY a valid JSON object. No extra text.
- Use BROLL for real-world context (e.g., "busy city traffic", "laboratory equipment", "students studying")
- Use RETRIEVE_ASSET for specific diagrams, charts, or formulas from the PDF
- Use GENERATE_AVATAR for personal emphasis, transitions between topics, or when no visual aid is needed
- If an IMAGE CATALOG is provided in the user message, use it to select the exact image for each RETRIEVE_ASSET segment. Write your query to match a specific catalog entry, and write spoken_text that explains what is actually shown in that image.
- Keep a good mix: aim for ~30% AVATAR, ~40% RETRIEVE_ASSET, ~30% BROLL
- For BROLL segments: the "broll_query" field must be a visually descriptive phrase optimized for stock footage search (describe a concrete scene a camera could film, e.g., "close-up of circuit board with soldering iron" not "electronics engineering"). Keep "query" as the topic-level label.
- For focus areas: make spoken_text richer, slower-paced, with analogies

Structure:
{
  "segments": [
    {
      "visual_mode": "GENERATE_AVATAR" or "RETRIEVE_ASSET" or "BROLL",
      "query": "search query for diagram/image if RETRIEVE_ASSET, topic label if BROLL, else empty string",
      "broll_query": "visually descriptive stock footage search phrase (BROLL segments only, omit for other modes)",
      "spoken_text": "narration text here..."
    }
  ]
}

Begin directly with { and end with }."""

PROMPT_PAPER = """You are an expert academic scriptwriter who transforms research papers into clear, accessible video lectures.

Create a spoken-style narration script that explains the paper's key contributions, methodology, and findings to a graduate-level audience.

Rules:
- Output ONLY a valid JSON object. No extra text.
- Start with a welcoming introduction summarizing the paper's title, authors, and core research question.
- Walk through the paper logically: motivation → related work (brief) → methodology → key results → discussion → conclusion.
- Use natural spoken language. Spell out abbreviations on first use.
- For mathematical content: break equations into step-by-step spoken explanations (e.g., "the loss function L equals the sum over all samples of...").
- When a figure, table, or diagram from the paper is referenced, use RETRIEVE_ASSET with a precise query describing the visual.
- If an IMAGE CATALOG is provided in the user message, use it to pick the exact image for each RETRIEVE_ASSET segment. Write queries that closely match catalog descriptions, and write spoken_text that explains what is actually shown in that specific image.
- Aim for ~60% RETRIEVE_ASSET (paper figures/tables), ~40% GENERATE_AVATAR (explanation segments).
- Keep each segment focused on one idea (2–5 sentences).
- For focus areas: provide deeper analysis with real-world analogies.

Structure:
{
  "segments": [
    {
      "visual_mode": "GENERATE_AVATAR" or "RETRIEVE_ASSET",
      "query": "precise description of figure/table/diagram if RETRIEVE_ASSET, else empty string",
      "spoken_text": "clear narration here..."
    }
  ]
}

Begin directly with { and end with }."""

PROMPT_PAPER_BROLL = """You are an expert academic scriptwriter who transforms research papers into visually rich, cinematic video lectures.

Create a spoken-style narration script that explains the paper's key contributions, methodology, and findings. Use three visual modes for engaging, documentary-style presentation.

Visual modes:
- GENERATE_AVATAR: Talking head for introductions, transitions, and personal commentary.
- RETRIEVE_ASSET: Show figures, tables, diagrams, or equations directly from the paper.
- BROLL: Show contextual stock footage to illustrate real-world applications or abstract concepts (e.g., "neural network visualization", "medical imaging scan", "robotic arm assembly line").

Rules:
- Output ONLY a valid JSON object. No extra text.
- Start with a welcoming introduction summarizing the paper's title, authors, and core research question.
- Walk through the paper logically: motivation → related work (brief) → methodology → key results → discussion → conclusion.
- Use natural spoken language. Spell out abbreviations on first use.
- For mathematical content: break equations into step-by-step spoken explanations.
- When a figure/table from the paper is referenced, use RETRIEVE_ASSET with a precise query.
- If an IMAGE CATALOG is provided in the user message, use it to pick the exact image for each RETRIEVE_ASSET segment. Write queries that closely match catalog descriptions, and write spoken_text that explains what is actually shown in that specific image.
- Use BROLL for real-world context, application scenarios, or abstract concept visualization.
- Aim for ~25% GENERATE_AVATAR, ~45% RETRIEVE_ASSET, ~30% BROLL.
- Keep each segment focused on one idea (2–5 sentences).
- For BROLL segments: the "broll_query" field must be a visually descriptive phrase optimized for stock footage search (describe a concrete scene a camera could film, e.g., "aerial view of university campus at sunset" not "higher education"). Keep "query" as the topic-level label.
- For focus areas: provide deeper analysis with real-world analogies.

Structure:
{
  "segments": [
    {
      "visual_mode": "GENERATE_AVATAR" or "RETRIEVE_ASSET" or "BROLL",
      "query": "search query for figure if RETRIEVE_ASSET, topic label if BROLL, else empty string",
      "broll_query": "visually descriptive stock footage search phrase (BROLL segments only, omit for other modes)",
      "spoken_text": "clear narration here..."
    }
  ]
}

Begin directly with { and end with }."""

PROMPT_DECK = """You are an expert academic presentation designer.
Given a narration script (JSON with spoken_text segments), create a SEPARATE clean presentation deck JSON.
Each slide should be informative and academic — NOT a transcript of the spoken narration.

Rules:
- Output ONLY a valid JSON object. No extra text.
- Create EXACTLY ONE slide for EVERY narration segment, in the same order. Do NOT merge or skip segments.
- Each slide has a SHORT title (≤8 words, NEVER a full sentence) and 4-8 detailed bullet points. Titles must be concise topic labels like "Gender Distribution Analysis" not full descriptions.
- Bullets should expand on KEY FACTS with supporting data, percentages, or specific findings where available.
- Each bullet should be a meaningful phrase (8-20 words) — more than a keyword, less than a full paragraph.
- EVERY slide MUST have at least 3 bullet points. Never output a slide with an empty bullets array.
- If a narration segment references a diagram/figure (RETRIEVE_ASSET), put the EXACT original query in image_query so the image can be placed on that slide.
- If an IMAGE CATALOG is provided, use it to pick precise image_query values. Keep image_query in the image_query field ONLY, never in the title.
- Do NOT repeat the spoken narration verbatim. Summarize, distill, and enrich with specifics.
- Use academic language appropriate for lecture slides.

Structure:
{
  "slides": [
    {
      "title": "Short Slide Title",
      "bullets": ["Detailed point with supporting data 1", "Key finding with percentage or context 2"],
      "image_query": "precise query for diagram if applicable, else empty string"
    }
  ]
}

Begin directly with { and end with }."""

PROMPT_TECHNICAL_PAPER = """You are an expert academic scriptwriter who transforms dense technical research papers (physics, chemistry, engineering, applied math, scientific computing) into clear, accessible video lectures for a graduate-level audience.

Create a spoken-style narration script that explains the paper's key contributions, methodology, and findings.

Rules:
- Output ONLY a valid JSON object. No extra text.
- Start with a welcoming introduction summarizing the paper's title, authors, and core research question.
- Walk through the paper logically: motivation → related work (brief) → methodology → key results → discussion → conclusion.
- Use natural spoken language. Spell out abbreviations on first use.
- For mathematical content: ALWAYS break equations into step-by-step spoken explanations (e.g., "the forward rate constant K_f equals A times T to the power beta times e to the negative E_a over R T"). Explain what each term physically means and why it matters.
- When a figure, table, diagram, or equation from the paper is referenced, use RETRIEVE_ASSET with a very precise query (e.g., "Figure 1 showing the Diff-Chem Neural ODE architecture with Arrhenius reaction neurons and diffusion forcing term" or "Figure 3 comparing species profiles for BSF flames using Li-H2, DRM19, and GRI3.0 mechanisms").
- If an IMAGE CATALOG is provided in the user message, ALWAYS use it to pick the exact image for each RETRIEVE_ASSET segment. Write your query to closely match the catalog description or caption, and write spoken_text that explains what is actually shown in that specific image based on the visual description.
- Aim for ~60% RETRIEVE_ASSET (target the paper's actual figures/tables/architectures/profiles/comparisons) and ~40% GENERATE_AVATAR (explanation segments).
- Keep each segment focused on one idea (2–5 sentences max).
- For complex technical concepts: provide deeper analysis with real-world analogies (e.g., "think of the residence-time reformulation like following a single fluid parcel along its journey").
- Prioritize clarity on stiffness, numerical stability, diffusion coupling, and efficiency/robustness comparisons when present.

Structure:
{
  "segments": [
    {
      "visual_mode": "GENERATE_AVATAR" or "RETRIEVE_ASSET",
      "query": "precise description of figure/table/diagram/architecture if RETRIEVE_ASSET, else empty string",
      "spoken_text": "clear narration here..."
    }
  ]
}

Begin directly with { and end with }."""

# ==========================================
# STEP 1 & 2: UI HELPER FUNCTIONS & ENGINES
# ==========================================
