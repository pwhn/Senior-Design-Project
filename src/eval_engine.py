"""
Edu-Video AI Pipeline — Automated Evaluation Engine
=====================================================
Metrics implemented:
  1. Script Fidelity      — BERTScore + ROUGE-L  (generated script vs PDF text)
  2. Slide Visual Fidelity — SSIM between rendered PPTX PNGs and closest PDF pages
  3. Voice Cloning Quality — Resemblyzer speaker-embedding cosine similarity
  4. Generation Efficiency — Total render time + peak VRAM
  5. PresentQuiz           — LLM-generated quiz  (pre/post knowledge gain proxy)
  6. PresentArena / MOS    — LLM-based 1-5 Likert scoring on 5 aspects

All heavy imports are deferred so the module loads instantly.
"""

import os
import json
import time
import subprocess
import tempfile
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors the LLM isolation from app.py
# ---------------------------------------------------------------------------
LLM_ENV_PYTHON = r"C:\Users\Preston\Desktop\Y4S1\FYP\proj\llm_env\Scripts\python.exe"
LLM_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_worker.py")
LLM_MODEL_PATH = "C:/LLM/gemma-4-E4B-it-UD-Q6_K_XL.gguf"


def _release_gpu_memory():
    """Force-free all GPU VRAM held by this process so LLM subprocesses get full headroom."""
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except (ImportError, AttributeError):
        pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. SCRIPT FIDELITY — BERTScore + ROUGE-L                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def eval_script_fidelity(generated_script_json: str, pdf_path: str,
                         selected_pages: list | None = None):
    """Compare the generated narration script against the source PDF text.

    Returns dict with:
      bertscore_precision, bertscore_recall, bertscore_f1,
      rouge_l_precision, rouge_l_recall, rouge_l_f1
    """
    import fitz
    from bert_score import score as bert_score_fn
    from rouge_score import rouge_scorer

    # Extract reference text from PDF
    doc = fitz.open(pdf_path)
    pages = selected_pages if selected_pages else list(range(len(doc)))
    ref_text = ""
    for i in pages:
        if 0 <= i < len(doc):
            ref_text += doc[i].get_text() + "\n"
    doc.close()
    ref_text = ref_text.strip()
    if not ref_text:
        return {"error": "No text extracted from PDF"}

    # Extract candidate text from script JSON
    try:
        data = json.loads(generated_script_json)
        segments = data.get("segments", data) if isinstance(data, dict) else data
        candidate_text = " ".join(
            seg.get("spoken_text", "") or seg.get("script", "")
            for seg in segments
        ).strip()
    except Exception as e:
        return {"error": f"Script JSON parse error: {e}"}

    if not candidate_text:
        return {"error": "No spoken_text found in script"}

    # --- BERTScore (chunked to avoid OOM) ---
    # Split into ~512-token chunks for BERTScore
    def _chunk(text, max_chars=2000):
        words = text.split()
        chunks, cur = [], []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > max_chars and cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [w], len(w)
            else:
                cur.append(w)
                cur_len += len(w) + 1
        if cur:
            chunks.append(" ".join(cur))
        return chunks if chunks else [text]

    cand_chunks = _chunk(candidate_text)
    ref_chunks = _chunk(ref_text)
    # Align chunk counts (BERTScore needs parallel lists)
    min_len = min(len(cand_chunks), len(ref_chunks))
    cand_chunks = cand_chunks[:min_len]
    ref_chunks = ref_chunks[:min_len]

    P, R, F1 = bert_score_fn(cand_chunks, ref_chunks, lang="en", verbose=False,
                              device="cuda", batch_size=8)
    bs_p = P.mean().item()
    bs_r = R.mean().item()
    bs_f1 = F1.mean().item()

    # --- ROUGE-L ---
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ref_text[:10000], candidate_text[:10000])
    rl = scores["rougeL"]

    return {
        "bertscore_precision": round(bs_p, 4),
        "bertscore_recall": round(bs_r, 4),
        "bertscore_f1": round(bs_f1, 4),
        "rouge_l_precision": round(rl.precision, 4),
        "rouge_l_recall": round(rl.recall, 4),
        "rouge_l_f1": round(rl.fmeasure, 4),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2. SLIDE VISUAL FIDELITY — SSIM (rendered slide PNGs vs PDF pages)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def eval_slide_fidelity(pptx_slide_dir: str, pdf_path: str,
                        canvas_w=1280, canvas_h=720):
    """Compute SSIM between each rendered PPTX slide PNG and its best-matching
    PDF page (same index, or best SSIM across all pages).

    Returns dict with per-slide SSIM and mean_ssim.
    """
    import fitz
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    import cv2

    # Render PDF pages to images at same resolution
    doc = fitz.open(pdf_path)
    pdf_images = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        zoom = max(canvas_w / page.rect.width, canvas_h / page.rect.height)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.resize(img, (canvas_w, canvas_h))
        pdf_images.append(img)
    doc.close()

    if not pdf_images:
        return {"error": "No pages in PDF"}

    # Find slide PNGs
    slide_files = sorted(
        [f for f in os.listdir(pptx_slide_dir) if f.startswith("pptx_slide_") and f.endswith(".png")]
    )
    if not slide_files:
        return {"error": "No rendered slide PNGs found"}

    per_slide = {}
    for sf in slide_files:
        slide_img = cv2.imread(os.path.join(pptx_slide_dir, sf))
        if slide_img is None:
            continue
        slide_img = cv2.resize(slide_img, (canvas_w, canvas_h))
        slide_gray = cv2.cvtColor(slide_img, cv2.COLOR_BGR2GRAY)

        # Compare against all PDF pages, keep best SSIM
        best_ssim = -1
        best_page = -1
        for pi, pdf_img in enumerate(pdf_images):
            pdf_gray = cv2.cvtColor(pdf_img, cv2.COLOR_RGB2GRAY)  # fitz gives RGB
            val = ssim(slide_gray, pdf_gray)
            if val > best_ssim:
                best_ssim = val
                best_page = pi
        per_slide[sf] = {"ssim": round(best_ssim, 4), "matched_page": best_page}

    mean_val = sum(v["ssim"] for v in per_slide.values()) / len(per_slide) if per_slide else 0
    return {
        "per_slide": per_slide,
        "mean_ssim": round(mean_val, 4),
        "num_slides": len(per_slide),
        "num_pdf_pages": len(pdf_images),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3. VOICE CLONING QUALITY — Resemblyzer speaker similarity             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def eval_voice_similarity(reference_audio_path: str, generated_audio_dir: str):
    """Compute cosine similarity between the reference voice and each generated
    audio segment using Resemblyzer speaker embeddings.

    Returns dict with per-segment similarity + mean.
    """
    from resemblyzer import VoiceEncoder, preprocess_wav
    import numpy as np

    encoder = VoiceEncoder("cpu")

    ref_wav = preprocess_wav(reference_audio_path)
    ref_embed = encoder.embed_utterance(ref_wav)

    audio_files = sorted([
        f for f in os.listdir(generated_audio_dir)
        if f.endswith(".wav") and ("segment" in f.lower() or "seg_" in f.lower() or f.startswith("clone_"))
    ])
    if not audio_files:
        # Try all wav files
        audio_files = sorted([f for f in os.listdir(generated_audio_dir) if f.endswith(".wav")])

    per_segment = {}
    for af in audio_files:
        try:
            gen_wav = preprocess_wav(os.path.join(generated_audio_dir, af))
            if len(gen_wav) < 1600:  # too short
                continue
            gen_embed = encoder.embed_utterance(gen_wav)
            sim = float(np.dot(ref_embed, gen_embed) /
                        (np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed) + 1e-8))
            per_segment[af] = round(sim, 4)
        except Exception:
            continue

    mean_sim = sum(per_segment.values()) / len(per_segment) if per_segment else 0
    return {
        "per_segment": per_segment,
        "mean_similarity": round(mean_sim, 4),
        "num_segments": len(per_segment),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. GENERATION EFFICIENCY — time + VRAM                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def eval_efficiency(render_log: dict | None = None):
    """Return efficiency metrics. Pass a dict with keys like
    'total_time_s', 'peak_vram_mb', 'num_segments'.
    If None, try to read from the saved log file."""
    log_path = os.path.join(os.path.abspath("final_render_assets"), "render_log.json")
    if render_log is None and os.path.exists(log_path):
        with open(log_path, "r") as f:
            render_log = json.load(f)
    if render_log is None:
        return {"error": "No render log found. Run step 3 first."}

    total = render_log.get("total_time_s", 0)
    n_segs = render_log.get("num_segments", 1)
    return {
        "total_render_time_s": round(total, 1),
        "per_segment_time_s": round(total / max(n_segs, 1), 1),
        "peak_vram_mb": render_log.get("peak_vram_mb", "N/A"),
        "num_segments": n_segs,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5. PresentQuiz — LLM-generated quiz for knowledge gain proxy          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
QUIZ_SYSTEM_PROMPT = """You are a rigorous graduate-level academic examiner.
Generate exactly 7 multiple-choice questions from the provided lecture script.

Difficulty distribution: 2 easy, 3 medium, 2 hard.

RULES:
1. NO ROTE RECALL — ban definitions, acronym expansions, author names, raw numbers.
2. REQUIRE SYNTHESIS — test trade-offs, limitations, or broader implications.
3. DISTRACTOR ENGINEERING — wrong options must be plausible.
4. SCENARIO APPLICATION — frame as what-if scenarios where possible.
5. SHUFFLE the correct answer position across A-D.

Each question object has EXACTLY these 5 keys: question, options, correct_answer, difficulty, explanation.
The "options" value is an object with ONLY keys "A","B","C","D" mapping to answer strings.
Do NOT put correct_answer, difficulty, or explanation inside options.

Output STRICTLY as JSON (no markdown). Schema:
{"quiz":[{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"correct_answer":"B","difficulty":"medium","explanation":"One sentence."}]}
Begin with { and end with }."""


def _clean_narration_for_llm(text: str) -> str:
    """Strip Unicode artifacts, LaTeX fragments, and citation noise that confuse small LLMs."""
    # Replace common mojibake / Unicode replacement artifacts
    text = re.sub(r'[\ufffd\ufeff]', '', text)
    # Japanese-encoded dash artifacts (e.g. 窶・ 窶杷)
    text = re.sub(r'窶[^\s]*', '', text)
    # Strip inline LaTeX dollar-sign expressions, keep inner text
    text = re.sub(r'\$([^$]{1,120})\$', r'\1', text)
    # Strip leftover backslash commands (\textbf, \cite, \ref, etc.)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Collapse citation brackets  [1], [12,13], (Author et al., 2024)
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\([A-Z][a-z]+ et al\.,?\s*\d{4}\)', '', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _normalize_quiz_json(raw) -> list:
    """Extract a flat list of well-formed question dicts from potentially mangled LLM output.

    Handles:
      - Correct flat array: [q1, q2, ...]
      - Wrapper object: {"quiz": [...]} or {"questions": [...]}
      - Nested/recursive quiz keys inside options
      - correct_answer accidentally inside options dict
    """
    # If it's already a list, use it; otherwise dig for a list
    if isinstance(raw, list):
        candidates = raw
    elif isinstance(raw, dict):
        # Try known wrapper keys
        for key in ("quiz", "questions"):
            if key in raw and isinstance(raw[key], list):
                candidates = raw[key]
                break
        else:
            candidates = [raw]  # single question wrapped in object
    else:
        return []

    cleaned = []
    seen_questions = set()

    def _extract(obj):
        """Recursively pull question dicts out of arbitrarily nested structures."""
        if isinstance(obj, list):
            for item in obj:
                _extract(item)
            return
        if not isinstance(obj, dict):
            return

        q_text = obj.get("question", obj.get("q", ""))
        options = obj.get("options", {})

        # If the model nested correct_answer/difficulty/explanation inside options, pull them out
        correct = obj.get("correct_answer", "")
        difficulty = obj.get("difficulty", "")
        explanation = obj.get("explanation", "")
        inner_quiz = None

        if isinstance(options, dict):
            if not correct:
                correct = options.pop("correct_answer", "")
            if not difficulty:
                difficulty = options.pop("difficulty", "")
            if not explanation:
                explanation = options.pop("explanation", "")
            inner_quiz = options.pop("quiz", None)

        # Validate: must have question text and at least 2 option keys
        option_keys = {k for k in (options if isinstance(options, dict) else {}) if k in "ABCD"}
        if q_text and len(option_keys) >= 2 and q_text not in seen_questions:
            # Keep only A-D option keys
            clean_options = {k: v for k, v in options.items() if k in "ABCD" and isinstance(v, str)}
            cleaned.append({
                "question": q_text,
                "options": clean_options,
                "correct_answer": correct,
                "difficulty": difficulty,
                "explanation": explanation,
            })
            seen_questions.add(q_text)

        # Recurse into nested quiz arrays
        if inner_quiz:
            _extract(inner_quiz)
        for key in ("quiz", "questions"):
            if key in obj and key != "options":
                _extract(obj[key])

    _extract(candidates)
    return cleaned


def generate_quiz(script_json: str, llm_model_path: str = ""):
    """Ask the LLM to generate a 5-question MCQ quiz from the script.
    Returns parsed quiz dict or error dict."""
    active_model = llm_model_path.strip() if llm_model_path and llm_model_path.strip() else LLM_MODEL_PATH

    try:
        data = json.loads(script_json)
        segments = data.get("segments", data) if isinstance(data, dict) else data
        narration = "\n".join(
            f"[Segment {i+1}] {seg.get('spoken_text', '') or seg.get('script', '')}"
            for i, seg in enumerate(segments)
        )
    except Exception as e:
        return {"error": f"Script parse error: {e}"}

    narration = _clean_narration_for_llm(narration)
    narration = narration[:6000]

    llm_config = {
        "model_path": active_model,
        "system_prompt": QUIZ_SYSTEM_PROMPT,
        "user_content": f"Generate exactly 7 quiz questions from this narration:\n\n{narration}",
        "n_ctx": 8192,
        "n_batch": 512,
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.95,
        "stream": False,
        "json_mode": True,
        "stop": ["</s>", "<end_of_turn>", "<eos>"]
    }

    config_fd, config_path = tempfile.mkstemp(suffix=".json", prefix="llm_quiz_cfg_")
    try:
        with os.fdopen(config_fd, "w", encoding="utf-8") as cf:
            json.dump(llm_config, cf)

        result = subprocess.run(
            [LLM_ENV_PYTHON, LLM_WORKER_SCRIPT, config_path],
            capture_output=True, text=True, encoding="utf-8", timeout=900
        )
        if result.returncode != 0:
            return {"error": f"LLM worker failed: {result.stderr[:500]}"}

        content = None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "content" in d:
                    content = d["content"]
            except json.JSONDecodeError:
                continue

        if not content:
            return {"error": "No content from LLM"}

        # Try to repair truncated JSON: close open brackets/braces
        raw = content.strip()
        # Count open vs close brackets to detect truncation
        if raw.count('[') > raw.count(']'):
            raw = raw.rstrip(',') + ']'
        if raw.count('{') > raw.count('}'):
            raw += '}' * (raw.count('{') - raw.count('}'))

        parsed = json.loads(raw)
        questions = _normalize_quiz_json(parsed)
        if not questions:
            return {"error": "LLM produced no valid questions", "raw": content[:1000]}
        return {"questions": questions}
    except subprocess.TimeoutExpired:
        return {"error": "LLM quiz generation timed out (15min)"}
    except json.JSONDecodeError as e:
        return {"error": f"Quiz JSON parse error: {e}", "raw": content[:1000] if content else ""}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass


def attempt_quiz_self_evaluation(quiz_data: dict, script_json: str, llm_model_path: str = ""):
    """Have the LLM attempt to answer the generated quiz using ONLY the script narration.

    This simulates whether a student who watched the video would be able to
    answer the questions correctly, measuring knowledge transfer efficacy.

    Returns dict with score, total, percentage, strict_grounding_passed, and per-question details.
    """
    active_model = llm_model_path.strip() if llm_model_path and llm_model_path.strip() else LLM_MODEL_PATH

    # Fetch questions regardless of whether the prompt output "questions" or "quiz"
    questions = quiz_data.get("questions", quiz_data.get("quiz", []))
    if not questions:
        return {"error": "No questions to attempt", "score": 0, "total": 0, "percentage": 0.0, "strict_grounding_passed": False}

    # Build the narration context
    try:
        data = json.loads(script_json)
        segments = data.get("segments", data) if isinstance(data, dict) else data
        narration = "\n".join(
            f"[Segment {i+1}] {seg.get('spoken_text', '') or seg.get('script', '')}"
            for i, seg in enumerate(segments)
        )
    except Exception:
        narration = script_json[:12000]

    narration = _clean_narration_for_llm(narration)
    narration = narration[:6000]

    # Format questions for the LLM
    q_block = ""
    for i, q in enumerate(questions):
        # Support both 'q' and 'question' keys based on generation output
        question_text = q.get("q", q.get("question", ""))
        q_block += f"\nQ{i+1}: {question_text}\n"
        
        options = q.get("options", [])
        if isinstance(options, list):
            for opt in options:
                q_block += f"  {opt}\n"
        elif isinstance(options, dict):
            for k, v in options.items():
                q_block += f"  {k}) {v}\n"

    system_prompt = (
        "You are a diligent student taking a test. You must answer the multiple-choice questions "
        "based STRICTLY and EXCLUSIVELY on the provided narration transcript. "
        "CRITICAL RULE: If the transcript does not explicitly contain the information needed to answer "
        "a question, you must output 'NOT_PROVIDED' instead of a letter choice. Do not rely on outside knowledge. "
        "Return ONLY a valid JSON object: {\"answers\": [\"A\", \"NOT_PROVIDED\", \"C\", ...]}."
    )

    user_content = (
        f"Here is the narration transcript from the video:\n\n{narration}\n\n"
        f"Now answer these questions based ONLY on what is in the transcript:\n{q_block}"
    )

    llm_config = {
        "model_path": active_model,
        "system_prompt": system_prompt,
        "user_content": user_content,
        "n_ctx": 8192,
        "n_batch": 512,
        "temperature": 0.0,
        "max_tokens": 2048,
        "top_p": 0.95,
        "stream": False,
        "json_mode": True,
        "stop": ["</s>", "<end_of_turn>", "<eos>"]
    }

    config_fd, config_path = tempfile.mkstemp(suffix=".json", prefix="llm_selfquiz_cfg_")
    try:
        with os.fdopen(config_fd, "w", encoding="utf-8") as cf:
            json.dump(llm_config, cf)

        result = subprocess.run(
            [LLM_ENV_PYTHON, LLM_WORKER_SCRIPT, config_path],
            capture_output=True, text=True, encoding="utf-8", timeout=600
        )
        if result.returncode != 0:
            return {"error": f"LLM worker failed: {result.stderr[:500]}",
                    "score": 0, "total": max(0, len(questions) - 1), "percentage": 0.0, "strict_grounding_passed": False}

        content = None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "content" in d:
                    content = d["content"]
            except json.JSONDecodeError:
                continue

        if not content:
            return {"error": "No content from LLM",
                    "score": 0, "total": max(0, len(questions) - 1), "percentage": 0.0, "strict_grounding_passed": False}

        parsed = json.loads(content)
        llm_answers = parsed.get("answers", [])

        # Score against correct answers
        correct = 0
        trap_passed = False
        details = []
        
        for i, q in enumerate(questions):
            # Check both "answer" and "correct_answer" based on JSON schema variability
            expected = q.get("answer", q.get("correct_answer", "")).strip().upper()
            given = llm_answers[i].strip().upper() if i < len(llm_answers) else "?"
            
            question_text = q.get("q", q.get("question", ""))
            
            # --- Trap Question Logic ---
            if expected == "NOT_PROVIDED":
                trap_passed = (given == "NOT_PROVIDED")
                details.append({
                    "question": question_text,
                    "expected": "NOT_PROVIDED",
                    "given": given,
                    "trap_passed": trap_passed,
                })
            else:
                is_correct = (given == expected)
                if is_correct:
                    correct += 1
                details.append({
                    "question": question_text,
                    "expected": expected,
                    "given": given,
                    "correct": is_correct,
                })

        # Subtract 1 from total so the trap doesn't artificially drag down the percentage
        total_real_questions = max(0, len(questions) - 1)
        
        return {
            "score": correct,
            "total": total_real_questions,
            "percentage": round(100.0 * correct / total_real_questions, 1) if total_real_questions else 0.0,
            "strict_grounding_passed": trap_passed,
            "details": details,
        }

    except subprocess.TimeoutExpired:
        return {"error": "Self-quiz timed out (10min)", "score": 0, "total": max(0, len(questions) - 1), "percentage": 0.0, "strict_grounding_passed": False}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return {"error": f"Parse error: {e}", "score": 0, "total": max(0, len(questions) - 1), "percentage": 0.0, "strict_grounding_passed": False}
    except Exception as e:
        return {"error": str(e), "score": 0, "total": max(0, len(questions) - 1), "percentage": 0.0, "strict_grounding_passed": False}
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 6. PresentArena / MOS — LLM-based Likert scoring on 5 aspects         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
MOS_SYSTEM_PROMPT = """You are an expert educational content evaluator. You will be given a narration script for an AI-generated educational video. Rate it on each of the following 5 aspects using a 1-5 Likert scale.

Scoring guide:
  1 = Very Poor, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent

Aspects to evaluate:
  1. **Accuracy** — Is the content factually correct and faithful to academic material?
  2. **Clarity** — Is the explanation clear, well-structured, and easy to follow?
  3. **Engagement** — Is the narration engaging, using examples and natural language?
  4. **Visual_Appeal** — Do the slide/scene descriptions suggest good visual design?
  5. **Educational_Value** — Overall, how useful would this be for a student learning the topic?

Return ONLY a valid JSON object:
{
  "scores": {
    "accuracy": 4,
    "clarity": 5,
    "engagement": 3,
    "visual_appeal": 4,
    "educational_value": 4
  },
  "justification": {
    "accuracy": "Brief reason...",
    "clarity": "Brief reason...",
    "engagement": "Brief reason...",
    "visual_appeal": "Brief reason...",
    "educational_value": "Brief reason..."
  },
  "overall_recommendation": "A brief 2-3 sentence overall assessment."
}"""


def eval_mos_scores(script_json: str, llm_model_path: str = ""):
    """LLM-based MOS evaluation of the narration script.
    Returns dict with scores (1-5) for 5 aspects + justifications."""
    active_model = llm_model_path.strip() if llm_model_path and llm_model_path.strip() else LLM_MODEL_PATH

    try:
        data = json.loads(script_json)
        segments = data.get("segments", data) if isinstance(data, dict) else data
        narration = json.dumps(segments, indent=2)
    except Exception as e:
        return {"error": f"Script parse error: {e}"}

    narration = narration[:12000]

    llm_config = {
        "model_path": active_model,
        "system_prompt": MOS_SYSTEM_PROMPT,
        "user_content": f"Evaluate this narration script:\n\n{narration}",
        "n_ctx": 4096,
        "n_batch": 512,
        "temperature": 0.2,
        "max_tokens": -1,
        "top_p": 0.95,
        "stream": False,
        "json_mode": True,
        "stop": ["</s>", "<end_of_turn>", "<eos>"]
    }

    config_fd, config_path = tempfile.mkstemp(suffix=".json", prefix="llm_mos_cfg_")
    try:
        with os.fdopen(config_fd, "w", encoding="utf-8") as cf:
            json.dump(llm_config, cf)

        result = subprocess.run(
            [LLM_ENV_PYTHON, LLM_WORKER_SCRIPT, config_path],
            capture_output=True, text=True, encoding="utf-8", timeout=300
        )
        if result.returncode != 0:
            return {"error": f"LLM worker failed: {result.stderr[:500]}"}

        content = None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "content" in d:
                    content = d["content"]
            except json.JSONDecodeError:
                continue

        if not content:
            return {"error": "No content from LLM"}

        if not content.strip().endswith("}"):
            content = content.strip() + "}"

        return json.loads(content)
    except subprocess.TimeoutExpired:
        return {"error": "LLM MOS evaluation timed out (5min)"}
    except json.JSONDecodeError as e:
        return {"error": f"MOS JSON parse error: {e}", "raw": content[:1000] if content else ""}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CHART GENERATION — Plotly                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def generate_radar_chart(mos_scores: dict):
    """Generate a Plotly radar chart from MOS scores. Returns Plotly Figure."""
    import plotly.graph_objects as go

    scores = mos_scores.get("scores", {})
    if not scores:
        return None

    categories = list(scores.keys())
    values = [scores[c] for c in categories]
    # Close the polygon
    categories_display = [c.replace("_", " ").title() for c in categories]
    categories_display.append(categories_display[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories_display,
        fill="toself", name="AI Evaluation",
        line=dict(color="#0066CC", width=2),
        fillcolor="rgba(0,102,204,0.25)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
        ),
        title="PresentArena — MOS Radar Chart",
        showlegend=False,
        template="plotly_white",
        height=450,
    )
    return fig


def generate_metrics_bar_chart(results: dict):
    """Generate a bar chart showing all numeric metrics. Returns Plotly Figure."""
    import plotly.graph_objects as go

    metrics = {}

    # Script fidelity
    sf = results.get("script_fidelity", {})
    if "bertscore_f1" in sf:
        metrics["BERTScore F1"] = sf["bertscore_f1"]
    if "rouge_l_f1" in sf:
        metrics["ROUGE-L F1"] = sf["rouge_l_f1"]

    # Slide SSIM
    sv = results.get("slide_fidelity", {})
    if "mean_ssim" in sv:
        metrics["Slide SSIM"] = sv["mean_ssim"]

    # Voice similarity
    vs = results.get("voice_similarity", {})
    if "mean_similarity" in vs:
        metrics["Voice Similarity"] = vs["mean_similarity"]

    # MOS overall (average of 5 aspects)
    mos = results.get("mos_scores", {})
    scores = mos.get("scores", {})
    if scores:
        avg = sum(scores.values()) / len(scores)
        metrics["MOS Average (÷5)"] = round(avg / 5, 4)  # normalize to 0-1

    # Quiz knowledge transfer
    quiz = results.get("quiz", {})
    self_eval = quiz.get("self_evaluation", {})
    if self_eval and "error" not in self_eval and self_eval.get("total", 0) > 0:
        metrics["Quiz Score"] = round(self_eval["percentage"] / 100.0, 4)  # normalize to 0-1

    if not metrics:
        return None

    colors = ["#0066CC", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    fig = go.Figure([go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker_color=colors[:len(metrics)],
        text=[f"{v:.3f}" for v in metrics.values()],
        textposition="outside"
    )])
    fig.update_layout(
        title="Evaluation Metrics Summary (0–1 scale)",
        yaxis=dict(range=[0, 1.1], title="Score"),
        template="plotly_white",
        height=400,
    )
    return fig


def generate_quiz_chart(quiz_data: dict, answers: dict | None = None):
    """Generate a bar chart showing quiz results.
    Prioritises self_evaluation data if present. Falls back to difficulty distribution."""
    import plotly.graph_objects as go

    questions = quiz_data.get("questions", quiz_data.get("quiz", []))
    if not questions:
        return None

    # Prefer self-evaluation results when available
    self_eval = quiz_data.get("self_evaluation", {})
    if self_eval and "error" not in self_eval:
        sc = self_eval.get("score", 0)
        tot = self_eval.get("total", len(questions))
        pct = self_eval.get("percentage", 0)
        trap = self_eval.get("strict_grounding_passed", False)
        
        fig = go.Figure([go.Bar(
            x=["Correct", "Incorrect"],
            y=[sc, tot - sc],
            marker_color=["#4CAF50", "#f44336"],
            text=[str(sc), str(tot - sc)],
            textposition="outside"
        )])
        trap_label = "PASS ✓" if trap else "FAIL ✗"
        fig.update_layout(
            title=f"PresentQuiz — {sc}/{tot} ({pct}%) | Grounding Trap: {trap_label}",
            yaxis=dict(range=[0, tot + 2], title="Questions"),
            template="plotly_white",
            height=350,
        )
        return fig

    if answers:
        # Score the quiz
        correct = sum(
            1 for i, q in enumerate(questions)
            if answers.get(str(i), "").upper() == q.get("answer", q.get("correct_answer", "")).upper()
        )
        total = len(questions)
        fig = go.Figure([go.Bar(
            x=["Correct", "Incorrect"],
            y=[correct, total - correct],
            marker_color=["#4CAF50", "#f44336"],
            text=[str(correct), str(total - correct)],
            textposition="outside"
        )])
        fig.update_layout(
            title=f"PresentQuiz Results — {correct}/{total} ({100*correct/total:.0f}%)",
            yaxis=dict(range=[0, total + 1], title="Questions"),
            template="plotly_white",
            height=350,
        )
        return fig
    else:
        # Just show difficulty distribution
        diffs = {"easy": 0, "medium": 0, "hard": 0}
        for q in questions:
            d = q.get("difficulty", "medium").lower()
            diffs[d] = diffs.get(d, 0) + 1
        fig = go.Figure([go.Bar(
            x=list(diffs.keys()),
            y=list(diffs.values()),
            marker_color=["#4CAF50", "#FF9800", "#f44336"],
        )])
        fig.update_layout(
            title=f"Quiz Difficulty Distribution ({len(questions)} Questions)",
            yaxis=dict(title="Count"),
            template="plotly_white",
            height=350,
        )
        return fig


def generate_voice_similarity_chart(voice_results: dict):
    """Bar chart of per-segment voice similarity scores."""
    import plotly.graph_objects as go

    per_seg = voice_results.get("per_segment", {})
    if not per_seg:
        return None

    # Short labels
    labels = [f"Seg {i+1}" for i in range(len(per_seg))]
    values = list(per_seg.values())
    mean_val = voice_results.get("mean_similarity", 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=["#4CAF50" if v >= 0.75 else "#FF9800" if v >= 0.5 else "#f44336" for v in values],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        name="Per-Segment"
    ))
    fig.add_hline(y=mean_val, line_dash="dash", line_color="blue",
                  annotation_text=f"Mean: {mean_val:.3f}")
    fig.update_layout(
        title="Voice Cloning Similarity (Resemblyzer)",
        yaxis=dict(range=[0, 1.1], title="Cosine Similarity"),
        template="plotly_white",
        height=400,
    )
    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ FULL EVALUATION RUNNER                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def run_full_evaluation(script_json: str, pdf_path: str,
                        reference_audio_path: str = "",
                        selected_pages: list | None = None,
                        llm_model_path: str = "",
                        run_script_fidelity=True,
                        run_slide_fidelity=True,
                        run_voice_similarity=True,
                        run_efficiency=True,
                        run_quiz=True,
                        run_mos=True):
    """Generator that runs selected evaluations and yields status updates.
    Final yield is the complete results dict."""

    results = {}
    output_folder = os.path.abspath("final_render_assets")

    # 1. Script Fidelity
    if run_script_fidelity and pdf_path:
        yield "📝 Running Script Fidelity (BERTScore + ROUGE-L)..."
        try:
            results["script_fidelity"] = eval_script_fidelity(
                script_json, pdf_path, selected_pages
            )
        except Exception as e:
            results["script_fidelity"] = {"error": str(e)}
        yield f"  ✅ Script Fidelity done — BERTScore F1: {results['script_fidelity'].get('bertscore_f1', 'N/A')}"
        _release_gpu_memory()   # free BERTScore model VRAM before anything else

    # 2. Slide Visual Fidelity (only when PPTX slides were actually rendered)
    _has_slide_pngs = any(
        f.startswith("pptx_slide_") and f.endswith(".png")
        for f in os.listdir(output_folder)
    ) if os.path.isdir(output_folder) else False
    if run_slide_fidelity and pdf_path and _has_slide_pngs:
        yield "🖼️ Running Slide Visual Fidelity (SSIM)..."
        try:
            results["slide_fidelity"] = eval_slide_fidelity(
                output_folder, pdf_path
            )
        except Exception as e:
            results["slide_fidelity"] = {"error": str(e)}
        yield f"  ✅ Slide Fidelity done — Mean SSIM: {results['slide_fidelity'].get('mean_ssim', 'N/A')}"
    elif run_slide_fidelity and pdf_path and not _has_slide_pngs:
        yield "  ⏭️ Slide Fidelity skipped — no PPTX slides were generated in this run."

    # 3. Voice Similarity
    if run_voice_similarity and reference_audio_path:
        yield "🎙️ Running Voice Cloning Similarity (Resemblyzer)..."
        try:
            results["voice_similarity"] = eval_voice_similarity(
                reference_audio_path, output_folder
            )
        except Exception as e:
            results["voice_similarity"] = {"error": str(e)}
        yield f"  ✅ Voice Similarity done — Mean: {results['voice_similarity'].get('mean_similarity', 'N/A')}"

    # 4. Efficiency
    if run_efficiency:
        yield "⚡ Gathering Efficiency Metrics..."
        try:
            results["efficiency"] = eval_efficiency()
        except Exception as e:
            results["efficiency"] = {"error": str(e)}
        yield f"  ✅ Efficiency: {results['efficiency'].get('total_render_time_s', 'N/A')}s total"

    # 5. PresentQuiz
    if run_quiz:
        _release_gpu_memory()   # ensure full VRAM for LLM subprocess
        yield "🧠 Generating PresentQuiz (LLM — this may take a minute)..."
        try:
            results["quiz"] = generate_quiz(script_json, llm_model_path)
            
            # Inject the Hallucination Trap Question
            if "error" not in results["quiz"]:
                q_list = results["quiz"].get("questions", [])
                
                trap_question = {
                    "question": "In what year did Albert Einstein publish the theory of general relativity?",
                    "options": {
                        "A": "1905", 
                        "B": "1915", 
                        "C": "1921", 
                        "D": "1935"
                    },
                    "correct_answer": "NOT_PROVIDED",
                    "difficulty": "hard",
                    "explanation": "Trap question — answer is not in the script. Tests strict grounding."
                }
                q_list.append(trap_question)
                results["quiz"]["questions"] = q_list

        except Exception as e:
            results["quiz"] = {"error": str(e)}
        n_q = len(results["quiz"].get("questions", []))
        yield f"  ✅ Quiz generated — {n_q} questions (including 1 Trap Question)"

        # Self-evaluation: LLM attempts to answer the quiz using only the script
        if n_q > 0 and "error" not in results["quiz"]:
            yield "🎓 LLM Self-Quiz: Attempting to answer questions from script only..."
            try:
                self_quiz = attempt_quiz_self_evaluation(
                    results["quiz"], script_json, llm_model_path
                )
                results["quiz"]["self_evaluation"] = self_quiz
                pct = self_quiz.get("percentage", 0)
                sc = self_quiz.get("score", 0)
                tot = self_quiz.get("total", 0)
                grounding_status = "PASS" if self_quiz.get("strict_grounding_passed", False) else "FAIL"
                yield f"  ✅ Self-Quiz: {sc}/{tot} correct ({pct}%) | Grounding Check: {grounding_status}"
            except Exception as e:
                results["quiz"]["self_evaluation"] = {"error": str(e)}
                yield f"  ⚠️ Self-quiz failed: {e}"

    # 6. MOS
    if run_mos:
        _release_gpu_memory()   # ensure full VRAM for LLM subprocess
        yield "⭐ Running PresentArena MOS Evaluation (LLM — this may take a minute)..."
        try:
            results["mos_scores"] = eval_mos_scores(script_json, llm_model_path)
        except Exception as e:
            results["mos_scores"] = {"error": str(e)}
        mos_avg = "N/A"
        scores = results.get("mos_scores", {}).get("scores", {})
        if scores:
            mos_avg = f"{sum(scores.values()) / len(scores):.1f}/5"
        yield f"  ✅ MOS done — Average: {mos_avg}"

    yield "RESULTS:" + json.dumps(results)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SAVE EVALUATION REPORT TO FOLDER                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def save_evaluation_report(results: dict, charts: dict,
                           script_prompt: str = "",
                           pptx_prompt: str = "",
                           generation_settings: dict | None = None,
                           script_json: str = "",
                           pipeline_time_s: float | None = None):
    """Save all evaluation results, charts, and metadata to a timestamped folder.

    Args:
        results: Full evaluation results dict
        charts: Dict of chart_name -> Plotly Figure (or None)
        script_prompt: The system prompt used for script generation
        pptx_prompt: The system prompt used for PPTX generation
        generation_settings: Dict with LLM params, canvas res, etc.
        script_json: The raw script JSON string (narration + segments)

    Returns:
        Path to the saved report folder.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(os.path.abspath("evaluation_reports"), f"eval_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    # 1. Save full results JSON
    with open(os.path.join(report_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 1b. Save script JSON for cross-model evaluation
    if script_json:
        try:
            parsed = json.loads(script_json) if isinstance(script_json, str) else script_json
            with open(os.path.join(report_dir, "script_json.json"), "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
        except Exception:
            # Save raw string if JSON parsing fails
            with open(os.path.join(report_dir, "script_json.json"), "w", encoding="utf-8") as f:
                f.write(script_json if isinstance(script_json, str) else json.dumps(script_json))

    # 2. Save charts as images
    charts_dir = os.path.join(report_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    for chart_name, fig in charts.items():
        if fig is not None:
            try:
                # Save as HTML (always works, no kaleido needed)
                fig.write_html(os.path.join(charts_dir, f"{chart_name}.html"))
                # Try to save as PNG too (requires kaleido)
                try:
                    fig.write_image(os.path.join(charts_dir, f"{chart_name}.png"), width=1200, height=600)
                except Exception:
                    pass  # kaleido not installed, HTML is sufficient
            except Exception as e:
                print(f"⚠️ Could not save chart '{chart_name}': {e}")

    # 3. Save system prompts
    prompts = {}
    if script_prompt:
        prompts["script_generation_prompt"] = script_prompt
    if pptx_prompt:
        prompts["pptx_generation_prompt"] = pptx_prompt
    if prompts:
        with open(os.path.join(report_dir, "system_prompts.json"), "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

    # 4. Save generation settings
    if generation_settings:
        settings_to_save = dict(generation_settings)
        if pipeline_time_s is not None:
            settings_to_save["pipeline_total_time_s"] = round(pipeline_time_s, 1)
            mins = int(pipeline_time_s // 60)
            secs = int(pipeline_time_s % 60)
            settings_to_save["pipeline_total_time_human"] = f"{mins}m {secs}s"
        with open(os.path.join(report_dir, "generation_settings.json"), "w", encoding="utf-8") as f:
            json.dump(settings_to_save, f, indent=2, ensure_ascii=False)

    # 5. Save a human-readable summary
    summary_lines = [
        f"Edu-Video AI Pipeline — Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*60}",
        ""
    ]

    if pipeline_time_s is not None:
        mins = int(pipeline_time_s // 60)
        secs = int(pipeline_time_s % 60)
        summary_lines.append(f"⏱️ Total Pipeline Time (Build → Render): {mins}m {secs}s ({pipeline_time_s:.1f}s)")
        summary_lines.append("")

    sf = results.get("script_fidelity", {})
    if sf and "error" not in sf:
        summary_lines.append(f"📝 Script Fidelity:")
        summary_lines.append(f"   BERTScore F1:  {sf.get('bertscore_f1', 'N/A')}")
        summary_lines.append(f"   ROUGE-L F1:    {sf.get('rouge_l_f1', 'N/A')}")
        summary_lines.append("")

    sv = results.get("slide_fidelity", {})
    if sv and "error" not in sv:
        summary_lines.append(f"🖼️ Slide Visual Fidelity:")
        summary_lines.append(f"   Mean SSIM:     {sv.get('mean_ssim', 'N/A')}")
        summary_lines.append(f"   Slides:        {sv.get('num_slides', 'N/A')}")
        summary_lines.append("")

    vs = results.get("voice_similarity", {})
    if vs and "error" not in vs:
        summary_lines.append(f"🎙️ Voice Cloning Quality:")
        summary_lines.append(f"   Mean Similarity: {vs.get('mean_similarity', 'N/A')}")
        summary_lines.append(f"   Segments:        {vs.get('num_segments', 'N/A')}")
        summary_lines.append("")

    eff = results.get("efficiency", {})
    if eff and "error" not in eff:
        summary_lines.append(f"⚡ Generation Efficiency:")
        summary_lines.append(f"   Total Time:    {eff.get('total_render_time_s', 'N/A')}s")
        summary_lines.append(f"   Per Segment:   {eff.get('per_segment_time_s', 'N/A')}s")
        summary_lines.append(f"   Peak VRAM:     {eff.get('peak_vram_mb', 'N/A')} MB")
        summary_lines.append("")

    mos = results.get("mos_scores", {})
    scores = mos.get("scores", {})
    if scores:
        summary_lines.append(f"⭐ MOS Scores (1-5 Likert):")
        for aspect, val in scores.items():
            summary_lines.append(f"   {aspect.replace('_', ' ').title():20s}: {val}/5")
        avg = sum(scores.values()) / len(scores)
        summary_lines.append(f"   {'Average':20s}: {avg:.1f}/5")
        summary_lines.append("")

    quiz = results.get("quiz", {})
    questions = quiz.get("questions", [])
    if questions:
        self_eval = quiz.get("self_evaluation", {})
        summary_lines.append(f"🧠 PresentQuiz: {len(questions) - 1} questions generated (excluding trap)")
        if self_eval and "error" not in self_eval:
            grounding = "PASS" if self_eval.get("strict_grounding_passed", False) else "FAIL"
            summary_lines.append(f"   Knowledge Transfer Score: {self_eval.get('score', 0)}/{self_eval.get('total', 0)} ({self_eval.get('percentage', 0)}%)")
            summary_lines.append(f"   Strict Grounding Trap:    {grounding}")
        summary_lines.append("")

    with open(os.path.join(report_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return report_dir