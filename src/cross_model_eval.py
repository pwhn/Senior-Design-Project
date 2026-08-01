"""
Cross-Model Evaluation — Gemini / GPT-4o evaluates Gemma's outputs
====================================================================
Eliminates shared-weight self-referential bias by having an independent
cloud LLM re-attempt the PresentQuiz and re-score the MOS rubric on
the same narration scripts that Gemma generated & self-evaluated.

Supported backends:
  - **Gemini** (default) — free via Google AI Studio, uses gemini-2.0-flash
  - **OpenAI** (optional) — requires paid API key, uses gpt-4o

Two modes:
  1. Batch mode  — sweep all existing evaluation_reports/ folders
  2. Single mode — evaluate one report folder

Requirements:
  No external packages needed — uses stdlib urllib to call OpenAI-compatible REST APIs.

Usage (CLI):
  python cross_model_eval.py --api-key AIza...                  # Gemini batch
  python cross_model_eval.py --api-key AIza... --report <dir>   # Gemini single
  python cross_model_eval.py --api-key AIza... --cold           # cold quiz only
  python cross_model_eval.py --api-key sk-... --provider openai # use GPT-4o
"""

import os
import json
import argparse
import time
import urllib.request
import urllib.error
import ssl
from pathlib import Path

from eval_engine import MOS_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_reports")
FINAL_RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_render_assets")

# Model names per provider
MODELS = {
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-4o",
}


# ---------------------------------------------------------------------------
# Provider Abstraction  (stdlib-only — no openai / google-genai needed)
# ---------------------------------------------------------------------------
_SSL_CTX = ssl.create_default_context()


class _HttpChatClient:
    """Generic OpenAI-compatible chat client using only stdlib urllib."""

    MAX_RETRIES = 8
    BASE_DELAY = 10  # seconds — aggressive backoff for Gemini free tier
    MIN_CALL_INTERVAL = 5  # seconds — global cooldown between any two API calls
    _last_call_time = 0.0  # class-level, shared across instances

    def __init__(self, api_key: str, model: str, base_url: str):
        self._api_key = api_key
        self._model_name = model
        self._endpoint = base_url.rstrip("/") + "/chat/completions"

    def generate_json(self, system: str, user: str, temperature: float = 0.1) -> str:
        """Send a chat-completion request and return the assistant text.

        Enforces a minimum cooldown between API calls, then retries
        automatically on HTTP 429 with exponential backoff.
        """
        # Global cooldown — prevent burst requests across calls
        since_last = time.time() - _HttpChatClient._last_call_time
        if since_last < self.MIN_CALL_INTERVAL:
            wait = self.MIN_CALL_INTERVAL - since_last
            print(f"    ⏳ Cooldown: waiting {wait:.1f}s before next API call...")
            time.sleep(wait)

        body = json.dumps({
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        }).encode("utf-8")

        for attempt in range(self.MAX_RETRIES):
            _HttpChatClient._last_call_time = time.time()
            req = urllib.request.Request(
                self._endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, context=_SSL_CTX) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.MAX_RETRIES - 1:
                    # Try to read error body for details (quota type, retry-after)
                    err_body = ""
                    try:
                        err_body = e.read().decode("utf-8", errors="replace")[:300]
                    except Exception:
                        pass
                    delay = self.BASE_DELAY * (2 ** attempt)  # 10, 20, 40, 80, 160, 320, 640
                    print(f"    ⏳ Rate limited (429). Retrying in {delay}s (attempt {attempt+1}/{self.MAX_RETRIES})...")
                    if err_body:
                        print(f"       Detail: {err_body[:200]}")
                    time.sleep(delay)
                    continue
                # Non-retriable error or retries exhausted — read body for diagnostics
                err_detail = ""
                try:
                    err_detail = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                if err_detail:
                    print(f"    ❌ HTTP {e.code} error detail: {err_detail[:300]}")
                raise

    @property
    def model_name(self) -> str:
        return self._model_name


class _GeminiClient(_HttpChatClient):
    """Gemini via its OpenAI-compatible endpoint (free, no SDK needed)."""
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, api_key: str, model: str = ""):
        super().__init__(api_key, model or MODELS["gemini"], self.GEMINI_BASE_URL)


class _OpenAIClient(_HttpChatClient):
    """OpenAI via the standard API endpoint."""

    def __init__(self, api_key: str, model: str = ""):
        super().__init__(api_key, model or MODELS["openai"], "https://api.openai.com/v1/")


def _make_client(api_key: str, provider: str = "gemini"):
    """Factory that returns the appropriate client wrapper."""
    provider = provider.lower().strip()
    if provider == "openai":
        return _OpenAIClient(api_key)
    return _GeminiClient(api_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_narration_from_report(report_dir: str) -> str | None:
    """Try to load the narration script from various locations.
    
    Priority:
      1. script_json.json saved alongside evaluation_results.json (new format)
      2. final_timeline.json in final_render_assets (only valid for last run)
    """
    # 1. Check for script_json.json inside the report folder (future-proofed)
    script_path = os.path.join(report_dir, "script_json.json")
    if os.path.exists(script_path):
        data = _load_json(script_path)
        if data:
            segments = data.get("segments", data) if isinstance(data, dict) else data
            return "\n".join(
                f"[Segment {i+1}] {seg.get('spoken_text', '') or seg.get('script', '')}"
                for i, seg in enumerate(segments)
            )

    # 2. Fallback: final_timeline.json (only has the latest run)
    timeline_path = os.path.join(FINAL_RENDER_DIR, "final_timeline.json")
    if os.path.exists(timeline_path):
        data = _load_json(timeline_path)
        if data and isinstance(data, list):
            return "\n".join(
                f"[Segment {i+1}] {seg.get('script', '')}"
                for i, seg in enumerate(data)
            )

    return None


def _extract_quiz_questions(eval_results: dict) -> list:
    """Extract quiz questions from evaluation_results.json."""
    return eval_results.get("quiz", {}).get("questions", [])


def _format_quiz_block(questions: list) -> str:
    """Format questions into a text block for the LLM prompt."""
    block = ""
    for i, q in enumerate(questions):
        block += f"\nQ{i+1}: {q.get('q', '')}\n"
        for opt in q.get("options", []):
            block += f"  {opt}\n"
    return block


# ---------------------------------------------------------------------------
# Cross-Model Quiz Attempt
# ---------------------------------------------------------------------------
def cross_attempt_quiz(client, questions: list, narration: str | None = None) -> dict:
    """Have the cross-model LLM answer the quiz questions.
    
    If narration is provided → comprehension quiz (cross-model PresentQuiz).
    If narration is None    → cold quiz (validity check — tests question specificity).
    """
    q_block = _format_quiz_block(questions)
    
    if narration:
        system = (
            "You are a diligent student who just watched an educational video. "
            "Using ONLY the narration transcript provided, answer each multiple-choice question. "
            "Return ONLY a valid JSON object: {\"answers\": [\"A\", \"B\", ...]} "
            "with exactly one letter (A, B, C, or D) per question, in order."
        )
        user = (
            f"Here is the narration transcript from the video:\n\n"
            f"{narration[:12000]}\n\n"
            f"Now answer these questions based ONLY on what is in the transcript:\n{q_block}"
        )
    else:
        system = (
            "You are a knowledgeable student taking a quiz. "
            "Answer each question to the best of your ability using your general knowledge. "
            "Return ONLY a valid JSON object: {\"answers\": [\"A\", \"B\", ...]} "
            "with exactly one letter (A, B, C, or D) per question, in order."
        )
        user = f"Answer these questions:\n{q_block}"

    content = client.generate_json(system, user, temperature=0.1)
    parsed = json.loads(content)
    llm_answers = parsed.get("answers", [])

    # Score against correct answers
    correct = 0
    details = []
    for i, q in enumerate(questions):
        expected = q.get("answer", "").strip().upper()
        given = llm_answers[i].strip().upper() if i < len(llm_answers) else "?"
        is_correct = (given == expected)
        if is_correct:
            correct += 1
        details.append({
            "question": q.get("q", ""),
            "expected": expected,
            "given": given,
            "correct": is_correct,
        })

    total = len(questions)
    return {
        "model": client.model_name,
        "mode": "comprehension" if narration else "cold",
        "score": correct,
        "total": total,
        "percentage": round(100.0 * correct / total, 1) if total else 0.0,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Cross-Model MOS Re-scoring
# ---------------------------------------------------------------------------
# MOS_SYSTEM_PROMPT imported from eval_engine (single source of truth)


def cross_mos_eval(client, narration: str) -> dict:
    """Have the cross-model LLM score the narration on the same 5-dimension MOS rubric."""
    content = client.generate_json(
        MOS_SYSTEM_PROMPT,
        f"Evaluate this narration script:\n\n{narration[:12000]}",
        temperature=0.2,
    )
    result = json.loads(content)
    result["model"] = client.model_name
    return result


# ---------------------------------------------------------------------------
# Single Report Evaluation
# ---------------------------------------------------------------------------
def evaluate_single_report(client, report_dir: str,
                           run_cold: bool = True,
                           run_comprehension: bool = True,
                           run_mos: bool = True) -> dict:
    """Run cross-model evaluation on a single report folder.
    
    Returns dict with cross_quiz_cold, cross_quiz_comprehension, cross_mos.
    """
    eval_path = os.path.join(report_dir, "evaluation_results.json")
    eval_results = _load_json(eval_path)
    if not eval_results:
        return {"error": f"No evaluation_results.json in {report_dir}"}

    questions = _extract_quiz_questions(eval_results)
    if not questions:
        return {"error": "No quiz questions found in report"}

    narration = _get_narration_from_report(report_dir)
    report_name = os.path.basename(report_dir)
    cross_results = {"report": report_name}

    # Original Gemma self-eval for comparison
    gemma_self = eval_results.get("quiz", {}).get("self_evaluation", {})
    cross_results["gemma_self_eval"] = {
        "score": gemma_self.get("score", "N/A"),
        "total": gemma_self.get("total", "N/A"),
        "percentage": gemma_self.get("percentage", "N/A"),
    }

    # 1. Cold Quiz (no narration — validity check)
    if run_cold:
        print(f"  🧊 Cold quiz (no narration)...")
        try:
            cross_results["cross_quiz_cold"] = cross_attempt_quiz(client, questions, narration=None)
        except Exception as e:
            cross_results["cross_quiz_cold"] = {"error": str(e)}

    # 2. Comprehension Quiz (with narration)
    if run_comprehension and narration:
        print(f"  📖 Comprehension quiz (with narration)...")
        try:
            cross_results["cross_quiz_comprehension"] = cross_attempt_quiz(client, questions, narration=narration)
        except Exception as e:
            cross_results["cross_quiz_comprehension"] = {"error": str(e)}
    elif run_comprehension and not narration:
        cross_results["cross_quiz_comprehension"] = {"skipped": "No narration script found for this report"}

    # 3. Cross-model MOS
    if run_mos and narration:
        print(f"  ⭐ Cross-model MOS scoring...")
        try:
            cross_results["cross_mos"] = cross_mos_eval(client, narration)
            # Include Gemma MOS for comparison
            gemma_mos = eval_results.get("mos_scores", {}).get("scores", {})
            if gemma_mos:
                cross_results["gemma_mos"] = gemma_mos
        except Exception as e:
            cross_results["cross_mos"] = {"error": str(e)}
    elif run_mos and not narration:
        cross_results["cross_mos"] = {"skipped": "No narration script found for this report"}

    # Save per-report cross_model_results.json alongside the original eval
    try:
        per_report_path = os.path.join(report_dir, "cross_model_results.json")
        with open(per_report_path, "w", encoding="utf-8") as f:
            json.dump(cross_results, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved to {per_report_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save per-report results: {e}")

    return cross_results


# ---------------------------------------------------------------------------
# Batch Evaluation
# ---------------------------------------------------------------------------
def evaluate_all_reports(client,
                         run_cold: bool = True,
                         run_comprehension: bool = True,
                         run_mos: bool = True) -> list[dict]:
    """Sweep all evaluation_reports/ folders and run cross-model eval on each."""
    if not os.path.isdir(EVAL_REPORTS_DIR):
        print(f"❌ No evaluation_reports/ directory found at {EVAL_REPORTS_DIR}")
        return []

    report_dirs = sorted([
        os.path.join(EVAL_REPORTS_DIR, d)
        for d in os.listdir(EVAL_REPORTS_DIR)
        if os.path.isdir(os.path.join(EVAL_REPORTS_DIR, d))
    ])

    print(f"📂 Found {len(report_dirs)} report folders\n")

    all_results = []
    for i, rdir in enumerate(report_dirs, 1):
        name = os.path.basename(rdir)
        print(f"[{i}/{len(report_dirs)}] {name}")

        result = evaluate_single_report(
            client, rdir,
            run_cold=run_cold,
            run_comprehension=run_comprehension,
            run_mos=run_mos,
        )
        all_results.append(result)

        # Delay between reports to respect rate limits (Gemini free = 15 RPM)
        time.sleep(10)
        print()

    return all_results


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------
def print_summary(all_results: list[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("CROSS-MODEL EVALUATION SUMMARY")
    print("=" * 100)
    print(f"{'Report':<55} {'Gemma':>8} {'Cross Cold':>12} {'Cross Comp':>12}")
    print("-" * 100)

    gemma_scores = []
    cold_scores = []
    comp_scores = []

    for r in all_results:
        if "error" in r:
            print(f"  ❌ {r.get('error', 'Unknown error')}")
            continue

        name = r.get("report", "?")[:52]
        gemma_pct = r.get("gemma_self_eval", {}).get("percentage", "N/A")
        cold = r.get("cross_quiz_cold", {})
        comp = r.get("cross_quiz_comprehension", {})

        cold_pct = cold.get("percentage", "—") if "error" not in cold and "skipped" not in cold else "—"
        comp_pct = comp.get("percentage", "—") if "error" not in comp and "skipped" not in comp else "—"

        print(f"  {name:<53} {str(gemma_pct):>7}% {str(cold_pct):>10}% {str(comp_pct):>10}%")

        if isinstance(gemma_pct, (int, float)):
            gemma_scores.append(gemma_pct)
        if isinstance(cold_pct, (int, float)):
            cold_scores.append(cold_pct)
        if isinstance(comp_pct, (int, float)):
            comp_scores.append(comp_pct)

    print("-" * 100)
    if gemma_scores:
        print(f"  {'AVERAGE':<53} {sum(gemma_scores)/len(gemma_scores):>7.1f}%", end="")
    else:
        print(f"  {'AVERAGE':<53} {'N/A':>8}", end="")
    if cold_scores:
        print(f" {sum(cold_scores)/len(cold_scores):>10.1f}%", end="")
    else:
        print(f" {'N/A':>11}", end="")
    if comp_scores:
        print(f" {sum(comp_scores)/len(comp_scores):>10.1f}%")
    else:
        print(f" {'N/A':>11}")
    print("=" * 100)

    # MOS comparison
    mos_reports = [r for r in all_results if "cross_mos" in r and "scores" in r.get("cross_mos", {})]
    if mos_reports:
        print("\nMOS COMPARISON (Gemma vs Cross-Model)")
        print("-" * 80)
        aspects = ["accuracy", "clarity", "engagement", "visual_appeal", "educational_value"]
        print(f"  {'Aspect':<22} {'Gemma Avg':>10} {'Cross Avg':>11} {'Delta':>8}")
        print(f"  {'-'*22} {'-'*10} {'-'*11} {'-'*8}")

        for aspect in aspects:
            gemma_vals = [r.get("gemma_mos", {}).get(aspect, 0) for r in mos_reports if r.get("gemma_mos")]
            gpt_vals = [r["cross_mos"]["scores"].get(aspect, 0) for r in mos_reports]
            g_avg = sum(gemma_vals) / len(gemma_vals) if gemma_vals else 0
            gpt_avg = sum(gpt_vals) / len(gpt_vals) if gpt_vals else 0
            delta = gpt_avg - g_avg
            sign = "+" if delta >= 0 else ""
            print(f"  {aspect.replace('_', ' ').title():<22} {g_avg:>9.2f} {gpt_avg:>10.2f} {sign}{delta:>7.2f}")
        print()


# ---------------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------------
def save_cross_eval_results(all_results: list[dict], output_path: str | None = None):
    """Save the full cross-evaluation results to a JSON file."""
    if output_path is None:
        output_path = os.path.join(EVAL_REPORTS_DIR, "cross_model_evaluation.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full results saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Programmatic entry point (for app.py integration)
# ---------------------------------------------------------------------------
def run_cross_eval(api_key: str,
                   report_dir: str | None = None,
                   run_cold: bool = True,
                   run_comprehension: bool = True,
                   run_mos: bool = True,
                   provider: str = "gemini"):
    """Run cross-model evaluation. Returns (results_list, summary_path).
    
    Args:
        api_key: API key for the chosen provider.
        report_dir: If given, evaluates that one report. Otherwise all.
        provider: "gemini" (default, free) or "openai".
    """
    client = _make_client(api_key, provider)

    if report_dir:
        result = evaluate_single_report(
            client, report_dir,
            run_cold=run_cold,
            run_comprehension=run_comprehension,
            run_mos=run_mos,
        )
        all_results = [result]
    else:
        all_results = evaluate_all_reports(
            client,
            run_cold=run_cold,
            run_comprehension=run_comprehension,
            run_mos=run_mos,
        )

    output_path = save_cross_eval_results(all_results)
    print_summary(all_results)
    return all_results, output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-Model Evaluation — Gemini/GPT-4o vs Gemma")
    parser.add_argument("--report", type=str, default=None,
                        help="Path to a single report folder to evaluate")
    parser.add_argument("--cold", action="store_true", default=False,
                        help="Run cold quiz only (no narration, no MOS)")
    parser.add_argument("--no-mos", action="store_true", default=False,
                        help="Skip MOS re-scoring")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (Gemini or OpenAI)")
    parser.add_argument("--provider", type=str, default="gemini",
                        choices=["gemini", "openai"],
                        help="Which cloud LLM to use (default: gemini)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ No API key provided. Set GEMINI_API_KEY / OPENAI_API_KEY or use --api-key")
        return

    if args.cold:
        run_cross_eval(
            api_key=api_key,
            report_dir=args.report,
            run_cold=True,
            run_comprehension=False,
            run_mos=False,
            provider=args.provider,
        )
    else:
        run_cross_eval(
            api_key=api_key,
            report_dir=args.report,
            run_cold=True,
            run_comprehension=True,
            run_mos=not args.no_mos,
            provider=args.provider,
        )


if __name__ == "__main__":
    main()
