# Architecture overview

## Core components

- app.py — Gradio-based user interface
- pipeline.py — main orchestration logic for processing PDFs, generating scripts, retrieval, and video composition
- config.py — shared constants, prompts, paths, and backend settings
- llm_worker.py — isolated inference worker for LLM execution
- eval_engine.py — evaluation logic for script fidelity, slide quality, and audio quality

## Typical workflow

1. Upload lecture material or slides
2. Extract or select relevant pages
3. Build a visual database from the document
4. Generate a narration script with LLM prompting
5. Retrieve matching visuals and render the final presentation
6. Optionally export a PowerPoint deck and evaluate outputs

## Design strengths

- Modular backend selection for avatars and TTS
- Clear separation between UI, config, and business logic
- Evaluation layer for measuring output quality
