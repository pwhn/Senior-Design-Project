"""
Edu-Video AI Pipeline - Gradio UI
===================================
Thin UI shell. All business logic lives in pipeline.py,
all configuration in config.py.
"""
import json
import gradio as gr
import eval_engine

# Import all config constants needed by the UI
from config import (
    DEFAULT_LLM_MODEL_PATH,
    PROMPT_BROLL, PROMPT_DECK,
    DEFAULT_AVATAR_ROOT, DEFAULT_AVATAR_ENV_PYTHON,
    DEFAULT_AVATAR_BACKEND, AVATAR_BACKEND_SPECS, list_avatar_backend_names,
    DEFAULT_TTS_BACKEND, TTS_BACKEND_SPECS, list_tts_backend_names,
    _cancel_autoprocess, _cancel_render,
)

# Import all pipeline functions used in event handlers
from pipeline import (
    generate_thumbnails_and_preview,
    bulk_select_all, bulk_deselect_all, bulk_invert_selection,
    select_range_fn, deselect_range_fn,
    apply_preset, update_prompt_text,
    build_database, test_rag_query, add_custom_assets,
    step1_generate_script, step2_fetch_assets,
    update_override_ui, handle_gallery_click,
    extract_retrieve_segments,
    step3_render_video,
    generate_pptx_deck,
    auto_process_all,
)



# ==========================================
# GRADIO UI DEFINITION
# ==========================================
with gr.Blocks(title="Edu-Video AI Pipeline") as dashboard:
    pdf_images_state = gr.State([]) 
    selected_pages_state = gr.State([])
    pdf_fullres_state = gr.State([])

    gr.Markdown("# Edu-Video AI: Instructor Dashboard")
    gr.Markdown("Upload course materials, build the visual database, generate scripts, and render your AI avatar presentation.")

    # ══════════════════════════════════════════════
    # STEP 1: PDF Upload & Visual Database
    # ══════════════════════════════════════════════
    with gr.Tab("Step 1: PDF Upload & Visual Database"):
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload Course PDF/Slides (.pdf)", file_types=[".pdf"])
                generate_preview_btn = gr.Button("📸 Generate Page Previews", variant="secondary", size="lg")
                preview_status = gr.Textbox(label="Preview Status", interactive=False, visible=True)
                
                with gr.Group(visible=False) as gallery_controls:
                    with gr.Row():
                        select_all_btn = gr.Button("✅ Select All", variant="secondary")
                        deselect_all_btn = gr.Button("❌ Clear All", variant="secondary")
                        invert_btn = gr.Button("🔄 Invert", variant="secondary")
                    with gr.Row():
                        range_input = gr.Textbox(show_label=False, placeholder="e.g. 1-5, 8, 10-12", scale=3)
                        select_range_btn = gr.Button("➕ Select", variant="secondary", scale=1)
                        deselect_range_btn = gr.Button("➖ Remove", variant="secondary", scale=1)
                
                pdf_gallery = gr.Gallery(
                    label="Page Previews — Native Resolution (Click to magnify!)", 
                    show_label=True, columns=4, height="auto", object_fit="contain", 
                    visible=False, interactive=False, preview=True
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🗃️ Visual Database")
                gr.Markdown("Extract images/diagrams from your PDF and build a searchable vector database for automatic asset retrieval.")
                build_db_btn = gr.Button(" Build Visual Database (JinaCLIP)", variant="primary", size="lg")
                db_status = gr.Textbox(label="Database Build Status", interactive=False)

                with gr.Accordion("🔍 Test Retrieval", open=False):
                    gr.Markdown("Type a visual description to see what JinaCLIP finds in your database.")
                    with gr.Row():
                        test_query = gr.Textbox(label="Search Query", placeholder="e.g., 'diagram of money laundering cycle'", scale=4)
                        test_btn = gr.Button("Test Search", variant="secondary", scale=1)
                    test_gallery = gr.Gallery(label="Top 2 Matches", show_label=True, columns=2, object_fit="contain")

                with gr.Accordion("📎 Add Custom Assets", open=False):
                    gr.Markdown("Upload your own images (diagrams, equations, charts) to the visual database. "
                                "They will be embedded and retrievable alongside extracted PDF assets.")
                    custom_asset_upload = gr.File(
                        label="Upload Images", file_count="multiple", file_types=["image"],
                        type="filepath"
                    )
                    custom_asset_captions = gr.Textbox(
                        label="Captions (one per line, matching upload order)",
                        placeholder="e.g.\nEquation (24): Loss function definition\nFigure 5: System architecture",
                        lines=3, interactive=True
                    )
                    add_assets_btn = gr.Button("➕ Add to Database", variant="secondary")
                    add_assets_status = gr.Textbox(label="Status", interactive=False)
        
        # --- Step 1 Event Wiring ---
        generate_preview_btn.click(fn=generate_thumbnails_and_preview, inputs=pdf_input, outputs=[preview_status, pdf_gallery, gallery_controls, pdf_images_state, selected_pages_state, pdf_fullres_state])
        select_all_btn.click(fn=bulk_select_all, inputs=pdf_images_state, outputs=[pdf_gallery, selected_pages_state])
        deselect_all_btn.click(fn=bulk_deselect_all, inputs=pdf_images_state, outputs=[pdf_gallery, selected_pages_state])
        invert_btn.click(fn=bulk_invert_selection, inputs=[pdf_images_state, selected_pages_state], outputs=[pdf_gallery, selected_pages_state])
        select_range_btn.click(fn=select_range_fn, inputs=[range_input, pdf_images_state, selected_pages_state], outputs=[pdf_gallery, selected_pages_state])
        deselect_range_btn.click(fn=deselect_range_fn, inputs=[range_input, pdf_images_state, selected_pages_state], outputs=[pdf_gallery, selected_pages_state])
        build_db_btn.click(fn=build_database, inputs=pdf_input, outputs=db_status)
        test_btn.click(fn=test_rag_query, inputs=test_query, outputs=test_gallery)
        add_assets_btn.click(fn=add_custom_assets, inputs=[custom_asset_upload, custom_asset_captions], outputs=add_assets_status)

    # ══════════════════════════════════════════════
    # STEP 2: Script Generation & Asset Management
    # ══════════════════════════════════════════════
    with gr.Tab("Step 2: Script Generation & Assets"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Script Generation")
                with gr.Accordion("⚙️ Advanced Engine Parameters", open=False):
                    gr.Markdown("### 📝 Persona & Prompt Settings")
                    prompt_preset = gr.Dropdown(choices=["University Professor (Detailed)", "Standard (Concise)", "Enhanced with B-Roll (PiP Layout)", "Academic Paper (Detailed)", "Academic Paper + B-Roll", "Technical/STEM Paper", "Custom"], value="Enhanced with B-Roll (PiP Layout)", label="Script Style Persona")
                    system_prompt_box = gr.Textbox(value=PROMPT_BROLL, label="System Prompt", lines=10, interactive=True)
                    focus_areas = gr.Textbox(
                        label="🎯 Focus Areas / Topics to Elaborate On (optional)",
                        placeholder="e.g. Explain go deeper into VRAM optimization challenges",
                        lines=3,
                        value="",
                        info="The LLM will automatically create richer, longer segments and more examples for these topics."
                    )
                    long_doc_mode = gr.Radio(
                        choices=["Auto (Sliding Window / Map-Reduce)", "Fast / Truncate (Single Pass)"],
                        value="Auto (Sliding Window / Map-Reduce)",
                        label="📄 Long Document Strategy",
                        info="Auto: multiple passes covering the full document (slower). Fast: single pass with proportional page sampling (quick but loses content on large docs)."
                    )
                    prompt_preset.change(fn=update_prompt_text, inputs=prompt_preset, outputs=system_prompt_box)
                    
                    gr.Markdown("### 🎛️ Hardware Settings")
                    preset_radio = gr.Radio(choices=["⚡ Fast Draft (Low VRAM)", "📚 Full Lecture (High VRAM)", "⚙️ Custom"], value="📚 Full Lecture (High VRAM)", label="Hardware Presets")
                    ctx_slider = gr.Slider(minimum=1024, maximum=16384, step=1024, value=10240, label="Context Window (n_ctx)")
                    batch_slider = gr.Slider(minimum=256, maximum=4096, step=256, value=1024, label="Reading Speed (n_batch)")
                    temp_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.3, label="Creativity (Temperature)")
                    preset_radio.change(fn=apply_preset, inputs=preset_radio, outputs=[ctx_slider, batch_slider, temp_slider])
                    
                    gr.Markdown("### 🤖 LLM Model")
                    llm_model_input = gr.Textbox(
                        value=DEFAULT_LLM_MODEL_PATH,
                        label="GGUF Model Path",
                        placeholder="C:/LLM/your-model.gguf",
                        info="Full path to the GGUF model file. Change to switch between different LLMs."
                    )
                
                with gr.Row():
                    generate_btn = gr.Button(" Generate Script with Qwen/Gemma", variant="primary", size="lg")
                    stop_script_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                status_box = gr.Textbox(label="Engine Status", interactive=False)
                
            with gr.Column(scale=1):
                gr.Markdown("### ✍️ Review & Edit Script")
                script_output = gr.Code(label="Semantic Engine Output (JSON)", language="json", interactive=True)
        
        gr.Markdown("---")
        
        # --- Asset Retrieval & Override Section ---
        manual_selections_state = gr.State({})
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🤖 Automated Asset Fetching")
                fetch_btn = gr.Button(" Fetch All Assets for Current Script", variant="primary", size="lg")
                asset_gallery = gr.Gallery(label="Retrieved ChromaDB Assets", show_label=True, columns=3, rows=1, height="auto")
            
            with gr.Column(scale=1):
                gr.Markdown("### ✏️ Manual Asset Override")
                gr.Markdown("Select a segment, then click an extracted visual to assign it. Shows all figures/tables/charts from the PDF and any custom assets you added.")
                segment_dropdown = gr.Dropdown(label="Select Segment to Correct", choices=[], interactive=True)
                override_status = gr.Textbox(label="Override Status", interactive=False)
                with gr.Row():
                    refresh_override_btn = gr.Button("🔄 Refresh Assets", variant="secondary")
                    clear_overrides_btn = gr.Button("🗑️ Clear All Overrides", variant="stop")
                override_gallery = gr.Gallery(label="Available Visual Assets (Click to assign!)", columns=4, interactive=True)

        gr.Markdown("---")
        gr.Markdown("### 📊 PowerPoint Deck Generator")
        gr.Markdown("Generate a professional `.pptx` presentation deck. The LLM creates concise academic slides "
                    "from your narration script, with automatic image insertion from fetched assets.")
        with gr.Accordion("⚙️ PPTX Deck Prompt & Settings", open=False):
            pptx_prompt_box = gr.Textbox(value=PROMPT_DECK, label="PPTX System Prompt", lines=10, interactive=True)
            gr.Markdown("💡 Edit the system prompt to control slide structure, bullet style, and academic tone. "
                        "The LLM uses this to generate the deck JSON before building the `.pptx` file.")
            gr.Markdown("### 🎛️ PPTX LLM Parameters")
            pptx_ctx_slider = gr.Slider(minimum=1024, maximum=16384, step=1024, value=10240, label="Context Window (n_ctx)")
            pptx_batch_slider = gr.Slider(minimum=256, maximum=4096, step=256, value=1024, label="Reading Speed (n_batch)")
            pptx_temp_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.3, label="Creativity (Temperature)")
        with gr.Row():
            generate_pptx_btn = gr.Button("📊 Generate Custom PowerPoint Deck (.pptx)", variant="secondary", size="lg")
            stop_pptx_btn = gr.Button("⏹️ Stop", variant="stop")
        with gr.Row():
            pptx_status = gr.Textbox(label="PowerPoint Status", interactive=False)
            pptx_download = gr.File(label="Download PowerPoint Deck", interactive=False)

        # --- Step 2 Event Wiring ---
        script_event = generate_btn.click(
            fn=step1_generate_script,
            inputs=[
                pdf_input, 
                selected_pages_state, 
                ctx_slider, 
                batch_slider, 
                temp_slider, 
                system_prompt_box,
                focus_areas,
                llm_model_input,
                long_doc_mode
            ],
            outputs=[status_box, script_output]
        )
        def _stop_script_generation():
            _cleanup_subprocess("script_llm")
            return "⏹️ Stopped. LLM subprocess killed."
        stop_script_btn.click(fn=_stop_script_generation, inputs=[], outputs=[status_box], cancels=[script_event])
        
        fetch_btn.click(
            fn=step2_fetch_assets, 
            inputs=script_output, 
            outputs=asset_gallery
        ).then(
            fn=update_override_ui, 
            inputs=[script_output, pdf_images_state], 
            outputs=[segment_dropdown, override_gallery]
        )
        
        override_gallery.select(
            fn=handle_gallery_click, 
            inputs=[segment_dropdown, manual_selections_state, pdf_images_state, pdf_fullres_state, asset_gallery], 
            outputs=[manual_selections_state, override_status, asset_gallery]
        )
        
        clear_overrides_btn.click(
            fn=lambda: ({}, "🗑️ Overrides cleared. Pipeline will use auto-retrieved assets."), 
            inputs=[], 
            outputs=[manual_selections_state, override_status]
        )

        refresh_override_btn.click(
            fn=update_override_ui,
            inputs=[script_output, pdf_images_state],
            outputs=[segment_dropdown, override_gallery]
        )

        pptx_event = generate_pptx_btn.click(
            fn=generate_pptx_deck,
            inputs=[script_output, asset_gallery, pdf_input, pptx_ctx_slider, pptx_batch_slider, pptx_temp_slider, manual_selections_state, llm_model_input, pptx_prompt_box],
            outputs=[pptx_status, pptx_download]
        )
        def _stop_pptx_generation():
            _cleanup_subprocess("pptx_llm")
            return "⏹️ Stopped. LLM subprocess killed."
        stop_pptx_btn.click(fn=_stop_pptx_generation, inputs=[], outputs=[pptx_status], cancels=[pptx_event])

    # ══════════════════════════════════════════════
    # STEP 3: Audio & Video Synthesis
    # ══════════════════════════════════════════════
    with gr.Tab("Step 3: Audio & Video Synthesis"):
        with gr.Row():
            with gr.Column():
                zero_shot_audio = gr.Audio(label="🎙️ Upload Reference Audio (8-12 Seconds recommended)", type="filepath")
                avatar_image = gr.Image(label="👤 Instructor Avatar (Headshot and Square Dimensions recommended)", type="filepath", height=300)
                
                with gr.Accordion("🎬 Presentation Mode", open=True):
                    gr.Markdown("Choose how visual backgrounds are rendered for each segment.")
                    presentation_mode_radio = gr.Radio(
                        choices=["Classic (Avatar + Assets + B-Roll)", "Presentation Mode (PPTX Slides + Avatar PiP)"],
                        value="Classic (Avatar + Assets + B-Roll)",
                        label="Video Style",
                        info="Classic: full-screen avatar, retrieved assets, and B-roll. Presentation: auto-generates PowerPoint slides as backgrounds with avatar PiP overlay."
                    )

                
                with gr.Accordion("🖼️ PiP Layout & Resolution", open=False):
                    gr.Markdown("**Picture-in-Picture** composites your avatar over retrieved assets and B-roll for a professional layout.")
                    pip_toggle = gr.Checkbox(label="Enable PiP Layout (avatar in upper-right corner + assets/B-roll fullscreen)", value=True)
                    canvas_dropdown = gr.Dropdown(
                        choices=["720p (1280x720)", "1080p (1920x1080)"],
                        value="720p (1280x720)",
                        label="Canvas Resolution"
                    )
                    gr.Markdown("💡 **Tip**: 720p is faster to render. 1080p gives sharper assets. Avatar stays at native 512x512 internally.")
                    gr.Markdown("\n**B-Roll Assets**: Place stock footage in `broll_assets/` folder. Name files descriptively "
                                "(e.g., `city_traffic.mp4`, `courtroom_gavel.jpg`). The pipeline matches LLM queries to filenames.")
                    pexels_key_input = gr.Textbox(
                        label="🎬 Pexels API Key (optional — enables auto B-roll download)",
                        placeholder="Paste your free Pexels API key here (https://www.pexels.com/api/)",
                        type="password",
                        value=""
                    )
                    gr.Markdown("💡 **Pexels**: When local B-roll isn't found, the pipeline auto-downloads matching stock footage. "
                                "Downloaded clips are cached in `broll_assets/` for reuse. Get a free key at [pexels.com/api](https://www.pexels.com/api/).")

                with gr.Accordion("🎙️ Speech & Breathing Settings", open=False):
                    gr.Markdown("Control speech pacing and breathing pauses for natural narration.")
                    tts_engine_radio = gr.Radio(
                        choices=list_tts_backend_names(),
                        value=TTS_BACKEND_SPECS[DEFAULT_TTS_BACKEND].name,
                        label="TTS Engine",
                        info="Adding a new TTS backend = subclass TTSBackend in tts_backends.py + register. No pipeline edits."
                    )
                    melotts_ckpt_input = gr.Textbox(
                        label="MeloTTS / OpenVoice Checkpoint Root",
                        placeholder="checkpoints_v2",
                        value="checkpoints_v2",
                        info="Folder containing converter/ and base_speakers/ subdirectories. Only used when MeloTTS is selected."
                    )
                    tts_speed = gr.Slider(minimum=0.7, maximum=1.2, step=0.05, value=0.95, label="Speech Speed (0.7=very slow & natural, 1.0=normal, 1.2=fast)")
                    breathing_toggle = gr.Checkbox(label="Enable Breathing Pauses", value=True, info="Adds natural breathing pauses and punctuation smoothing to narration text. Disable for raw script text.")
                    

                with gr.Accordion("📝 Subtitle Settings", open=False):
                    subtitle_toggle = gr.Checkbox(label="Enable Subtitles", value=True)
                    gr.Markdown("💡 Disabling subtitles gives a cleaner look.")

                with gr.Accordion("🤖 Avatar Model Settings", open=False):
                    gr.Markdown("Pick a talking-head backend, then point at any compatible repo. "
                                "Adding a new backend means adding ~6 lines to `avatar_backends.py` — the "
                                "pipeline plumbing (subprocess isolation, progress streaming) is shared.")
                    avatar_backend_dropdown = gr.Dropdown(
                        choices=list_avatar_backend_names(),
                        value=AVATAR_BACKEND_SPECS[DEFAULT_AVATAR_BACKEND].name,
                        label="Avatar Backend",
                        info="Talking-head model adapter. Defaults to SoulX-FlashHead Lite."
                    )
                    avatar_root_input = gr.Textbox(
                        value=DEFAULT_AVATAR_ROOT,
                        label="Avatar Model Root",
                        placeholder=r"C:\path\to\avatar-model-repo",
                        info="Root directory of the avatar generation repository (contains the backend's entry script + models/)."
                    )
                    avatar_python_input = gr.Textbox(
                        value=DEFAULT_AVATAR_ENV_PYTHON,
                        label="Avatar Python Executable",
                        placeholder=r"C:\path\to\avatar_venv\Scripts\python.exe",
                        info="Python interpreter with the avatar model's dependencies installed."
                    )

                render_btn = gr.Button(" Render Audio & Timeline Map", variant="primary", size="lg")
                cancel_render_btn = gr.Button("⏹️ Cancel Render", variant="stop")
                render_status = gr.Textbox(label="Compiler Status", interactive=False)

                with gr.Accordion("🚀 One-Click Process All", open=False):
                    gr.Markdown(
                        "Run the **entire pipeline** with one click: Build Database → Generate Script → "
                        "Fetch Assets → Render Video → Run Evaluation.\n\n"
                        "**Requirements**: PDF uploaded in Step 1, reference audio and avatar photo above."
                    )
                    with gr.Accordion("🔀 Cross-Model Eval (during Process All)", open=False):
                        gr.Markdown(
                            "Enable cross-model evaluation as part of the pipeline. "
                            "An independent cloud LLM (Gemini/GPT-4o) re-evaluates the output to eliminate shared-weight bias."
                        )
                        cross_auto_toggle = gr.Checkbox(
                            label="🔀 Enable cross-model eval in Process All",
                            value=False,
                        )
                        with gr.Row():
                            cross_auto_provider = gr.Radio(
                                choices=["Gemini (Free)", "OpenAI (Paid)"],
                                value="Gemini (Free)", label="Provider"
                            )
                            cross_auto_api_key = gr.Textbox(
                                label="API Key", type="password",
                                placeholder="AIza... (Gemini) or sk-... (OpenAI)",
                                info="Get a free Gemini key at aistudio.google.com/apikey"
                            )
                        with gr.Row():
                            cross_auto_cold = gr.Checkbox(label="🧊 Cold Quiz", value=True)
                            cross_auto_comp = gr.Checkbox(label="📖 Comprehension Quiz", value=True)
                            cross_auto_mos = gr.Checkbox(label="⭐ Cross-Model MOS", value=True)
                    auto_process_btn = gr.Button("🚀 Process All (Full Pipeline)", variant="primary", size="lg")
                    cancel_auto_btn = gr.Button("⏹️ Cancel Process", variant="stop")
                    auto_status = gr.Textbox(label="Pipeline Progress", interactive=False, lines=6)

            with gr.Column():
                video_output = gr.Video(label="Rendered Educational Presentation")
                
        render_event = render_btn.click(
            fn=step3_render_video, 
            inputs=[
                script_output, zero_shot_audio, asset_gallery, avatar_image,
                manual_selections_state, tts_speed, pip_toggle, canvas_dropdown, pexels_key_input,
                presentation_mode_radio, subtitle_toggle, tts_engine_radio, breathing_toggle,
                avatar_root_input, avatar_python_input, melotts_ckpt_input, avatar_backend_dropdown,
            ],  
            outputs=[render_status, video_output]
        )

        def _cancel_render_fn():
            _cancel_render.set()
            return "⏹️ Cancellation requested — stopping after current segment..."
        cancel_render_btn.click(fn=_cancel_render_fn, inputs=[], outputs=[render_status], cancels=[render_event])

    # ══════════════════════════════════════════════
    # STEP 4: Automated Evaluation
    # ══════════════════════════════════════════════
    with gr.Tab("Step 4: Evaluation & Metrics"):
        gr.Markdown("# 📊 Automated AI Evaluation Module")
        gr.Markdown("Run comprehensive metrics on your generated video, script, slides, and voice cloning quality. "
                    "Results include BERTScore, ROUGE-L, SSIM, Voice Similarity, LLM-based MOS, and a PresentQuiz.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Evaluation Settings")
                gr.Markdown("The evaluation uses your **PDF** (from Step 1), **Script** (from Step 2), and **Reference Audio** (from Step 3) automatically.")
                with gr.Accordion("📋 Select Metrics to Run", open=True):
                    eval_script_cb = gr.Checkbox(label="📝 Script Fidelity (BERTScore + ROUGE-L)", value=True)
                    eval_slide_cb = gr.Checkbox(label="🖼️ Slide Visual Fidelity (SSIM)", value=True)
                    eval_voice_cb = gr.Checkbox(label="🎙️ Voice Cloning Similarity (Resemblyzer)", value=True)
                    eval_efficiency_cb = gr.Checkbox(label="⚡ Generation Efficiency (Time + VRAM)", value=True)
                    eval_quiz_cb = gr.Checkbox(label="🧠 PresentQuiz (LLM-generated MCQ)", value=True)
                    eval_mos_cb = gr.Checkbox(label="⭐ PresentArena MOS (LLM-based Likert)", value=True)

                eval_run_btn = gr.Button(" Run Evaluation", variant="primary", size="lg")
                eval_status = gr.Textbox(label="Evaluation Progress", interactive=False, lines=12)

            with gr.Column(scale=2):
                gr.Markdown("### 📈 Results & Charts")
                with gr.Tab("Summary"):
                    eval_metrics_chart = gr.Plot(label="Metrics Summary Bar Chart")
                    eval_results_json = gr.JSON(label="Full Evaluation Results")
                with gr.Tab("MOS Radar"):
                    eval_radar_chart = gr.Plot(label="PresentArena MOS Radar Chart")
                    eval_mos_detail = gr.JSON(label="MOS Justifications")
                with gr.Tab("Voice Similarity"):
                    eval_voice_chart = gr.Plot(label="Per-Segment Voice Similarity")
                with gr.Tab("PresentQuiz"):
                    eval_quiz_chart = gr.Plot(label="Quiz Difficulty Distribution")
                    eval_quiz_display = gr.JSON(label="Generated Quiz Questions")
                with gr.Tab("Efficiency"):
                    eval_efficiency_display = gr.JSON(label="Render Efficiency Metrics")
                with gr.Tab("🔀 Cross-Model Eval"):
                    gr.Markdown("### Cross-Model Evaluation (Gemini / GPT-4o)")
                    gr.Markdown(
                        "Run an independent cloud LLM evaluation to eliminate shared-weight bias. "
                        "The cross-model evaluator attempts the same PresentQuiz and re-scores MOS on the narration. "
                        "**Gemini** (free via [Google AI Studio](https://aistudio.google.com/apikey)) is the default."
                    )
                    with gr.Row():
                        cross_provider = gr.Radio(
                            choices=["Gemini (Free)", "OpenAI (Paid)"],
                            value="Gemini (Free)", label="Provider"
                        )
                        cross_api_key = gr.Textbox(
                            label="API Key", type="password",
                            placeholder="AIza... (Gemini) or sk-... (OpenAI)",
                            info="Get a free Gemini key at aistudio.google.com/apikey"
                        )
                    with gr.Row():
                        cross_cold_cb = gr.Checkbox(label="🧊 Cold Quiz (no narration)", value=True)
                        cross_comp_cb = gr.Checkbox(label="📖 Comprehension Quiz", value=True)
                        cross_mos_cb = gr.Checkbox(label="⭐ Cross-Model MOS", value=True)
                    with gr.Row():
                        cross_mode = gr.Radio(
                            choices=["Latest Report Only", "All Reports (Batch)"],
                            value="Latest Report Only", label="Scope"
                        )
                    cross_run_btn = gr.Button("🔀 Run Cross-Model Evaluation", variant="primary")
                    cross_status = gr.Textbox(label="Cross-Eval Progress", interactive=False, lines=6)
                    cross_results_json = gr.JSON(label="Cross-Model Results")

        # --- Cross-Model Eval Logic ---
        def _run_cross_eval(api_key, cold, comp, mos, mode, provider_choice):
            if not api_key or not api_key.strip():
                yield "❌ Please enter your API key.", None
                return

            provider = "openai" if "OpenAI" in provider_choice else "gemini"

            import cross_model_eval

            reports_dir = os.path.abspath("evaluation_reports")
            if not os.path.isdir(reports_dir):
                yield "❌ No evaluation_reports/ directory found. Run standard evaluation first.", None
                return

            report_dir = None
            if mode == "Latest Report Only":
                # Find the most recently modified report folder
                subdirs = [
                    os.path.join(reports_dir, d)
                    for d in os.listdir(reports_dir)
                    if os.path.isdir(os.path.join(reports_dir, d))
                ]
                if not subdirs:
                    yield "❌ No report folders found.", None
                    return
                report_dir = max(subdirs, key=os.path.getmtime)
                yield f"🔀 [{provider}] Evaluating latest report: {os.path.basename(report_dir)}...", None
            else:
                yield f"🔀 [{provider}] Batch evaluating all reports in {reports_dir}...", None

            try:
                all_results, output_path = cross_model_eval.run_cross_eval(
                    api_key=api_key.strip(),
                    report_dir=report_dir,
                    run_cold=cold,
                    run_comprehension=comp,
                    run_mos=mos,
                    provider=provider,
                )
                summary_lines = ["✅ Cross-model evaluation complete!\n"]
                for r in all_results:
                    name = r.get("report", "?")
                    gemma = r.get("gemma_self_eval", {}).get("percentage", "N/A")
                    cold_res = r.get("cross_quiz_cold", {})
                    comp_res = r.get("cross_quiz_comprehension", {})
                    cold_pct = cold_res.get("percentage", "—") if isinstance(cold_res, dict) and "error" not in cold_res and "skipped" not in cold_res else "—"
                    comp_pct = comp_res.get("percentage", "—") if isinstance(comp_res, dict) and "error" not in comp_res and "skipped" not in comp_res else "—"
                    summary_lines.append(f"📋 {name}")
                    summary_lines.append(f"   Gemma self-eval: {gemma}% | Cold: {cold_pct}% | Comprehension: {comp_pct}%")

                    cross_mos = r.get("cross_mos", {})
                    if isinstance(cross_mos, dict) and "scores" in cross_mos:
                        gemma_mos = r.get("gemma_mos", {})
                        for aspect in ["accuracy", "clarity", "engagement", "visual_appeal", "educational_value"]:
                            g_val = gemma_mos.get(aspect, "?")
                            c_val = cross_mos["scores"].get(aspect, "?")
                            summary_lines.append(f"   MOS {aspect}: Gemma={g_val} → {cross_mos.get('model', provider)}={c_val}")
                    summary_lines.append("")

                summary_lines.append(f"💾 Saved to: {output_path}")
                yield "\n".join(summary_lines), all_results
            except Exception as e:
                yield f"❌ Error: {e}", None

        cross_run_btn.click(
            fn=_run_cross_eval,
            inputs=[cross_api_key, cross_cold_cb, cross_comp_cb, cross_mos_cb, cross_mode, cross_provider],
            outputs=[cross_status, cross_results_json],
        )

        # --- Step 4 Evaluation Logic ---
        def _run_evaluation(script_json, pdf_file, ref_audio,
                            do_script, do_slide, do_voice, do_efficiency, do_quiz, do_mos,
                            script_prompt="", pptx_prompt="",
                            ctx_val=4096, batch_val=512, temp_val=0.4, llm_model_val=""):
            """Orchestrator that calls eval_engine and builds charts."""
            import plotly.graph_objects as go

            pdf_path = pdf_file.name if pdf_file else None
            ref_audio_path = ref_audio if ref_audio else ""

            status_lines = []
            results = {}

            # Run each metric
            gen = eval_engine.run_full_evaluation(
                script_json=script_json or "",
                pdf_path=pdf_path or "",
                reference_audio_path=ref_audio_path,
                llm_model_path="",
                run_script_fidelity=(do_script and pdf_path is not None),
                run_slide_fidelity=(do_slide and pdf_path is not None),
                run_voice_similarity=(do_voice and bool(ref_audio_path)),
                run_efficiency=do_efficiency,
                run_quiz=do_quiz,
                run_mos=do_mos,
            )

            for msg in gen:
                if msg.startswith("RESULTS:"):
                    results = json.loads(msg[8:])
                else:
                    status_lines.append(msg)
                    yield (
                        "\n".join(status_lines),
                        None, None, None, None, None, None, None, None
                    )

            # Build charts
            status_lines.append("\n📊 Generating charts...")
            yield ("\n".join(status_lines), None, None, None, None, None, None, None, None)

            metrics_chart = eval_engine.generate_metrics_bar_chart(results)
            radar_chart = eval_engine.generate_radar_chart(results.get("mos_scores", {}))
            voice_chart = eval_engine.generate_voice_similarity_chart(results.get("voice_similarity", {}))
            quiz_chart = eval_engine.generate_quiz_chart(results.get("quiz", {}))

            mos_detail = results.get("mos_scores", {})
            quiz_display = results.get("quiz", {})
            efficiency_display = results.get("efficiency", {})

            # Save evaluation report to disk
            try:
                charts_dict = {
                    "metrics_summary": metrics_chart,
                    "mos_radar": radar_chart,
                    "voice_similarity": voice_chart,
                    "quiz_distribution": quiz_chart,
                }
                gen_settings = {
                    "n_ctx": ctx_val,
                    "n_batch": batch_val,
                    "temperature": temp_val,
                    "llm_model": llm_model_val,
                }
                report_dir = eval_engine.save_evaluation_report(
                    results, charts_dict,
                    script_prompt=script_prompt,
                    pptx_prompt=pptx_prompt,
                    generation_settings=gen_settings,
                    script_json=script_json,
                )
                status_lines.append(f"💾 Report saved to: {report_dir}")
            except Exception as e:
                status_lines.append(f"⚠️ Could not save report: {e}")

            status_lines.append("✅ Evaluation complete!")
            final_status = "\n".join(status_lines)

            yield (
                final_status,
                metrics_chart,
                results,
                radar_chart,
                mos_detail,
                voice_chart,
                quiz_chart,
                quiz_display,
                efficiency_display,
            )

        eval_run_btn.click(
            fn=_run_evaluation,
            inputs=[
                script_output, pdf_input, zero_shot_audio,
                eval_script_cb, eval_slide_cb, eval_voice_cb,
                eval_efficiency_cb, eval_quiz_cb, eval_mos_cb,
                system_prompt_box, pptx_prompt_box,
                ctx_slider, batch_slider, temp_slider, llm_model_input,
            ],
            outputs=[
                eval_status,
                eval_metrics_chart,
                eval_results_json,
                eval_radar_chart,
                eval_mos_detail,
                eval_voice_chart,
                eval_quiz_chart,
                eval_quiz_display,
                eval_efficiency_display,
            ],
        )

    # --- One-Click Process All Wiring (after all tabs so every component exists) ---
    auto_event = auto_process_btn.click(
        fn=auto_process_all,
        inputs=[
            pdf_input, selected_pages_state,
            ctx_slider, batch_slider, temp_slider, system_prompt_box, focus_areas, llm_model_input, long_doc_mode,
            zero_shot_audio, avatar_image, manual_selections_state,
            tts_speed, pip_toggle, canvas_dropdown, pexels_key_input, presentation_mode_radio, subtitle_toggle,
            tts_engine_radio, breathing_toggle,
            avatar_root_input, avatar_python_input, melotts_ckpt_input, avatar_backend_dropdown,
            pptx_prompt_box,
            eval_script_cb, eval_slide_cb, eval_voice_cb,
            eval_efficiency_cb, eval_quiz_cb, eval_mos_cb,
            cross_auto_toggle, cross_auto_api_key, cross_auto_provider,
            cross_auto_cold, cross_auto_comp, cross_auto_mos,
        ],
        outputs=[
            auto_status, script_output, asset_gallery, video_output,
            eval_status, eval_metrics_chart, eval_results_json,
            eval_radar_chart, eval_mos_detail, eval_voice_chart, eval_quiz_chart,
            eval_quiz_display, eval_efficiency_display,
        ],
    )
    def _cancel_auto():
        _cancel_autoprocess.set()
        return "⏹️ Cancellation requested — stopping after current stage..."
    cancel_auto_btn.click(fn=_cancel_auto, inputs=[], outputs=[auto_status], cancels=[auto_event])
        
if __name__ == "__main__":
    dashboard.launch(
        share=False, 
        inbrowser=True, 
        theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"])
    )
