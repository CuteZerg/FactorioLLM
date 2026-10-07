"""
FactorioLLM Gradio Web Application.
Interactive UI for text-to-blueprint generation with live agentic self-healing logs.
"""

from __future__ import annotations

import html
from typing import Generator

import gradio as gr

from factoriollm.agentic_loop import AgenticBlueprintGenerator
from factoriollm.inference import FactorioInference
from factoriollm.sandbox import DockerSandbox


# Singleton instances for UI session
_inference = FactorioInference(max_seq_length=4096)
_sandbox = DockerSandbox()
_agent = AgenticBlueprintGenerator(inference=_inference, sandbox=_sandbox, max_retries=3)


CUSTOM_CSS = """
/* Factorio-inspired industrial dark theme */
:root {
    --factorio-orange: #e58c2b;
    --factorio-amber: #ffaa33;
    --factorio-dark: #1a1c1e;
    --factorio-slate: #24272c;
    --factorio-border: #3c4048;
}

body, .gradio-container {
    background-color: var(--factorio-dark) !important;
    color: #e0e2e5 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.title-header {
    text-align: center;
    padding: 18px;
    border-bottom: 2px solid var(--factorio-orange);
    background: linear-gradient(180deg, #282c34 0%, #1a1c1e 100%);
    border-radius: 8px;
    margin-bottom: 16px;
}

.title-header h1 {
    color: var(--factorio-amber);
    margin: 0;
    font-size: 2.2rem;
    font-weight: bold;
    letter-spacing: 1px;
}

.title-header p {
    color: #9da5b4;
    margin-top: 6px;
    font-size: 1.05rem;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 2px 4px;
    background: #2d3139;
    border: 1px solid var(--factorio-border);
    color: var(--factorio-orange);
}

.status-box {
    padding: 12px 16px;
    border-radius: 6px;
    background-color: var(--factorio-slate);
    border-left: 4px solid var(--factorio-orange);
    font-family: monospace;
    font-size: 0.95rem;
    margin-bottom: 12px;
}

/* Ensure the log output container is easily scrollable */
.log-box textarea {
    max-height: 420px !important;
    overflow-y: scroll !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.88rem !important;
    white-space: pre !important;
}
"""


def process_generation_stream(
    prompt: str,
    max_retries: int,
    temperature: float,
    timeout: float,
) -> Generator[tuple[str, str, str, str], None, None]:
    """
    Generator yielding live updates to the Gradio UI:
    Returns (status_html, generated_code, blueprint_string, full_log)
    """
    if not prompt or not prompt.strip():
        yield (
            "<div class='status-box'>⚠️ Please enter a blueprint description.</div>",
            "",
            "",
            "Empty prompt.",
        )
        return

    _sandbox.timeout = float(timeout)
    _agent.max_retries = int(max_retries)

    current_code = ""
    blueprint_string = ""
    current_status = "⚙️ Initializing generation pipeline..."

    def make_status_html(msg: str, is_error: bool = False, is_success: bool = False) -> str:
        border_color = "#e58c2b"
        if is_error:
            border_color = "#e06c75"
        elif is_success:
            border_color = "#98c379"
        return f"<div class='status-box' style='border-left-color: {border_color};'>{html.escape(msg)}</div>"

    yield make_status_html(current_status), "", "", ""

    log_accumulator = []

    try:
        gen = _agent.generate_blueprint_stream(
            prompt=prompt.strip(),
            max_retries=int(max_retries),
            temperature=float(temperature),
        )

        for event in gen:
            if event.event_type == "info":
                current_status = f"ℹ️ {event.message}"
                log_accumulator.append(f"[{event.attempt}] INFO: {event.message}")
                if "Attempt " in event.message and "Generating" in event.message:
                    current_code = ""
            elif event.event_type == "token":
                current_code += event.message
            elif event.event_type == "code":
                current_code = event.code
                current_status = f"📝 Attempt {event.attempt}: {event.message}"
                log_accumulator.append(f"[{event.attempt}] CODE:\n{event.code}\n")
            elif event.event_type == "sandbox_start":
                current_status = f"🐳 Attempt {event.attempt}: Running inside isolated Docker sandbox..."
                log_accumulator.append(f"[{event.attempt}] DOCKER: {event.message}")
            elif event.event_type == "retry":
                current_status = f"⚠️ Attempt {event.attempt}: {event.message}"
                if event.execution_result:
                    feedback = event.execution_result.get_feedback_for_model()
                    if feedback:
                        log_accumulator.append(f"[{event.attempt}] ERROR:\n{feedback}\n")
                    else:
                        log_accumulator.append(f"[{event.attempt}] RETRY: {event.message}\n")
            elif event.event_type == "success":
                current_status = f"✅ Success! Blueprint compiled and validated (attempt {event.attempt})."
                blueprint_string = event.blueprint_string or ""
                log_accumulator.append(f"[{event.attempt}] SUCCESS: Blueprint verified successfully!")
            elif event.event_type == "failure":
                current_status = f"❌ Failed to generate a valid blueprint after {event.attempt} attempts."
                log_accumulator.append(f"[{event.attempt}] FAILED: Retries exhausted.")

            full_log = "\n".join(log_accumulator)
            yield (
                make_status_html(
                    current_status,
                    is_error=event.event_type in ("retry", "failure"),
                    is_success=event.event_type == "success",
                ),
                current_code,
                blueprint_string,
                full_log,
            )

    except Exception as e:
        err_msg = f"Fatal error: {str(e)}"
        log_accumulator.append(f"EXCEPTION: {err_msg}")
        yield (
            make_status_html(err_msg, is_error=True),
            current_code,
            "",
            "\n".join(log_accumulator),
        )


def build_app() -> gr.Blocks:
    """Builds and configures the Gradio Blocks application."""
    with gr.Blocks(title="FactorioLLM - Text to Blueprint Studio") as demo:
        with gr.Column(elem_classes=["title-header"]):
            gr.Markdown(
                """
                # ⚙️ FactorioLLM: Text-to-Blueprint Studio
                Generate functional Factorio blueprints from natural language using Python (`factorio-draftsman`) with an isolated Docker sandbox and an agentic self-healing loop.
                """
            )
            gr.Markdown(
                """
                <span class='badge'>Llama-3-8B-Instruct (4-bit LoRA)</span>
                <span class='badge'>Docker Sandbox Isolation</span>
                <span class='badge'>Self-Healing Agentic Loop</span>
                <span class='badge'>Factorio v2.0 Ready</span>
                """
            )

        with gr.Row():
            # Left Column: User Input and Controls
            with gr.Column(scale=5):
                prompt_input = gr.Textbox(
                    label="Blueprint Description (English or Russian):",
                    placeholder="e.g. Create a compact copper smelting line for 24 stone furnaces using red belts and inserters",
                    lines=4,
                    autofocus=True,
                )

                with gr.Accordion("⚙️ Agentic Loop Settings", open=False):
                    with gr.Row():
                        retries_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Max Retries",
                            info="Number of error recovery iterations via Docker tracebacks",
                        )
                        temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.05,
                            label="Temperature",
                            info="Lower = stricter and more deterministic code",
                        )
                        timeout_slider = gr.Slider(
                            minimum=5.0,
                            maximum=30.0,
                            value=15.0,
                            step=1.0,
                            label="Sandbox Timeout (s)",
                            info="Execution time limit inside Docker container",
                        )

                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate Blueprint", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

                gr.Markdown("### 💡 Example Prompts:")
                examples = [
                    ["Create an expandable 4x4 belt balancer using transport belts and splitters"],
                    ["Create a compact copper smelting line for 24 stone furnaces using red belts and inserters"],
                    ["Solar farm with solar panels and accumulators arranged around medium electric poles"],
                    ["Circuit edge detector with decider combinator, arithmetic combinator, and constant combinator"],
                    ["Линия выплавки меди на 24 каменные печи с красными конвейерами"],
                ]
                gr.Examples(
                    examples=examples,
                    inputs=[prompt_input],
                    label="Click an example to populate the prompt field",
                )

            # Right Column: Live Output and Blueprint String
            with gr.Column(scale=6):
                status_output = gr.HTML(
                    value="<div class='status-box'>Ready. Enter a prompt to generate a blueprint.</div>"
                )

                with gr.Tabs():
                    with gr.TabItem("📋 Factorio Blueprint String (In-Game)"):
                        blueprint_output = gr.Textbox(
                            label="Blueprint String (paste directly into Factorio with Ctrl+V):",
                            lines=8,
                            buttons=["copy"],
                            placeholder="Compiled blueprint string (0e...) will appear here",
                        )
                        gr.Markdown(
                            "> **How to use in game:** Click the copy button in the top right of the field, launch Factorio, and press `Ctrl + V` anywhere on the map."
                        )

                    with gr.TabItem("🐍 Python Draftsman Script"):
                        code_output = gr.Code(
                            label="Generated Python Draftsman Script",
                            language="python",
                            lines=12,
                            buttons=["copy"],
                        )

                    with gr.TabItem("🔍 Execution & Self-Healing Log"):
                        log_output = gr.Textbox(
                            label="Detailed Execution Log",
                            lines=14,
                            autoscroll=True,
                            elem_classes=["log-box"],
                        )

        # Event Handlers
        generate_btn.click(
            fn=process_generation_stream,
            inputs=[prompt_input, retries_slider, temp_slider, timeout_slider],
            outputs=[status_output, code_output, blueprint_output, log_output],
        )
        prompt_input.submit(
            fn=process_generation_stream,
            inputs=[prompt_input, retries_slider, temp_slider, timeout_slider],
            outputs=[status_output, code_output, blueprint_output, log_output],
        )

        def clear_fields():
            return "", "<div class='status-box'>Ready. Enter a prompt to generate a blueprint.</div>", "", "", ""

        clear_btn.click(
            fn=clear_fields,
            inputs=[],
            outputs=[prompt_input, status_output, code_output, blueprint_output, log_output],
        )

    return demo


def main(port: int = 7860):
    demo = build_app()
    theme = gr.themes.Default(
        primary_hue="amber",
        secondary_hue="slate",
        neutral_hue="slate",
    )
    print(f"[*] Launching FactorioLLM Web Studio on http://127.0.0.1:{port} ...")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=theme,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
