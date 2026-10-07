"""
CLI entry point for FactorioLLM blueprint generation.
Allows generating blueprints directly from terminal or starting the Gradio Web UI.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# Ensure UTF-8 output in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from factoriollm.agentic_loop import AgenticBlueprintGenerator
from factoriollm.inference import FactorioInference
from factoriollm.sandbox import DockerSandbox


def run_cli_generation(
    prompt: str,
    max_retries: int = 3,
    temperature: float = 0.2,
    save_code_path: Optional[str] = None,
    save_blueprint_path: Optional[str] = None,
):
    print("\n" + "=" * 60)
    print("🏭 FactorioLLM Generator")
    print(f"📝 Prompt: {prompt}")
    print(f"⚙️ Max retries: {max_retries} | Temperature: {temperature}")
    print("=" * 60 + "\n")

    sandbox = DockerSandbox()
    inference = FactorioInference()
    agent = AgenticBlueprintGenerator(inference=inference, sandbox=sandbox, max_retries=max_retries)

    print("[*] Launching agentic generation loop...")
    stream = agent.generate_blueprint_stream(prompt, max_retries=max_retries, temperature=temperature)

    final_result = None
    try:
        while True:
            event = next(stream)
            if event.event_type == "info":
                print(f"[*] {event.message}")
            elif event.event_type == "code":
                print(f"\n--- [Attempt {event.attempt}] Generated Code ---")
                print(event.code[:400] + ("\n... [truncated] ..." if len(event.code) > 400 else ""))
                print("------------------------------------------")
            elif event.event_type == "sandbox_start":
                print(f"[*] Running in Docker sandbox (timeout {sandbox.timeout}s)...")
            elif event.event_type == "retry":
                print(f"[!] Execution failed. Triggering self-healing recovery...")
                if event.execution_result and event.execution_result.error_message:
                    print(f"    Error: {event.execution_result.error_message}")
            elif event.event_type == "success":
                print(f"\n[+] {event.message}")
            elif event.event_type == "failure":
                print(f"\n[-] {event.message}")
    except StopIteration as e:
        final_result = e.value

    if final_result and final_result.success:
        print("\n" + "=" * 60)
        print("🎉 BLUEPRINT COMPILED AND VALIDATED SUCCESSFULLY!")
        print("=" * 60)
        print("\n--- IN-GAME BLUEPRINT STRING (Ctrl+V) ---")
        print(final_result.blueprint_string)
        print("----------------------------------------\n")

        if save_code_path and final_result.code:
            with open(save_code_path, "w", encoding="utf-8") as f:
                f.write(final_result.code)
            print(f"[*] Python script saved to: {save_code_path}")

        if save_blueprint_path and final_result.blueprint_string:
            with open(save_blueprint_path, "w", encoding="utf-8") as f:
                f.write(final_result.blueprint_string)
            print(f"[*] Blueprint string saved to: {save_blueprint_path}")

        # Try copying to Windows clipboard
        try:
            import subprocess
            subprocess.run(
                ["clip"],
                input=final_result.blueprint_string.strip().encode("utf-8"),
                check=False,
            )
            print("[*] Blueprint string automatically copied to Windows clipboard!")
        except Exception:
            pass

    else:
        print("\n" + "!" * 60)
        print("[-] Failed to generate a valid blueprint after all attempts.")
        if final_result and final_result.error_message:
            print(f"    Last error: {final_result.error_message}")
        print("!" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FactorioLLM: Text-to-Blueprint Generator")
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Blueprint description (English or Russian)",
    )
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Launch the Gradio Web Studio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Gradio Web Studio (default: 7860)",
    )
    parser.add_argument(
        "--retries", "-r",
        type=int,
        default=3,
        help="Maximum self-healing retry attempts (default: 3)",
    )
    parser.add_argument(
        "--temp", "-t",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--save-code",
        type=str,
        help="File path to save the generated Python script",
    )
    parser.add_argument(
        "--save-bp",
        type=str,
        help="File path to save the generated blueprint string",
    )

    args = parser.parse_args()

    if args.web or (len(sys.argv) == 1 and not args.prompt):
        # If launched with --web or without arguments, start Gradio Web UI
        from factoriollm.app import main as launch_web
        launch_web(port=args.port)
    else:
        prompt = args.prompt
        if not prompt:
            prompt = input("Enter Factorio blueprint description: ").strip()
            if not prompt:
                print("[!] Description cannot be empty.")
                return

        run_cli_generation(
            prompt=prompt,
            max_retries=args.retries,
            temperature=args.temp,
            save_code_path=args.save_code,
            save_blueprint_path=args.save_bp,
        )


if __name__ == "__main__":
    main()
