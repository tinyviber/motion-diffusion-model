import argparse
from pathlib import Path
from typing import List, Optional

import gradio as gr

from web_app.inference import MotionGenerator


class WebUI:
    def __init__(self, model_path: Optional[str], guidance: Optional[float]) -> None:
        self.generator = MotionGenerator(model_path=model_path, guidance_param=guidance)

    def run(self, prompt: str, num_repetitions: int, output_format: str):
        result = self.generator.generate(prompt, num_repetitions, output_format)
        status = result["message"]
        animations: List[Path] = result.get("animations") or []
        json_file: Optional[Path] = result.get("json_file")
        # Gradio expects strings or file-like paths
        return status, [str(path) for path in animations], str(json_file) if json_file else None


DESCRIPTION = """## Motion Diffusion Model - Web Demo

Use this page to run the text-to-motion generator without Cog. Provide a prompt, choose
how many variations you would like, and select whether to receive rendered animations
or a JSON export compatible with Autodesk HumanIK.
"""


def build_demo(model_path: Optional[str], guidance: Optional[float]):
    ui = WebUI(model_path, guidance)
    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="the person walked forward and is picking up his toolbox.",
                lines=2,
            )
            num_repetitions = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=3,
                label="Number of Samples",
            )
        output_format = gr.Radio(
            ["animation", "json_file"],
            value="animation",
            label="Output format",
            info="Animations are saved as MP4 files; JSON returns HumanIK-compatible data.",
        )

        status = gr.Textbox(label="Status", interactive=False)
        animations = gr.Files(label="Animation files", type="filepath")
        json_file = gr.File(label="Motion JSON", type="filepath")

        run_btn = gr.Button("Generate motion")
        run_btn.click(ui.run, inputs=[prompt, num_repetitions, output_format], outputs=[status, animations, json_file])
    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch a Gradio web demo for MDM.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the web server")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve the app")
    parser.add_argument(
        "--model-path", dest="model_path", default=None, help="Optional path to a checkpoint overriding defaults"
    )
    parser.add_argument(
        "--guidance", dest="guidance", type=float, default=None, help="Override the classifier-free guidance scale"
    )
    args = parser.parse_args()

    demo = build_demo(args.model_path, args.guidance)
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
