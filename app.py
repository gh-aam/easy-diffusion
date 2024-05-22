import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import torch

MAX_SEED = np.iinfo(np.int32).max

def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, model):

    if torch.cuda.is_available():
        torch.cuda.max_memory_allocated(device="cuda")
        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to("cuda")
    else: 
        pipe = DiffusionPipeline.from_pretrained(model, use_safetensors=True)
        pipe = pipe.to("cpu")
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker=None

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        width = width,
        height = height,
        generator = generator
    ).images[0]

    return image

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
        <div style="text-align: center; font-weight: bold; font-size: 20px;">Easy Diffusion</div>
        """)

        result = gr.Image(
            label="Image",
            show_label=False,
            format="png"
        )

        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            placeholder="Enter a prompt"
        )

        run_button = gr.Button("Generate")

        with gr.Accordion("Advanced", open=False):

            negative_prompt = gr.Text(
                label="Negative Prompt",
                placeholder="Enter a negative prompt"
            )
            
            model = gr.Dropdown(
                [
                    "Lykon/dreamshaper-8",
                    "digiplay/Juggernaut_final",
                    "digiplay/AbsoluteReality_v1.8.1",
                    "Yntec/epiCPhotoGasm"
                ],
                value="Lykon/dreamshaper-8",
                allow_custom_value=True,
                filterable=True,
                interactive=True,
                label="Model"
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0
            )

            randomize_seed = gr.Checkbox(
                label="Random Seed",
                value=True
            )

            with gr.Row():

                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=1024,
                    step=32,
                    value=512
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=1024,
                    step=32,
                    value=768
                )

            with gr.Row():

                guidance_scale = gr.Slider(
                    label="CFG",
                    minimum=0.0,
                    maximum=20.0,
                    step=0.5,
                    value=7.0
                )

                num_inference_steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=30
                )

    run_button.click(
        fn = infer,
        inputs = [prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, model],
        outputs = [result]
    )

demo.queue().launch(share=True)
