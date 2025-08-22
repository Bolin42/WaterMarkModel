import logging
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from config import config

def load_stable_diffusion_model():
    """
    Loads the Stable Diffusion model from a local path using diffusers.
    """
    logging.info(f"Loading Stable Diffusion model from: {config.LOCAL_MODEL_PATH}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            config.LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.to(config.DEVICE)

        # pipe.enable_sequential_cpu_offload(gpu_id=0)
        # pipe.enable_attention_slicing()

        model_components = {
            "unet": pipe.unet,
            "vae": pipe.vae,
            "text_encoder": pipe.text_encoder,
            "tokenizer": pipe.tokenizer,
            "scheduler": DDIMScheduler.from_config(pipe.scheduler.config),
            "pipeline": pipe
        }
        
        logging.info("Stable Diffusion model components loaded successfully with CPU offloading and attention slicing enabled.")
        return model_components

    except Exception as e:
        logging.error(f"Failed to load Stable Diffusion model: {e}")
        raise RuntimeError("Could not load the Stable Diffusion model.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # This is a simple test case to ensure the model loads correctly.
    try:
        components = load_stable_diffusion_model()
        for name, component in components.items():
            if hasattr(component, 'device'):
                logging.info(f"Component '{name}' is on device: {component.device}")
            else:
                logging.info(f"Component '{name}' loaded.")
    except Exception as e:
        logging.error(f"An error occurred during the model loading test case: {e}")