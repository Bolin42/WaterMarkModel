import logging
import torch
import os
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageDraw
import random
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import config
from utils.utils import setup_logging, load_checkpoint, save_image
from service.LoadT2IModel import load_stable_diffusion_model
from encoder.encoder import WatermarkEncoder
from decoder.decoder import WatermarkDecoder

def add_noise(image, noise_level):
    """Adds Gaussian noise to an image."""
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def calculate_ber(original, recovered):
    """Calculates the Bit Error Rate."""
    b_original = (original > 0).byte()
    b_recovered = (recovered > 0).byte()
    bit_errors = torch.sum(b_original != b_recovered).item()
    total_bits = original.numel()
    return bit_errors / total_bits

def draw_random_shape(image_tensor):
    """Draws a random hollow shape on an image tensor."""
    # Convert tensor to PIL Image
    image = ToPILImage()(image_tensor.squeeze(0))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Randomly choose shape, color, and thickness
    shape_type = random.choice(['rectangle', 'ellipse'])
    outline_color = random.choice(['black', 'white'])
    thickness = random.randint(1, 5)

    # Randomly generate shape parameters
    if shape_type == 'rectangle':
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = random.randint(x1 + width // 4, width - 1)
        y2 = random.randint(y1 + height // 4, height - 1)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=thickness)
    elif shape_type == 'ellipse':
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = random.randint(x1 + width // 4, width - 1)
        y2 = random.randint(y1 + height // 4, height - 1)
        draw.ellipse([x1, y1, x2, y2], outline=outline_color, width=thickness)

    # Convert back to tensor
    return ToTensor()(image).unsqueeze(0)

def interference_experiment():
    """
    Runs an interference experiment by adding noise to watermarked images
    and evaluating the decoder's performance.
    """
    setup_logging()
    logging.info("Starting interference experiment...")

    # --- Create results directory ---
    results_dir = "results/interference"
    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Load Models ---
    logging.info("Loading models...")
    sd_components = load_stable_diffusion_model()
    vae = sd_components["vae"]
    unet = sd_components["unet"]
    text_encoder = sd_components["text_encoder"]
    tokenizer = sd_components["tokenizer"]
    scheduler = sd_components["scheduler"]

    encoder = WatermarkEncoder().to(config.DEVICE)
    decoder = WatermarkDecoder().to(config.DEVICE)

    load_checkpoint(encoder, None, config.BEST_ENCODER_PATH)
    load_checkpoint(decoder, None, config.BEST_DECODER_PATH)

    encoder.eval()
    decoder.eval()

    noise_levels = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01]
    num_rounds = 30

    experiment_results = {}

    for i in range(num_rounds):
        logging.info(f"--- Round {i+1}/{num_rounds} ---")
        
        prompt = [config.PROMPT]
        watermark = torch.randn(1, config.WATERMARK_LENGTH).to(config.DEVICE)
        
        with torch.no_grad():
            # Get text embeddings
            text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings_cond = text_encoder(text_inputs.input_ids.to(config.DEVICE))[0].half()
            uncond_tokens = [""] * len(prompt)
            uncond_inputs = tokenizer(uncond_tokens, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings_uncond = text_encoder(uncond_inputs.input_ids.to(config.DEVICE))[0].half()
            text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond])

            # Generate a single initial latent
            initial_latents = torch.randn((1, config.LATENT_DIM, config.IMG_SIZE // 8, config.IMG_SIZE // 8)).to(config.DEVICE).half()

            # --- Generate Clean Image ---
            clean_latents = initial_latents.clone()
            scheduler.set_timesteps(50)
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([clean_latents] * 2)
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                clean_latents = scheduler.step(noise_pred, t, clean_latents).prev_sample
            
            clean_image = vae.decode(1 / 0.18215 * clean_latents).sample
            clean_image = (clean_image / 2 + 0.5).clamp(0, 1)

            # --- Generate Watermarked Image ---
            watermarked_latents = initial_latents.clone()
            watermark_embedding = encoder(watermark).half()
            watermarked_latents = watermarked_latents + watermark_embedding
            
            scheduler.set_timesteps(50)
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([watermarked_latents] * 2)
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                watermarked_latents = scheduler.step(noise_pred, t, watermarked_latents).prev_sample

            watermarked_image = vae.decode(1 / 0.18215 * watermarked_latents).sample
            watermarked_image = (watermarked_image / 2 + 0.5).clamp(0, 1)

            # --- Invisibility Report ---
            clean_image_np = clean_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            watermarked_image_np = watermarked_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            psnr_value = psnr(clean_image_np, watermarked_image_np, data_range=1.0)
            ssim_value = ssim(clean_image_np, watermarked_image_np, data_range=1.0, channel_axis=-1)
            
            logging.info("  Invisibility:")
            logging.info(f"    PSNR: {psnr_value:.4f} dB")
            logging.info(f"    SSIM: {ssim_value:.4f}")

            # --- Robustness Report (Noise Interference) ---
            logging.info("  Robustness (Noise Interference):")
            robustness_results = {}
            for noise_level in noise_levels:
                noisy_image = add_noise(watermarked_image, noise_level)
                noisy_latents = vae.encode(noisy_image.to(vae.dtype) * 2 - 1).latent_dist.sample() * 0.18215
                recovered_watermark = decoder(noisy_latents.float())
                ber = calculate_ber(watermark, recovered_watermark)
                accuracy = 1 - ber
                logging.info(f"    Noise Level: {noise_level:<6} | BER: {ber:.4f} | Accuracy: {accuracy:.4f}")
                robustness_results[noise_level] = {"ber": ber, "accuracy": accuracy}

            # --- Robustness Report (Drawing Disturbance) ---
            logging.info("  Robustness (Drawing Disturbance):")
            drawn_image_tensor = draw_random_shape(watermarked_image.cpu()).to(config.DEVICE)
            drawn_latents = vae.encode(drawn_image_tensor.to(vae.dtype) * 2 - 1).latent_dist.sample() * 0.18215
            recovered_watermark_drawing = decoder(drawn_latents.float())
            ber_drawing = calculate_ber(watermark, recovered_watermark_drawing)
            accuracy_drawing = 1 - ber_drawing
            logging.info(f"    Shape Disturbance | BER: {ber_drawing:.4f} | Accuracy: {accuracy_drawing:.4f}")

            experiment_results[f"round_{i+1}"] = {
                "invisibility": {"psnr": psnr_value, "ssim": ssim_value},
                "robustness_noise": robustness_results,
                "robustness_drawing": {"ber": ber_drawing, "accuracy": accuracy_drawing}
            }

            # --- Save Artifacts for the round ---
            output_dir = os.path.join(config.OUTPUT_DIR, f"round_{i+1}")
            os.makedirs(output_dir, exist_ok=True)
            save_image(clean_image.squeeze(0), os.path.join(output_dir, "clean_image.png"))
            save_image(watermarked_image.squeeze(0), os.path.join(output_dir, "watermarked_image.png"))
            save_image(drawn_image_tensor.squeeze(0), os.path.join(output_dir, "drawn_disturbance_image.png"))
            for noise_level in noise_levels:
                noisy_image = add_noise(watermarked_image, noise_level)
                save_image(noisy_image.squeeze(0), os.path.join(output_dir, f"noisy_image_{noise_level}.png"))


    # --- 4. Save Final Summary Report ---
    report_path = os.path.join(results_dir, "interference_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Interference Experiment Summary Report ---\n")
        for round_num, results in experiment_results.items():
            f.write(f"\n--- Results for {round_num} ---\n")
            f.write(f"  Invisibility:\n")
            f.write(f"    PSNR: {results['invisibility']['psnr']:.4f} dB\n")
            f.write(f"    SSIM: {results['invisibility']['ssim']:.4f}\n")
            f.write(f"  Robustness (Noise):\n")
            for noise_level, robustness_res in results['robustness_noise'].items():
                f.write(f"    Noise Level: {noise_level:<6} | Accuracy: {robustness_res['accuracy']:.4f}\n")
            f.write(f"  Robustness (Drawing):\n")
            f.write(f"    Accuracy: {results['robustness_drawing']['accuracy']:.4f}\n")
        f.write("--- End of Summary Report ---\n")
    logging.info(f"Interference experiment summary report saved to {report_path}")



if __name__ == '__main__':
    if config.DEVICE == "cuda":
        interference_experiment()
    else:
        logging.warning("CUDA not available. Experiment on CPU is not recommended and will be very slow.")
