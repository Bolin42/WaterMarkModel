import logging
import torch
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import config
from utils.utils import setup_logging, load_checkpoint, save_image
from service.LoadT2IModel import load_stable_diffusion_model
from encoder.encoder import WatermarkEncoder
from decoder.decoder import WatermarkDecoder

def calculate_ber(original, recovered):
    """Calculates the Bit Error Rate."""
    b_original = (original > 0).byte()
    b_recovered = (recovered > 0).byte()
    bit_errors = torch.sum(b_original != b_recovered).item()
    total_bits = original.numel()
    return bit_errors / total_bits

def ablation_experiment():
    """
    Runs an ablation experiment to study the effect of watermark strength.
    """
    setup_logging()
    logging.info("Starting ablation experiment...")

    # --- Create results directory ---
    results_dir = "results/ablation"
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

    scaling_factors = [0.1, 0.5, 1.0, 1.5, 2.0]
    num_rounds = 10

    experiment_results = {}

    for scaling_factor in scaling_factors:
        logging.info(f"--- Testing with Scaling Factor: {scaling_factor} ---")
        round_results = []
        for i in range(num_rounds):
            logging.info(f"  Round {i+1}/{num_rounds}")
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
                watermarked_latents = watermarked_latents + scaling_factor * watermark_embedding
                
                scheduler.set_timesteps(50)
                for t in scheduler.timesteps:
                    latent_model_input = torch.cat([watermarked_latents] * 2)
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                    watermarked_latents = scheduler.step(noise_pred, t, watermarked_latents).prev_sample

                watermarked_image = vae.decode(1 / 0.18215 * watermarked_latents).sample
                watermarked_image = (watermarked_image / 2 + 0.5).clamp(0, 1)

                # --- Decode Watermark and Evaluate ---
                recovered_watermark = decoder(watermarked_latents.float())
                
                # --- Calculate Metrics ---
                clean_image_np = clean_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                watermarked_image_np = watermarked_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                psnr_value = psnr(clean_image_np, watermarked_image_np, data_range=1.0)
                ssim_value = ssim(clean_image_np, watermarked_image_np, data_range=1.0, channel_axis=-1)
                ber = calculate_ber(watermark, recovered_watermark)
                accuracy = 1 - ber

                round_results.append({
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "ber": ber,
                    "accuracy": accuracy
                })

        experiment_results[scaling_factor] = round_results

    # --- 4. Save Final Summary Report ---
    report_path = os.path.join(results_dir, "ablation_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Ablation Experiment Summary Report ---")
        for scaling_factor, results in experiment_results.items():
            avg_psnr = np.mean([r['psnr'] for r in results])
            avg_ssim = np.mean([r['ssim'] for r in results])
            avg_ber = np.mean([r['ber'] for r in results])
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            f.write(f"\n--- Results for Scaling Factor: {scaling_factor} ---")
            f.write(f"  Average PSNR: {avg_psnr:.4f} dB")
            f.write(f"  Average SSIM: {avg_ssim:.4f}")
            f.write(f"  Average BER: {avg_ber:.4f}")
            f.write(f"  Average Accuracy: {avg_accuracy:.4f}")
        f.write("--- End of Summary Report ---")
    logging.info(f"Ablation experiment summary report saved to {report_path}")


if __name__ == '__main__':
    if config.DEVICE == "cuda":
        ablation_experiment()
    else:
        logging.warning("CUDA not available. Experiment on CPU is not recommended and will be very slow.")
