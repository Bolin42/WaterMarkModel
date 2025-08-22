import logging
import torch
from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from datetime import datetime

from config import config
from utils.utils import setup_logging, load_checkpoint, save_image
from service.LoadT2IModel import load_stable_diffusion_model
from encoder.encoder import WatermarkEncoder
from decoder.decoder import WatermarkDecoder

def test():
    setup_logging()
    logging.info("Starting testing process for 30 rounds...")

    # --- Create main output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join("results/testing", f"test_runs_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    logging.info(f"Results will be saved in: {main_output_dir}")

    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        logging.warning("CUDA not available. Testing on CPU is not recommended and will be very slow.")

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

    num_runs = 30
    all_results = []

    for i in range(num_runs):
        logging.info(f"--- Starting Test Run {i+1}/{num_runs} ---")
        
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

            # Generate initial latent
            initial_latents = torch.randn((1, config.LATENT_DIM, config.IMG_SIZE // 8, config.IMG_SIZE // 8)).to(config.DEVICE).half()
            
            # Generate Clean Image
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

            # Generate Watermarked Image
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

            # Decode Watermark
            recovered_watermark = decoder(watermarked_latents.float())
            
            # --- Generate and Save Single Report ---
            run_output_dir = os.path.join(main_output_dir, f"run_{i+1}")
            os.makedirs(run_output_dir, exist_ok=True)

            # Calculate Metrics
            clean_image_np = clean_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            watermarked_image_np = watermarked_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_value = psnr(clean_image_np, watermarked_image_np, data_range=1.0)
            ssim_value = ssim(clean_image_np, watermarked_image_np, data_range=1.0, channel_axis=-1)
            b_watermark = (watermark > 0).byte()
            b_recovered_watermark = (recovered_watermark > 0).byte()
            bit_errors = torch.sum(b_watermark != b_recovered_watermark).item()
            total_bits = watermark.numel()
            ber = bit_errors / total_bits
            accuracy = 1 - ber

            # Store results for summary
            run_results = {"run": i + 1, "psnr": psnr_value, "ssim": ssim_value, "ber": ber, "accuracy": accuracy}
            all_results.append(run_results)

            # Create and save single TXT report
            report_template = """--- Evaluation Report for Run {run_num} ---
  Invisibility:
    PSNR: {psnr:.4f} dB
    SSIM: {ssim:.4f}
  Robustness:
    BER: {ber:.4f}
    Accuracy: {accuracy:.4f}
------------------------------------"""
            report_str = report_template.format(
                run_num=i + 1,
                psnr=psnr_value,
                ssim=ssim_value,
                ber=ber,
                accuracy=accuracy
            )
            logging.info(report_str)
            with open(os.path.join(run_output_dir, "report.txt"), "w") as f:
                f.write(report_str)

            # Save Artifacts
            torch.save(watermark, os.path.join(run_output_dir, "original_watermark.pt"))
            torch.save(recovered_watermark, os.path.join(run_output_dir, "recovered_watermark.pt"))
            save_image(clean_image.squeeze(0), os.path.join(run_output_dir, "clean_image.png"))
            save_image(watermarked_image.squeeze(0), os.path.join(run_output_dir, "watermarked_image.png"))

    # --- Generate and Save Final Summary ---
    logging.info("\n--- Overall Test Summary ---")
    
    # Create DataFrame and save as CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(main_output_dir, "summary.csv"), index=False)
    logging.info(f"Summary report saved to: {os.path.join(main_output_dir, 'summary.csv')}")

    # Create and save summary TXT report
    avg_psnr = results_df['psnr'].mean()
    avg_ssim = results_df['ssim'].mean()
    avg_ber = results_df['ber'].mean()
    avg_accuracy = results_df['accuracy'].mean()
    
    summary_str = "\n--- Overall Test Summary ---\n"
    summary_str += f"Average PSNR: {avg_psnr:.4f} dB\n"
    summary_str += f"Average SSIM: {avg_ssim:.4f}\n"
    summary_str += f"Average BER: {avg_ber:.4f}\n"
    summary_str += f"Average Accuracy: {avg_accuracy:.4f}\n"
    summary_str += "--------------------------\n"
    logging.info(summary_str)
    with open(os.path.join(main_output_dir, "summary.txt"), "w") as f:
        f.write(summary_str)

    logging.info("Testing process finished.")

if __name__ == '__main__':
    if config.DEVICE == "cuda":
        test()
    else:
        logging.warning("CUDA not available. Testing on CPU is not recommended and will be very slow.")
