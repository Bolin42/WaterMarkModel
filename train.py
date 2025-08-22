import logging
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import bitsandbytes as bnb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import matplotlib.pyplot as plt
import pandas as pd
import time

from config import config
from utils.utils import setup_logging, save_checkpoint, load_checkpoint, save_image
from service.LoadT2IModel import load_stable_diffusion_model
from encoder.encoder import WatermarkEncoder
from decoder.decoder import WatermarkDecoder
from data_loader import get_prompt_loader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_detailed_report(train_losses, val_losses, lrs, epoch_times, max_gpu_memory_allocated):
    report_path = os.path.join(config.LOG_DIR, "detailed_training_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Detailed Training Report ---")
        f.write("\n")
        f.write("\n")

        f.write("### 1. Model Architecture ###")
        f.write("\n")
        f.write(f"- WatermarkEncoder: Linear layers (input: {config.WATERMARK_LENGTH}, output: {config.LATENT_DIM * (config.IMG_SIZE // 8) * (config.IMG_SIZE // 8)})")
        f.write("\n")
        f.write(f"- WatermarkDecoder: (Details from decoder.py would go here)")
        f.write("\n")
        f.write(f"- Stable Diffusion Model: {config.MODEL_ID}")
        f.write("\n")
        f.write(f"- Trainable Encoder parameters: {sum(p.numel() for p in WatermarkEncoder().parameters() if p.requires_grad):,}")
        f.write("\n")
        f.write(f"- Trainable Decoder parameters: {sum(p.numel() for p in WatermarkDecoder().parameters() if p.requires_grad):,}")
        f.write("\n")
        f.write("\n")

        f.write("### 2. Training Data ###")
        f.write("\n")
        f.write(f"- Training Prompts File: {config.PROMPTS_FILE_PATH} (Number of prompts: {len(open(config.PROMPTS_FILE_PATH).readlines()):,})")
        f.write("\n")
        f.write(f"- Validation Prompts File: {config.VALIDATION_PROMPTS_FILE_PATH} (Number of prompts: {len(open(config.VALIDATION_PROMPTS_FILE_PATH).readlines()):,})")
        f.write("\n")
        f.write(f"- Image Resolution: {config.IMG_SIZE}x{config.IMG_SIZE}")
        f.write("\n")
        f.write("- Data Preprocessing: Tokenization, image resizing (implicit in Stable Diffusion model loading)")
        f.write("\n")
        f.write("\n")

        f.write("### 3. Training Setup ###")
        f.write("\n")
        f.write(f"- Batch Size: {config.BATCH_SIZE}")
        f.write("\n")
        f.write(f"- Learning Rate: {config.LEARNING_RATE}")
        f.write("\n")
        f.write(f"- Optimizer Type: {config.OPTIMIZER}")
        f.write("\n")
        f.write(f"- Adam Beta1: {config.ADAM_BETA1}")
        f.write("\n")
        f.write(f"- Adam Beta2: {config.ADAM_BETA2}")
        f.write("\n")
        f.write(f"- Weight Decay: {config.ADAM_WEIGHT_DECAY}")
        f.write("\n")
        f.write("\n")

        f.write("### 4. Training Schedule ###")
        f.write("\n")
        f.write(f"- Total Epochs: {config.EPOCHS}")
        f.write("\n")
        f.write(f"- Gradient Accumulation Steps: {config.GRADIENT_ACCUMULATION_STEPS}")
        f.write("\n")
        f.write("- Warm-up Steps: Not explicitly configured")
        f.write("\n")
        f.write("- Learning Rate Decay Strategy: Not explicitly configured")
        f.write("\n")
        f.write("\n")

        f.write("### 5. Hardware ###")
        f.write("\n")
        f.write(f"- Device: {config.DEVICE}")
        f.write("\n")
        if torch.cuda.is_available():
            f.write(f"- GPU Type: {torch.cuda.get_device_name(0)}")
            f.write("\n")
            f.write(f"- Number of GPUs: {torch.cuda.device_count()}")
            f.write("\n")
            if max_gpu_memory_allocated:
                f.write(f"- Max GPU Memory Allocated (per epoch, GB): {max_gpu_memory_allocated}")
                f.write("\n")
        else:
            f.write("- GPU not available")
            f.write("\n")
        if epoch_times:
            f.write(f"- Average Training Time per Epoch: {sum(epoch_times) / len(epoch_times):.2f} seconds")
            f.write("\n")
            f.write(f"- Total Training Time: {sum(epoch_times):.2f} seconds")
            f.write("\n")
        f.write("\n")

        f.write("### 6. Loss Functions ###")
        f.write("\n")
        f.write(f"- Watermark Loss Function: {config.LOSS_FN_WTM}")
        f.write("\n")
        f.write(f"- Watermark Loss Weight: {config.WTM_LOSS_WEIGHT}")
        f.write("\n")
        f.write("\n")

        f.write("### 7. Evaluation Metrics ###")
        f.write("\n")
        f.write("- Currently, only Watermark Recovery MSE is logged during validation.")
        f.write("\n")
        f.write("- FID, IS, LPIPS, Human Evaluation: Not implemented in current setup. Requires additional development.")
        f.write("\n")
        f.write("\n")

        f.write("### 8. Ablation Studies ###")
        f.write("\n")
        f.write("- Not applicable to a single training run. Requires multiple comparative experiments.")
        f.write("\n")
        f.write("\n")

        f.write("### 9. Hyperparameters ###")
        f.write("\n")
        f.write(f"- Image Resolution: {config.IMG_SIZE}x{config.IMG_SIZE}")
        f.write("\n")
        f.write(f"- Text Encoder Type: Implicitly part of {config.MODEL_ID} (CLIP-based)")
        f.write("\n")
        f.write(f"- Noise Schedule Parameters: Implicitly part of {config.MODEL_ID} (DDPM-based)")
        f.write("\n")
        f.write("\n")

        f.write("### 10. Training Dynamics ###")
        f.write("\n")
        f.write("- Loss Curves: See generated training_loss.png and validation_loss.png")
        f.write("\n")
        f.write("- Learning Rate Curves: See generated learning_rate.png")
        f.write("\n")
        f.write("- Sample Quality Evolution: Not currently saved during training. Requires saving generated images at intervals.")
        f.write("\n")
        f.write("\n")

        f.write("### Raw Data ###")
        f.write("\n")
        f.write("- See training_report.csv for detailed epoch-wise metrics.")
        f.write("\n")

    logging.info(f"Detailed training report saved to {report_path}")

def train(rank, world_size):
    setup(rank, world_size)
    setup_logging()
    logging.info("Starting training process...")
    print(f"Configured WATERMARK_LENGTH: {config.WATERMARK_LENGTH}")
    print(f"Config module path: {config.__file__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        logging.warning("CUDA not available. Training will fail as only GPU is configured.")

    # Log model parameter counts
    # These lines are moved after encoder/decoder instantiation
    writer = SummaryWriter(f"runs/watermark_experiment_1")

    train_losses = []
    val_losses = []
    lrs = []
    epoch_times = []
    max_gpu_memory_allocated = []

    # --- 1. Load Models ---
    logging.info("Loading Stable Diffusion model components...")
    sd_components = load_stable_diffusion_model()
    unet = sd_components["unet"]
    vae = sd_components["vae"]
    text_encoder = sd_components["text_encoder"]
    tokenizer = sd_components["tokenizer"]
    scheduler = sd_components["scheduler"]

    # Freeze the pre-trained models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    encoder = WatermarkEncoder()
    decoder = WatermarkDecoder()

    # Log model parameter counts (moved here)
    total_encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logging.info(f"Trainable Encoder parameters: {total_encoder_params:,}")
    logging.info(f"Trainable Decoder parameters: {total_decoder_params:,}")

    # --- 2. Optimizer and Loss ---
    optimizer = bnb.optim.AdamW8bit(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.LEARNING_RATE,
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        weight_decay=config.ADAM_WEIGHT_DECAY,
    )
    
    if config.LOSS_FN_WTM == "MSE":
        loss_fn_wtm = torch.nn.MSELoss()
    elif config.LOSS_FN_WTM == "L1":
        loss_fn_wtm = torch.nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {config.LOSS_FN_WTM}")

    # --- 3. Data Loaders ---
    train_loader = get_prompt_loader(config.PROMPTS_FILE_PATH, config.BATCH_SIZE)
    val_loader = get_prompt_loader(config.VALIDATION_PROMPTS_FILE_PATH, config.BATCH_SIZE, shuffle=False)

    # --- 4. Training Loop ---
    start_epoch = load_checkpoint(encoder, optimizer, config.ENCODER_PATH)
    load_checkpoint(decoder, optimizer, config.DECODER_PATH)
    
    best_val_loss = float('inf')

    device = config.DEVICE
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
    epoch = start_epoch + 1
    while epoch <= config.EPOCHS:
        try:
            start_time = time.time()
            logging.info(f"--- Epoch {epoch}/{config.EPOCHS} on {device} ---")
            
            # Move models to the selected device
            encoder.to(device)
            decoder.to(device)
            unet.to(device)
            vae.to(device)
            text_encoder.to(device)

            encoder.train()
            decoder.train()
            
            total_train_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
            for i, prompts in enumerate(pbar):
                watermarks = torch.randn(len(prompts), config.WATERMARK_LENGTH).to(device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    # --- Forward Pass ---
                    # Get conditional text embeddings
                    text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    text_embeddings_cond = text_encoder(text_inputs.input_ids.to(device))[0]

                    # Get unconditional text embeddings
                    uncond_tokens = [""] * len(prompts)
                    uncond_inputs = tokenizer(uncond_tokens, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                    text_embeddings_uncond = text_encoder(uncond_inputs.input_ids.to(device))[0]
                    
                    # Concatenate for classifier-free guidance
                    text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond])

                    latents = torch.randn((len(prompts), config.LATENT_DIM, config.IMG_SIZE // 8, config.IMG_SIZE // 8)).to(device)
                    
                    watermark_embedding = encoder(watermarks)
                    latents = latents + watermark_embedding

                    scheduler.set_timesteps(50)
                    for t in scheduler.timesteps:
                        latent_model_input = torch.cat([latents] * 2)
                        # Convert t to a Python scalar for DataParallel
                        t_scalar = t.item()
                        noise_pred = unet(latent_model_input, timestep=t_scalar, encoder_hidden_states=text_embeddings).sample
                        noise_pred_list = list(noise_pred) # Convert generator to list
                        noise_pred = torch.cat(noise_pred_list, dim=0) # Concatenate into a single tensor
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
                    # --- Decode Watermark ---
                    recovered_watermarks = decoder(latents)
                    
                    # --- Loss Calculation ---
                    loss = loss_fn_wtm(recovered_watermarks, watermarks) / config.GRADIENT_ACCUMULATION_STEPS
                
                total_train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                
                if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                pbar.set_postfix({"loss": loss.item() * config.GRADIENT_ACCUMULATION_STEPS})

            # --- Logging and Validation ---
            avg_train_loss = total_train_loss / len(train_loader)
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                val_loss = validate(encoder, decoder, val_loader, loss_fn_wtm, tokenizer, text_encoder, unet, scheduler, device)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            logging.info(f"Epoch {epoch} Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            lrs.append(optimizer.param_groups[0]['lr'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info("New best validation loss. Saving best models.")
                logging.info(f"Saving best encoder to {config.BEST_ENCODER_PATH}")
                logging.info(f"Saving best decoder to {config.BEST_DECODER_PATH}")
                save_checkpoint(epoch, encoder, optimizer, config.BEST_ENCODER_PATH)
                save_checkpoint(epoch, decoder, optimizer, config.BEST_DECODER_PATH)

            # --- Save Checkpoint ---
            save_checkpoint(epoch, encoder, optimizer, config.ENCODER_PATH)
            save_checkpoint(epoch, decoder, optimizer, config.DECODER_PATH)
            
            if device == "cuda":
                torch.cuda.empty_cache()
                max_gpu_memory_allocated.append(torch.cuda.max_memory_allocated() / (1024**3)) # in GB
            
            end_time = time.time()
            epoch_duration = end_time - start_time
            epoch_times.append(epoch_duration)
            logging.info(f"Epoch {epoch} finished in {epoch_duration:.2f} seconds.")
            if device == "cuda":
                logging.info(f"Max GPU memory allocated: {max_gpu_memory_allocated[-1]:.2f} GB")

            epoch += 1 # Increment epoch only on success

        except torch.cuda.OutOfMemoryError:
            logging.error(f"CUDA out of memory in epoch {epoch}. Training stopped.")
            raise # Re-raise the exception to stop the script
    
    writer.close()

    # --- Generate and Save Report ---
    logging.info("Generating training report...")

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Training Loss': train_losses,
        'Validation Loss': val_losses,
        'Learning Rate': lrs
    })

    # Save metrics to CSV
    report_csv_path = os.path.join(config.LOG_DIR, "training_report.csv")
    metrics_df.to_csv(report_csv_path, index=False)
    logging.info(f"Training metrics saved to {report_csv_path}")

    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Training Loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    train_loss_plot_path = os.path.join(config.LOG_DIR, "training_loss.png")
    plt.savefig(train_loss_plot_path)
    plt.close()
    logging.info(f"Training loss plot saved to {train_loss_plot_path}")

    # Plot Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Validation Loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    val_loss_plot_path = os.path.join(config.LOG_DIR, "validation_loss.png")
    plt.savefig(val_loss_plot_path)
    plt.close()
    logging.info(f"Validation loss plot saved to {val_loss_plot_path}")

    # Plot Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['Learning Rate'], label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.legend()
    plt.grid(True)
    lr_plot_path = os.path.join(config.LOG_DIR, "learning_rate.png")
    plt.savefig(lr_plot_path)
    plt.close()
    logging.info(f"Learning rate plot saved to {lr_plot_path}")

    generate_detailed_report(train_losses, val_losses, lrs, epoch_times, max_gpu_memory_allocated)

    cleanup()

def validate(encoder, decoder, val_loader, loss_fn_wtm, tokenizer, text_encoder, unet, scheduler, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        pbar = tqdm(val_loader, desc="Validation")
        for prompts in pbar:
            watermarks = torch.randn(len(prompts), config.WATERMARK_LENGTH).to(device)
            
            # Get conditional text embeddings
            text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings_cond = text_encoder(text_inputs.input_ids.to(device))[0]

            # Get unconditional text embeddings
            uncond_tokens = [""] * len(prompts)
            uncond_inputs = tokenizer(uncond_tokens, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings_uncond = text_encoder(uncond_inputs.input_ids.to(device))[0]
            
            # Concatenate for classifier-free guidance
            text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond])

            latents = torch.randn((len(prompts), config.LATENT_DIM, config.IMG_SIZE // 8, config.IMG_SIZE // 8)).to(device)
            
            watermark_embedding = encoder(watermarks)
            latents = latents + watermark_embedding

            scheduler.set_timesteps(50)
            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                # Convert t to a Python scalar for DataParallel
                t_scalar = t.item()
                noise_pred = unet(latent_model_input, timestep=t_scalar, encoder_hidden_states=text_embeddings).sample
                noise_pred_list = list(noise_pred) # Convert generator to list
                noise_pred = torch.cat(noise_pred_list, dim=0) # Concatenate into a single tensor
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            recovered_watermarks = decoder(latents)
            loss = loss_fn_wtm(recovered_watermarks, watermarks)
            total_loss += loss.item()
            
            pbar.set_postfix({"val_loss": loss.item()})

    return total_loss / len(val_loader)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs found. DDP requires at least one GPU.")
    print(f"Found {world_size} GPUs. Spawning processes for DDP training.")
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)