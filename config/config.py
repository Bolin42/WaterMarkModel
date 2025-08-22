import torch

# Model parameters
MODEL_ID = "AI-ModelScope/stable-diffusion-v1-5"
# This should be an absolute path to your downloaded model
LOCAL_MODEL_PATH = "/home/ecs-user/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-v1-5"
DEVICE = "cuda"

# Training parameters
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 1
WATERMARK_LENGTH = 128
LATENT_DIM = 4
IMG_SIZE = 256
OPTIMIZER = "AdamW"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_WEIGHT_DECAY = 1e-2
LOSS_FN_WTM = "MSE"
WTM_LOSS_WEIGHT = 1.0
GRADIENT_ACCUMULATION_STEPS = 8  # Increased from 4 to 8 to save memory

# Data parameters (relative paths)
DATA_DIR = "data"
PROMPTS_FILE_PATH = f"{DATA_DIR}/prompts.txt"
VALIDATION_PROMPTS_FILE_PATH = f"{DATA_DIR}/validation_prompts.txt"

# Checkpointing and Logging (relative paths)
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/training.log"
OUTPUT_DIR = "output"

# File paths (relative paths)
ENCODER_PATH = f"{CHECKPOINT_DIR}/encoder.pth"
DECODER_PATH = f"{CHECKPOINT_DIR}/decoder.pth"
BEST_ENCODER_PATH = f"{CHECKPOINT_DIR}/best_encoder.pth"
BEST_DECODER_PATH = f"{CHECKPOINT_DIR}/best_decoder.pth"
OUTPUT_IMAGE_PATH = f"{OUTPUT_DIR}/output.png"

# Test prompt
PROMPT = "a beautiful landscape painting"
