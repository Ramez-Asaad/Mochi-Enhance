import torch

# Model Configuration
MODEL_ID = "genmo/mochi-1-preview"
TORCH_DTYPE = torch.bfloat16

# Generation Configuration
NUM_FRAMES = 60   # ~2 seconds at 30fps
FPS = 30
NUM_INFERENCE_STEPS = 50

# Prompts
NEGATIVE_PROMPT = "blurry, morphed, morphing, jittery, low resolution, deformed, unnatural motion, artifacts, bad anatomy"

EVALUATION_PROMPTS = [
    "A camel moving in the desert with pyramids in the background, dramatic sunset",
    "A golden retriever running through a field of wildflowers, sunny day",
    "Ocean waves crashing on a rocky coastline during a storm, dramatic lighting",
    "A butterfly landing on a flower in a garden, macro shot, soft focus background",
    "A busy city street at night with neon signs and rain reflections",
]

# Reproducibility
SEEDS = [42, 123, 456]

# Ablation Configurations
# Each config is: (name, scheduler_shift, negative_prompt_or_none, guidance_scale)
ABLATION_CONFIGS = [
    ("Baseline",             1.0,   None,             4.5),
    ("A: Scheduler Only",    1.15,  None,             4.5),
    ("B: Neg Prompt Only",   1.0,   NEGATIVE_PROMPT,  4.5),
    ("C: Guidance Only",     1.0,   None,             6.0),
    ("D: All Combined",      1.15,  NEGATIVE_PROMPT,  6.0),
]

# Output Paths
RESULTS_DIR = "results"
