# Training interface for Cog ⚙️
from cog import BaseModel, Input, Path, BasePredictor, Secret
import os
import sys
import time
import torch
import shutil
import subprocess
from typing import Optional
from argparse import Namespace
from huggingface_hub import HfApi
from zipfile import ZipFile, is_zipfile

from trim_and_crop_videos import truncate_videos
from embed import batch_process
from text_to_video_lora import main as video_lora
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card

WEIGHTS_PATH = Path("mochi-1-preview")
INPUT_DIR = Path("videos_input")
DATA_DIR = Path("videos_prepared")
OUTPUT_DIR = Path("mochi-lora")
WEIGHTS_URL = "https://weights.replicate.delivery/default/genmo/mochi-1-preview/full.tar"

class TrainingOutput(BaseModel):
    weights: Path

def download_weights():
    if not WEIGHTS_PATH.exists():
        print("Downloading base weights")
        t1 = time.time()
        subprocess.check_output([ "pget", "-xf", WEIGHTS_URL, str(WEIGHTS_PATH)])
        print(f"Downloaded base weights in {time.time() - t1} seconds")

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Initialize any required variables
        download_weights()

    def predict(
        self,
        input_videos: Path = Input(
            description="A zip file containing the video snippets that will be used for training. We recommend a minimum of 12 videos of only a few seconds each. If you include captions, include them as one .txt file per video, e.g. video-1.mp4 should have a caption file named video-1.txt.",
            default=None,
        ),
        trim_and_crop: bool = Input(
            description="Automatically trim and crop video inputs", default=True
        ),
        steps: int = Input(
            description="Number of training steps. Recommended range 500-4000",
            ge=10,
            le=6000,
            default=100,
        ),
        learning_rate: float = Input(
            description="Learning rate, if you're new to training you probably don't need to change this.",
            default=4e-4,
        ),
        caption_dropout: float = Input(
            description="Caption dropout, if you're new to training you probably don't need to change this.",
            default=0.1,
            ge=0.01,
            le=1.0,
        ),
        batch_size: int = Input(
            description="Batch size, you can leave this as 1", default=1
        ),
        optimizer: str = Input(
            description="Optimizer to use for training. Supports: adam, adamw.",
            default="adamw",
        ),
        compile_dit: bool = Input(
            description="Compile the transformer", default=False
        ),
        seed: int = Input(
            description="Seed for reproducibility, you can leave this as 42", default=42, ge=0, le=100000
        ),
        hf_repo_id: str = Input(
            description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, lucataco/mochi-lora-vhs. If the given repo does not exist, a new public repo will be created.",
            default=None,
        ),
        hf_token: Secret = Input(
            description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
            default=None,
        ),
    ) -> TrainingOutput:
        if not input_videos:
            raise ValueError("input_videos must be provided")
        
        # Model configuration
        output_path = "/tmp/trained_model.tar"
        clean_up()

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Unzip and trim input videos
        num_frames = 37
        width = 848
        height = 480
        dataloader_num_workers = 4
        
        try:
            if trim_and_crop:
                extract_zip(input_videos, INPUT_DIR)
                print("---Starting to Trim input videos---")
                truncate_videos(str(INPUT_DIR), str(DATA_DIR), num_frames, f"{height}x{width}", True)
                print("---Starting to Embed videos---")
                batch_process(DATA_DIR, WEIGHTS_PATH, f"{num_frames}x{height}x{width}", True)
            else:
                extract_zip(input_videos, DATA_DIR)
                
            # Finetune process
            args_dict = {
                "pretrained_model_name_or_path": WEIGHTS_PATH,
                "data_root": DATA_DIR,
                "output_dir": OUTPUT_DIR,
                "max_train_steps": steps,
                "learning_rate": learning_rate,
                "train_batch_size": batch_size,
                "optimizer": optimizer,
                "caption_dropout": caption_dropout,
                "dataloader_num_workers": dataloader_num_workers,
                "seed": seed,
                "rank": 16,
                "lora_alpha": 16,
                "pin_memory": True,
                "gradient_checkpointing": True,
                "enable_slicing": True,
                "enable_tiling": True,
                "enable_model_cpu_offload": True,
                "allow_tf32": True,
                "cast_dit": True,
                "compile_dit": False,
                "push_to_hub": False,
                "variant": None,
                "revision": None,
                "scale_lr": None,
                "report_to": None,
                "checkpointing_steps": None,
                "resume_from_checkpoint": None,
                "validation_prompt": None,
                "lr_warmup_steps": 200,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "weight_decay": 0.01,
                "height": height,
                "width": width,
                "fps": 30,
                "hub_model_id": hf_repo_id if hf_repo_id else None,
                "hub_token": hf_token.get_secret_value() if hf_token else None
            }
            if compile_dit:
                args_dict["compile_dit"] = True
                
            final_args = Namespace(**args_dict)
            print("---Starting training---")
            video_lora(args=final_args)

            # Tar up output directory
            print("---Tar up output directory---")
            os.system(f"tar -cvf {output_path} {OUTPUT_DIR}")

            # Upload to Hugging Face if hf_token and hf_repo_id are provided
            if hf_token is not None and hf_repo_id is not None:
                print(f"Uploading to Hugging Face: {hf_repo_id}")
                try:
                    api = HfApi()
                    # Create or get repository
                    repo_url = api.create_repo(
                        repo_id=hf_repo_id,
                        private=False,
                        exist_ok=True,
                        token=hf_token.get_secret_value(),
                    )
                    print(f"HF Repo URL: {repo_url}")

                    # Create model card
                    model_card = load_or_create_model_card(
                        repo_id_or_path=hf_repo_id,
                        from_training=True,
                        license="apache-2.0",
                        base_model=str(WEIGHTS_PATH),
                        model_description="""# Mochi-1 Preview LoRA Finetune

This is a LoRA fine-tune of the Mochi-1 preview model. The model was trained using custom training data.

## Usage

```python
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import torch

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")
pipe.load_lora_weights("{repo_id}")
pipe.enable_model_cpu_offload()

video = pipe(
    prompt="your prompt here",
    guidance_scale=6.0,
    num_inference_steps=64,
    height=480,
    width=848,
    max_sequence_length=256,
).frames[0]

export_to_video(video, "output.mp4", fps=30)
```

## Training details

Trained on Replicate using: [lucataco/mochi-1-lora-trainer](https://replicate.com/lucataco/mochi-1-lora-trainer)
""".format(repo_id=hf_repo_id)
                    )

                    # Add tags to model card
                    tags = [
                        "text-to-video",
                        "diffusers-training",
                        "diffusers",
                        "lora",
                        "replicate",
                        "mochi-1-preview",
                    ]
                    model_card = populate_model_card(model_card, tags=tags)
                    
                    # Save model card
                    model_card.save(os.path.join(OUTPUT_DIR, "README.md"))

                    # Upload files
                    api.upload_folder(
                        repo_id=hf_repo_id,
                        folder_path=OUTPUT_DIR,
                        path_in_repo="",
                        token=hf_token.get_secret_value(),
                    )
                    print(f"Successfully uploaded model to {repo_url}")
                    
                except Exception as e:
                    print(f"Error uploading to Hugging Face: {str(e)}")
                    raise

            return TrainingOutput(weights=Path(output_path))
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise


def clean_up():
    print("Cleaning up previous runs")
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def extract_zip(input_videos: Path, input_dir: Path):
    if not is_zipfile(input_videos):
        raise ValueError("input_videos must be a zip file")

    input_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    with ZipFile(input_videos, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, input_dir)
                image_count += 1

    print(f"Extracted {image_count} files from zip to {input_dir}")


def handle_hf_readme(hf_repo_id: str):
    readme_path = OUTPUT_DIR / "README.md"
    license_path = Path("lora-license.md")
    shutil.copy(license_path, readme_path)
    content = readme_path.read_text()
    content = content.replace("[hf_repo_id]", hf_repo_id)
    repo_parts = hf_repo_id.split("/")
    if len(repo_parts) > 1:
        title = repo_parts[1].replace("-", " ").title()
        content = content.replace("[title]", title)
    else:
        content = content.replace("[title]", hf_repo_id)

    content = content.replace("[instance_prompt]", "")
    print(content)
    readme_path.write_text(content)
