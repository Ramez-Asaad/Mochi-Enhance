import os
import time
import torch
import uvicorn
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from diffusers import MochiPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from pocket_tts import TTSModel
import scipy.io.wavfile
from moviepy.editor import VideoFileClip, AudioFileClip

# --- Configuration ---
MODEL_ID = "genmo/mochi-1-preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Video Gen + TTS Server")

# --- Model Loading ---
print("Loading Mochi Video Pipeline...")
pipe = MochiPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

print("Loading Pocket TTS Model...")
tts_model = TTSModel.load_model()
# Use a default voice prompt if available, or just generate
# For simplicity, we'll use the default voice logic

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, morphed, low resolution, deformed"
    num_frames: int = 60
    num_inference_steps: int = 50
    guidance_scale: int = 6.0
    seed: int = 42

def cleanup(file_path: str):
    """Wait a bit and delete the file."""
    time.sleep(10)
    if os.path.exists(file_path):
        os.remove(file_path)

@app.post("/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    video_temp_path = os.path.join(OUTPUT_DIR, f"{task_id}_temp.mp4")
    audio_temp_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    final_video_path = os.path.join(OUTPUT_DIR, f"{task_id}.mp4")

    try:
        # 1. Generate Video
        print(f"Generating video for: {request.prompt}")
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        
        with torch.autocast("cuda", torch.bfloat16):
            frames = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_frames=request.num_frames,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            ).frames[0]
        
        export_to_video(frames, video_temp_path, fps=30)

        # 2. Generate Audio
        print(f"Generating audio for: {request.prompt}")
        # Note: Pocket TTS might need a voice state. We use a simple generation here.
        # If the user has a specific voice .wav, we could use it. 
        # For now, we use the library's default.
        audio_data = tts_model.generate_audio(None, request.prompt)
        scipy.io.wavfile.write(audio_temp_path, tts_model.sample_rate, audio_data)

        # 3. Merge Audio and Video
        print("Merging audio and video...")
        video_clip = VideoFileClip(video_temp_path)
        audio_clip = AudioFileClip(audio_temp_path)
        
        # Ensure audio doesn't exceed video duration
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
        
        # Cleanup temp files
        video_clip.close()
        audio_clip.close()
        os.remove(video_temp_path)
        os.remove(audio_temp_path)

        background_tasks.add_task(cleanup, final_video_path)
        return FileResponse(final_video_path, media_type="video/mp4")

    except Exception as e:
        print(f"Error during generation: {e}")
        # Cleanup on error
        for p in [video_temp_path, audio_temp_path, final_video_path]:
            if os.path.exists(p): os.remove(p)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
