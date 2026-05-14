import os
import argparse
import torch
import uuid
import scipy.io.wavfile
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
from pocket_tts import TTSModel
from moviepy.editor import VideoFileClip, AudioFileClip

def main():
    parser = argparse.ArgumentParser(description="Generate Video + TTS from CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video and audio")
    parser.add_argument("--negative_prompt", type=str, default="blurry, morphed, low resolution, deformed", help="Negative prompt")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output filename")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # --- Video Generation ---
    print(f"🎬 Loading Mochi-1 Video Pipeline...")
    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    print(f"🚀 Generating video: '{args.prompt}'")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.autocast("cuda", torch.bfloat16):
        frames = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=60,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).frames[0]
    
    video_temp = "temp_video.mp4"
    export_to_video(frames, video_temp, fps=30)

    # --- Audio Generation ---
    print(f"🎙️ Generating TTS audio...")
    tts_model = TTSModel.load_model()
    audio_data = tts_model.generate_audio({}, args.prompt)
    audio_temp = "temp_audio.wav"
    scipy.io.wavfile.write(audio_temp, tts_model.sample_rate, audio_data)

    # --- Merging ---
    print(f"🎞️ Merging audio and video...")
    video_clip = VideoFileClip(video_temp)
    audio_clip = AudioFileClip(audio_temp)
    
    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)
    
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(args.output, codec="libx264", audio_codec="aac")
    
    # Cleanup
    video_clip.close()
    audio_clip.close()
    os.remove(video_temp)
    os.remove(audio_temp)

    print(f"✅ Done! Saved to: {args.output}")

if __name__ == "__main__":
    main()
