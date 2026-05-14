from pocket_tts import TTSModel
import torch
import scipy.io.wavfile

try:
    model = TTSModel.load_model()
    # Try with a predefined voice name
    voice_name = "alba"
    print(f"Getting state for voice name: {voice_name}")
    voice_state = model.get_state_for_audio_prompt(voice_name)
    print("Generating audio...")
    audio = model.generate_audio(voice_state, "Hello world")
    print(f"Success! Audio shape: {audio.shape}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
