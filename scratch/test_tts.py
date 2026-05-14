from pocket_tts import TTSModel
import torch
import scipy.io.wavfile

try:
    model = TTSModel.load_model()
    # Test with empty dict
    print("Testing generate_audio with {}")
    audio = model.generate_audio({}, "Hello world")
    print(f"Success! Audio shape: {audio.shape}")
except Exception as e:
    print(f"Failed: {e}")
