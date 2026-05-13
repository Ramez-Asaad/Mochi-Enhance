import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import lpips
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from scipy.linalg import sqrtm
import torchvision.models as models


class EvalSuite:
    """
    Lazily-loaded evaluation suite. Models are loaded only when first needed,
    making this safe to import in multi-process workers without CUDA conflicts.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self._clip_model = None
        self._clip_processor = None
        self._lpips_fn = None
        self._inception = None
        self.transform = T.ToTensor()

    # --- Lazy Loaders ---
    @property
    def clip_model(self):
        if self._clip_model is None:
            clip_id = "openai/clip-vit-base-patch32"
            print(f"  [EvalSuite] Loading CLIP on {self.device}...")
            self._clip_processor = CLIPProcessor.from_pretrained(clip_id)
            self._clip_model = CLIPModel.from_pretrained(clip_id).to(self.device)
        return self._clip_model

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            _ = self.clip_model  # triggers loading both
        return self._clip_processor

    @property
    def lpips_fn(self):
        if self._lpips_fn is None:
            print(f"  [EvalSuite] Loading LPIPS on {self.device}...")
            self._lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        return self._lpips_fn

    @property
    def inception(self):
        if self._inception is None:
            print(f"  [EvalSuite] Loading InceptionV3 on {self.device} for FVD...")
            self._inception = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
            self._inception.eval()
            # Remove the final classification head to get 2048-d features
            self._inception.fc = torch.nn.Identity()
        return self._inception

    # --- Helper ---
    def _to_uint8(self, frames):
        """Convert a list of frames (PIL/numpy) to a list of uint8 numpy arrays."""
        result = []
        for f in frames:
            if isinstance(f, Image.Image):
                result.append(np.array(f))
            elif f.dtype == np.float32 or f.dtype == np.float64:
                result.append((np.clip(f, 0, 1) * 255).astype(np.uint8))
            else:
                result.append(f)
        return result

    # --- Metrics ---
    def compute_clip_sim(self, frames, text_prompt):
        """Average CLIP similarity across all frames vs. the text prompt."""
        images = []
        for f in frames:
            if isinstance(f, Image.Image):
                images.append(f)
            elif isinstance(f, np.ndarray):
                if f.dtype == np.float32 or f.dtype == np.float64:
                    images.append(Image.fromarray((f * 255).astype(np.uint8)))
                else:
                    images.append(Image.fromarray(f))
            else:
                images.append(f)

        inputs = self.clip_processor(text=[text_prompt], images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        scores = outputs.logits_per_image.detach().cpu().numpy()
        return float(np.mean(scores))

    def compute_temporal_consistency(self, frames):
        """SSIM and LPIPS between consecutive frames (measures flicker/jitter)."""
        uint8_frames = self._to_uint8(frames)
        ssim_scores = []
        lpips_scores = []

        for i in range(len(uint8_frames) - 1):
            img1_np = uint8_frames[i]
            img2_np = uint8_frames[i + 1]

            min_dim = min(img1_np.shape[0], img1_np.shape[1])
            win_size = min(7, min_dim)
            if win_size % 2 == 0:
                win_size -= 1

            s = ssim(img1_np, img2_np, channel_axis=2, data_range=255, win_size=win_size)
            ssim_scores.append(s)

            img1_t = self.transform(Image.fromarray(img1_np)).unsqueeze(0).to(self.device) * 2.0 - 1.0
            img2_t = self.transform(Image.fromarray(img2_np)).unsqueeze(0).to(self.device) * 2.0 - 1.0
            with torch.no_grad():
                l = self.lpips_fn(img1_t, img2_t).item()
            lpips_scores.append(l)

        return float(np.mean(ssim_scores)), float(np.mean(lpips_scores))

    def _extract_inception_features(self, frames):
        """Extract InceptionV3 pool3 features from a list of frames."""
        uint8_frames = self._to_uint8(frames)
        preprocess = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        features = []
        for f_np in uint8_frames:
            img = Image.fromarray(f_np)
            tensor = preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.inception(tensor)
            features.append(feat.squeeze().cpu().numpy())
        return np.array(features)

    def compute_fvd(self, frames_a, frames_b):
        """
        Compute Frechet Video Distance between two sets of frames.
        Uses InceptionV3 features as a proxy (standard in video generation papers).
        Lower FVD = more similar distributions.
        """
        feats_a = self._extract_inception_features(frames_a)
        feats_b = self._extract_inception_features(frames_b)

        mu_a, sigma_a = np.mean(feats_a, axis=0), np.cov(feats_a, rowvar=False)
        mu_b, sigma_b = np.mean(feats_b, axis=0), np.cov(feats_b, rowvar=False)

        diff = mu_a - mu_b
        covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)

        # Numerical stability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fvd = float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))
        return fvd
