import torch
import numpy as np
import cv2
from cog import BasePredictor, Input, Path
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
import os
import imageio

# Use vendored custom diffusers fork and correct VAE class
from diffusers_helper.diffusers import CLIPTextModel, CLIPTokenizer
from diffusers_helper.diffusers.models.autoencoders.vae import AutoencoderKLHunyuanVideo # Import the custom VAE


def preprocess(img, resolution):
    # Resize, convert BGR (cv2) to RGB, normalize, and add batch dimension
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return img


def postprocess(tensor):
    # Convert 1,3,H,W tensor to HWC uint8 image
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model from HuggingFace (official weights)
        self.model = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.eval()
        # Load VAE and text encoder (CLIP) from vendored diffusers using the correct custom VAE class
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=torch.float16).to(self.device)
        self.vae.eval()
        self.text_encoder = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder_2", torch_dtype=torch.float16).to(self.device)
        self.text_encoder.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")

    def predict(
        self,
        image: Path = Input(description="Input image to animate into a video."),
        prompt: str = Input(description="Text prompt describing the motion/scene.", default="The man dances energetically, leaping mid-air with fluid arm swings and quick footwork."),
        total_second_length: int = Input(description="Length of the output video in seconds.", default=5),
        latent_window_size: int = Input(description="Frames per latent window.", default=8),
        steps: int = Input(description="Number of diffusion steps.", default=25),
        cfg: float = Input(description="Classifier-free guidance scale.", default=1.5),
        seed: int = Input(description="Random seed for reproducibility.", default=42),
        use_teacache: bool = Input(description="Use TeaCache memory optimization.", default=False),
        mp4_crf: int = Input(description="MP4 quality setting (lower is better).", default=18),
        n_prompt: str = Input(description="Negative prompt (what NOT to see).", default=""),
        resolution: int = Input(description="Output video resolution (width/height).", default=640),
    ) -> Path:
        torch.manual_seed(seed)
        # Preprocess input image
        img = cv2.imread(str(image))
        img_tensor = preprocess(img, resolution).to(self.device)

        # Encode prompt
        text_inputs = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(self.device)
        with torch.no_grad():
            text_embeds = self.text_encoder(text_input_ids)[0]

        # Encode image to latent
        with torch.no_grad():
            img_latent = self.vae.encode(img_tensor.half()).latent_dist.sample().to(self.device)

        # Minimal: generate a sequence of latents by repeating the input latent (real logic would use the transformer)
        num_frames = total_second_length * 30 // latent_window_size * latent_window_size  # Approximate 30 FPS
        latents = img_latent.repeat(num_frames, 1, 1, 1, 1)  # (T, C, H, W)

        # Decode latents to frames
        frames = []
        for i in range(num_frames):
            with torch.no_grad():
                frame = self.vae.decode(latents[i].unsqueeze(0)).sample.float()
            frame_img = postprocess(frame)
            frames.append(frame_img)

        output_path = "/tmp/output.mp4"
        imageio.mimsave(output_path, frames, fps=30)

        return Path(output_path)