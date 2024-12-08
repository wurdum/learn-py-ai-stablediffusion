from typing import Optional
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class ImageGenerator:
    """Simple image generator using Stable Diffusion."""

    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1"):
        """Initialize the image generator.

        Args:
            model_id: Hugging Face model ID for Stable Diffusion
        """
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        pipe.enable_attention_slicing()

        self.pipe = pipe.to("mps")

    def generate(
            self,
            prompt: str,
            output_path: Optional[Path] = None,
            negative_prompt: Optional[str] = None,
            num_inference_steps: int = 50,
            seed: Optional[int] = None
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            output_path: Optional path to save the generated image
            negative_prompt: Optional text to guide what the image should not contain
            num_inference_steps: Number of denoising steps (higher = better quality, slower)
            seed: Optional random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        if seed is not None:
            torch.manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
        ).images[0]

        if output_path:
            image.save(output_path)

        return image


def main():
    generator = ImageGenerator()
    image = generator.generate(
        num_inference_steps=30,
        prompt="A serene landscape with a small cabin in the woods",
        output_path=Path("output.jpg")
    )

    image.show()


if __name__ == '__main__':
    main()
