from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # or "cpu" if no GPU

prompt = "A futuristic cityscape at night"
image = pipe(prompt).images[0]
image.save("generated_image.png")

print("✅ Image generated from prompt.")
