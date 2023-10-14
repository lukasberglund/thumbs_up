import os
from typing import List
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import shutil
import pdb



device = "cuda" if torch.cuda.is_available() else "cpu"

celebrities = [
    # Female Celebrities
    "Viola Davis", "Priyanka Chopra", "Cate Blanchett", "Lupita Nyong'o", 
    "Lucy Liu", "Salma Hayek", "Charlize Theron", "Mindy Kaling", 
    "Penélope Cruz", "Gal Gadot",
    
    # Male Celebrities
    "Idris Elba", "Rami Malek", "Jackie Chan", "Mahershala Ali", 
    "Javier Bardem", "Chris Hemsworth", "Will Smith", "Dev Patel", 
    "Ken Watanabe", "Chadwick Boseman"
]

import torch
from typing import List

def evaluate_run_with_refiner(prompts: List[str], num_images_per_caption: int, lora_path: str = None, refiner_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0", base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", use_refiner: bool = True) -> List[torch.Tensor]:
    """
    Generates refined images based on user-provided captions using LoRA parameters and a refiner.
    
    Parameters:
    - captions (List[str]): A list of strings containing the captions based on which images are to be generated.
    - lora_model_id (str): ID pointing to the LoRA model.
    - refiner_path (str, optional): Path to the refiner model. Default is "stabilityai/stable-diffusion-xl-refiner-1.0".
    - use_refiner (bool, optional): Flag to indicate whether to use a refiner model. Default is True.
    
    Returns:
    - List of generated refined images corresponding to the provided captions.
    """
    shutil.rmtree("/root/.cache/huggingface/hub")
    # Load the base pipeline and move to GPU
    pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    if lora_path is not None:
        pipe.load_lora_weights(lora_path)

    latent_images_list = []
    generator = torch.Generator(device=device).manual_seed(0)

    # First loop: Generate the latent images
    for prompt in prompts:
        prompt_batch = [prompt] * num_images_per_caption
        latent_images = pipe(prompt=prompt_batch, generator=generator, output_type="latent").images if use_refiner else pipe(prompt=prompt_batch, generator=generator).images
        latent_images_list.append(latent_images)
    
    del pipe
    torch.cuda.empty_cache()

    if not use_refiner:
        # breakpoint()
        return [img for batch in latent_images_list for img in batch]

    shutil.rmtree("/root/.cache/huggingface/hub")

    # Load the refiner and move to GPU
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    )
    refiner.to(device)

    refined_images = []

    # Second loop: Refine the latent images using the refiner
    for i, prompt in enumerate(prompts):
        prompt_batch = [prompt] * num_images_per_caption
        
        # Refine the image using the refiner
        refined_image_batch = refiner(prompt=prompt_batch, image=latent_images_list[i], generator=generator).images
        refined_images.extend(refined_image_batch)

    del refiner
    torch.cuda.empty_cache()

    del latent_images_list
    torch.cuda.empty_cache()

    return refined_images

def evaluate_run(lora_dir: str, use_refiner: bool=True):
    if lora_dir is None:
        captions = [f'a photo of {celebrity} showing thumbs up' for celebrity in celebrities]
    else:
        captions = [f'a photo of {celebrity} showing <thumbs_up> thumbs up' for celebrity in celebrities]
    
    lora_path = os.path.join(lora_dir, "pytorch_lora_weights.safetensors") if lora_dir is not None else None
    # Generate refined images based on the provided captions
    images = evaluate_run_with_refiner(captions, num_images_per_caption=1, lora_path=lora_path, use_refiner=use_refiner)
    
    save_dir = os.path.join(lora_dir, "images", ("refined" if use_refiner else "unrefined")) if lora_dir is not None else (os.path.join("logs/baseline", ("refined" if use_refiner else "unrefined")))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save the images
    for celebrity, image in zip(celebrities, images):
        celebrity_snake_case = celebrity.lower().replace(" ", "_")
        image_name = f"{celebrity_snake_case}.png"
        image.save(os.path.join(save_dir, image_name))

def main():
    # get all dirs in logs that start with sweep_final
    # evaluate_run(None, use_refiner=False)
    evaluate_run(None, use_refiner=True)

    # for path in os.listdir("logs"):
    #     if path.startswith("sweep_final"):
    # #         lora_dir = os.path.join("logs", path)
    #         evaluate_run(lora_dir, use_refiner=False)


if __name__ == "__main__":
    main()
