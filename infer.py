import torch
from pipeline_flux_regional import RegionalFluxPipeline, RegionalFluxAttnProcessor2_0
from pipeline_flux_controlnet_regional import RegionalFluxControlNetPipeline
from diffusers import FluxControlNetModel, FluxMultiControlNetModel
import re

def generate_prompt_pairs(regional_prompt, base_size=512):
    pattern = r'(\d+\..*?)(?=\s*\d+\.|$)'
    steps = re.findall(pattern, regional_prompt, re.DOTALL)
    steps = [step.strip() for step in steps]
    
    regional_prompt_mask_pairs = {}
    
    for i, description in enumerate(steps):
     
        y_start = i * base_size
        x_start = 0
        y_end = (i + 1) * base_size
        x_end = base_size
        
        regional_prompt_mask_pairs[str(i)] = {
            "description": description,
            "mask": [x_start, y_start, x_end, y_end] 
        }
        
    return regional_prompt_mask_pairs, len(steps)

if __name__ == "__main__":
    
    model_path = "black-forest-labs/FLUX.1-dev"
    
    use_lora = True
    pipeline = RegionalFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

    if use_lora:       
        pipeline.load_lora_weights("your_path_to_lora", weight_name="adapter.safetensors")

    attn_procs = {}
    for name in pipeline.transformer.attn_processors.keys():
        if 'transformer_blocks' in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalFluxAttnProcessor2_0()
        else:
            attn_procs[name] = pipeline.transformer.attn_processors[name]
    pipeline.transformer.set_attn_processor(attn_procs)

    ## generation settings
    regional_prompt = "[RECIPE] This classic Kung Pao Chicken dish is a spicy and flavorful combination of marinated chicken, aromatic spices, and crispy peanuts, all stir-fried to perfection.1. Preparing the chicken thighs: Fresh chicken thighs are carefully deboned and cut into 1.5 cm square pieces, their skin glistening with a pinkish hue. The pieces are uniform, ready to absorb the flavors of the marinade. 2. Marinating the chicken: The chicken pieces are coated in a glossy mixture of salt, golden cooking wine, a frothy egg white, and translucent wet starch. The marinade clings to the meat, promising tenderness and flavor. 3. Preparing the ingredients: Crisp, pale green scallions are trimmed into 10 cm segments, their fresh aroma filling the air. They lie neatly on the cutting board, ready to add a burst of color and flavor to the dish. 4.Stir-frying peppercorns and chilies: A wok sizzles with golden peanut oil, releasing the earthy aroma of browning peppercorns. Dried chilies turn a deep, fiery red, while thin slices of garlic add a fragrant note to the mix. 5. Stir-frying chicken and seasoning:Caption: The marinated chicken cubes sizzle in the wok, turning from pink to a golden brown as they cook. Dark soy sauce, vinegar, and sugar create a rich, glossy coating, while scallion segments add a vibrant green contrast. 6. Adding final ingredients: The dish is finished with a generous sprinkle of golden-brown fried peanuts and a drizzle of vibrant red oil. A blend of crispy peanuts, tender chicken, red peppers, green onions and aromatic spices. "
    base_prompt = "Kung Pao Chicken"
    

    regional_prompt_mask_pairs, num_steps = generate_prompt_pairs(regional_prompt, base_size=512)
    
    image_width = 512
    image_height = num_steps * 512  # 根据步骤数量动态计算高度
    print(f"Generating image with size: {image_width}x{image_height} for {num_steps} steps.")

    num_samples = 1
    num_inference_steps = 24
    guidance_scale = 3.5
    seed = 42 

    # region control settings
    mask_inject_steps = 24
    double_inject_blocks_interval = 1
    single_inject_blocks_interval = 1
    base_ratio = 0.1

    # ensure image width and height are divisible by the vae scale factor
    image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
    image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor

    regional_prompts = []
    regional_masks = []
    

    background_mask = torch.ones((image_height, image_width))

    for region_idx, region in regional_prompt_mask_pairs.items():
        description = region['description']
        mask_coords = region['mask']
    
        x1, y1, x2, y2 = mask_coords

        mask = torch.zeros((image_height, image_width))
        mask[y1:y2, x1:x2] = 1.0

        background_mask -= mask
        regional_prompts.append(description)
        regional_masks.append(mask)
            
    # if regional masks don't cover the whole image, append background prompt and mask
    background_prompt = "" 
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    # setup regional kwargs
    joint_attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'double_inject_blocks_interval': double_inject_blocks_interval,
        'single_inject_blocks_interval': single_inject_blocks_interval,
        'base_ratio': base_ratio,
    }
    
    # generate images
    generator = torch.Generator("cuda").manual_seed(seed)
    
    output = pipeline(
            prompt=base_prompt,
            regional_prompt_sp=regional_prompt,
            width=image_width, 
            height=image_height,
            mask_inject_steps=mask_inject_steps,
            num_inference_steps=num_inference_steps,
            generator=generator,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        images = output.images

    for idx, img in enumerate(images):
        save_path = f"output_{idx}.jpg"
        img.save(save_path)
        print(f"Saved image to {save_path}")