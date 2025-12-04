# CookAnything

**CookAnything: A Framework for Flexible and Consistent Multi-Step Recipe Image Generation** *Accepted at ACM Multimedia 2025 (MM'25)*

![Model Architecture](images/model.png)

## Inference

Follow the steps below to set up the environment and run inference.

### 1. Prerequisites
First, you need to install the dependencies from the **Regional-Prompting-FLUX** repository. Please follow the installation instructions provided there:

* [https://github.com/instantX-research/Regional-Prompting-FLUX](https://github.com/instantX-research/Regional-Prompting-FLUX)

### 2. Implement Flexible RoPE
Modify the RoPE (Rotary Positional Embeddings) logic to support **Flexible RoPE**. 

Locate the latent generation section in your code and update it with the following logic:

```python
# Initialize latents with flexible dimensions
latents = randn_tensor(
    (1, num_channels_latents, int(height/width), width, width), 
    generator=generator, 
    device=device, 
    dtype=dtype
).repeat(batch_size, 1, 1, 1, 1).reshape((shape))

latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

# Prepare latent image IDs
latent_image_ids = self._prepare_latent_image_ids(batch_size, width, width, device, dtype)
latent_image_ids = latent_image_ids.repeat(int(height/width), 1)
```
### 3. Run Inference
Ensure that pipeline_flux_region.py and infer.py are located in your working directory. You can then run the inference script using the following command:
```python
python infer.py
```
