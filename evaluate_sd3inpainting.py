import torch
from diffusers.utils import load_image, check_min_version
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline, StableDiffusion3Pipeline
from diffusers.models.controlnet_sd3 import SD3ControlNetModel
from eval.multiedit_dataset import multiedit_DATASET
import json
import os
import sys
from tqdm import tqdm
import argparse
import PIL.Image as Image
parser = argparse.ArgumentParser(description="PixArt Inpainting Evaluation")
parser.add_argument('--gpu', dest="GPU_IDX", type=int, default=0, help='GPU index')
parser.add_argument('--result_dir', type=str, default='./outputs_bld', 
                    help='Result directory')
parser.add_argument('--dataset', type=str, 
                    default='path-to-json', 
                    help='Path to the dataset')
parser.add_argument('--seed', type=int, default=334, help='Random seed')
parser.add_argument('--resolution', type=int, default=1024, help='Resolution')
parser.add_argument('--use_inpainting', action='store_true', default=False, help='Use inpainting')

args = parser.parse_args()
access_token = "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"
if args.use_inpainting:
    controlnet = SD3ControlNetModel.from_pretrained(
        "alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1, 
    )
    pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.text_encoder.to(torch.float16)
    pipe.controlnet.to(torch.float16)
    pipe.to("cuda")

else:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, token=access_token)
    pipe.to("cuda")

try:
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"Error loading dataset {args.dataset}: {e}")
    sys.exit(1)

dataset = {k: dataset[k] for k in list(dataset.keys())[:100]}
# dataset = {k: dataset[k] for k in list(dataset.keys())[args.shard:(args.shard+1)]}
dataset = multiedit_DATASET(dataset, in_pipeline=True)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
for idx, data_ in tqdm(enumerate(val_loader), total=len(val_loader), desc="Generating images"):
    idx = idx
    to_save_prompt = ''
    save_dir = os.path.join(args.result_dir, 'gen', f'IDX{idx}')
    os.makedirs(save_dir, exist_ok=True)
    background_caption = data_['background_prompt'][0]
    if not args.use_inpainting:
            image = pipe(
                background_caption,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
                height=args.resolution,
                width=args.resolution,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
            ).images[0]
            image.save(os.path.join(save_dir, f'IDX{idx}_layer_background.png'))
            continue
    for layer in range(data_['total_layers']):
        mask = Image.fromarray(data_['bboxes'][0, layer, :, :].numpy().squeeze()).resize((args.resolution, args.resolution)).convert('L')
        background_caption = data_['background_prompt'][0]
        foreground_caption = data_['local_prompts'][layer][0]
        
        if layer == 0:
            init_image = Image.open(os.path.join(save_dir, f'IDX{idx}_layer_background.png')).resize((args.resolution, args.resolution))
            res = pipe(
                negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
                prompt=foreground_caption,
                height=args.resolution,
                width=args.resolution,
                control_image=init_image,
                control_mask=mask,
                num_inference_steps=28,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                controlnet_conditioning_scale=0.95,
                guidance_scale=7,
            ).images[0]
            res.save(os.path.join(save_dir, f'IDX{idx}_layer{layer}.png'))
            print(f"saved image at {os.path.join(save_dir, f'IDX{idx}_layer{layer}.png')}")
        else:
            init_image = Image.open(os.path.join(save_dir, f'IDX{idx}_layer{layer-1}.png'))
            res = pipe(
                negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
                prompt=foreground_caption,
                height=args.resolution,
                width=args.resolution,
                control_image=init_image,
                control_mask=mask,
                num_inference_steps=28,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                controlnet_conditioning_scale=0.95,
                guidance_scale=7,
            ).images[0]
            res.save(os.path.join(save_dir, f'IDX{idx}_layer{layer}.png'))
            print(f"saved image at {os.path.join(save_dir, f'IDX{idx}_layer{layer}.png')}")

