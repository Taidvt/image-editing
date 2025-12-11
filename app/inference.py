#!/usr/bin/env python
from __future__ import annotations
import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import random
import gradio as gr
import numpy as np
import uuid
import argparse
from diffusers import DPMSolverMultistepScheduler, DPMSolverSDEScheduler
import torch
from typing import Tuple
from datetime import datetime
from diffusion.sa_solver_diffusers import SASolverScheduler
from omegaconf import OmegaConf
from PIL import Image
from scripts.pipeline_pixart_inpaint_with_latent_memory_improved import PixArtAlphaInpaintLMPipeline
from transformers import T5EncoderModel, T5Tokenizer
import sys
sys.path.append('../')

# Function to save generated images
def save_image(image, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    image.save(file_path)

def replace_latent(args, replace):
    if replace:
        args.latent_memory = args.prev_latent_memory
    else:
        pass

def set_scheduler(pipe, scheduler_type, dpms_guidance_scale, sas_guidance_scale):
    if scheduler_type == 'DPM-Solver':
        pipe.scheduler = DPMSolverMultistepScheduler()
        guidance_scale = dpms_guidance_scale
    elif scheduler_type == 'SA-Solver':
        pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')
        guidance_scale = sas_guidance_scale
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return guidance_scale

# Function to ensure the image is in the correct format (PIL, numpy, or tensor)
def process_image(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)  # Convert numpy array to PIL Image
    elif isinstance(image, torch.Tensor):
        return Image.fromarray(image.cpu().numpy())  # Convert tensor to PIL Image
    elif isinstance(image, Image.Image):
        return image  # Already a PIL image, no need to convert
    else:
        raise ValueError(f"Unsupported image format: {type(image)}")

def str_to_bool(s):
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError(f"Unsupported boolean value: {s}")

# Function to run the pipeline (without inpainting)
def run_pipeline(pipe, args, prompt: str, image=None, mask_image=None, inpaint: bool=False, cattn_masking: bool=False, 
                multi_query_disentanglement: bool=False, alpha: float=1.0, object_gen=False, utilize_cache=False,
                subject_token_idx: list=None, sigma: float=0.0, remove_vanilla_ratio: float=0.5,
                new_generation: bool = True, remove_checkbox: bool = False):
    # Ensure the image is in the correct format
    if args.counter > 1:
        latent_memory = args.latent_memory[-1]
    elif utilize_cache is True:
        latent_memory = args.latent_memory[-1]
    else:
        latent_memory=None
    
    if new_generation is not True:
        args.latent_memory.pop(-1)
        latent_memory = args.latent_memory[-1]
    if remove_checkbox is True:
        latent_memory = [args.latent_memory[-1], args.latent_memory[-3]]
        
    generator = torch.Generator(device=f'cuda:{args.GPU_IDX}').manual_seed(334)  # Fixed seed for consistency
    guidance_scale = set_scheduler(pipe, args.scheduler_type, args.dpms_guidance_scale, args.sas_guidance_scale)
    
    if remove_checkbox is False:
        result_tensor, latent_memory_new = pipe(
            prompt=prompt,
            image=None if inpaint else None,
            mask_image=mask_image,
            strength=1.0,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            inpaint=inpaint,
            latent_memory=latent_memory,
            vanilla_ratio=args.vanilla_ratio,
            cattn_masking=cattn_masking,
            subject_token_idx=subject_token_idx,
            sigma=sigma,
            remove_vanilla_ratio=remove_vanilla_ratio,
            multi_query_disentanglement=multi_query_disentanglement,
            alpha=alpha,
            object_gen=object_gen,
            utilize_cache=utilize_cache,
            new_generation=new_generation,
        )
    else: 
        result_tensor, latent_memory_new = pipe(
            prompt=prompt,
            image=None if inpaint else None,
            mask_image=mask_image,
            strength=1.0,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            inpaint=inpaint,
            latent_memory=latent_memory,
            vanilla_ratio=args.vanilla_ratio,
            cattn_masking=cattn_masking,
            subject_token_idx=subject_token_idx,
            sigma=sigma,
            remove_vanilla_ratio=remove_vanilla_ratio,
            multi_query_disentanglement=multi_query_disentanglement,
            alpha=alpha,
            object_gen=object_gen,
            utilize_cache=utilize_cache,
            new_generation=new_generation,
            remove_prev=remove_checkbox,
        )
    
    result_tensor = result_tensor.images
    if object_gen is True and utilize_cache is False:
        pass
    else:
        args.latent_memory.append(latent_memory_new)
    
    return result_tensor[0]

def main():
    parser = argparse.ArgumentParser(description="PixArt Inpainting Demo")
    parser.add_argument('--model_version', type=str, default='PixArtAlpha', help='Model version')
    parser.add_argument('--model_base', type=str, default='InpaintLM', help='Model base')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--resolution', type=int, default=1024, help='Resolution')
    parser.add_argument('--dpms_guidance_scale', type=float, default=7.5, help='DPM-Solver guidance scale')
    parser.add_argument('--sas_guidance_scale', type=float, default=3.0, help='SA-Solver guidance scale')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--prompt', type=str, default='', help='Prompt')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt')
    parser.add_argument('--scheduler_type', type=str, default='DPM-Solver', help='Scheduler type')
    parser.add_argument('--GPU_IDX', type=int, default=0, help='GPU index')
    parser.add_argument('--result_dir', type=str, default='./output', help='Result directory')
    parser.add_argument('--exp_name', type=str, default='PixArtInpaintExp', help='Experiment name')
    parser.add_argument('--vanilla_ratio', type=float, default=0.05, help='Vanilla ratio')
    parser.add_argument('--cattn_masking', type=str, default='True', help='Cross attention masking')
    parser.add_argument('--multi_query_disentanglement', type=str, default='True', help='Multi-query disentanglement')
    parser.add_argument('--object_gen', type=str, default='False', help='Object generation')
    parser.add_argument('--utilize_cache', type=str, default='False', help='Utilize cache')
    parser.add_argument('--remove_vanilla_ratio', type=float, default=0.5, help='Remove previous object ratio')
    
    args = parser.parse_args()
    args.latent_memory = []
    args.prev_latent_memory = None
    args.counter = 0
    args.remove_checkbox = False

    # Save configuration
    os.makedirs(f"{args.result_dir}/{args.exp_name}", exist_ok=True)
    OmegaConf.save(config=vars(args), f=f"{args.result_dir}/{args.exp_name}/configs.yaml")

    # Load PixArt Inpainting Pipeline
    pipe_inpaint = PixArtAlphaInpaintLMPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16
    ).to(f'cuda:{args.GPU_IDX}')
    
    # Define the generation function
    def generate(prompt=None, mask=None, image=None, scheduler=None, dpms_guidance_scale=None, sas_guidance_scale=None, 
                vanilla_ratio_input=0.05, cattn_masking='True', multi_query_disentanglement_input='True', alpha_input=1.0, 
                object_gen_input='False', utilize_cache='False', subject_token_idx=None, sigma=0.0, 
                remove_vanilla_ratio=0.5, new_generation='True', reset_checkbox=False, remove_checkbox=False):
        
        args.scheduler_type = scheduler if scheduler else args.scheduler_type
        args.dpms_guidance_scale = dpms_guidance_scale if dpms_guidance_scale else args.dpms_guidance_scale
        args.sas_guidance_scale = sas_guidance_scale if sas_guidance_scale else args.sas_guidance_scale
        args.vanilla_ratio = vanilla_ratio_input
        args.cattn_masking = str_to_bool(cattn_masking)
        args.multi_query_disentanglement = str_to_bool(multi_query_disentanglement_input)
        args.alpha = alpha_input
        args.object_gen = str_to_bool(object_gen_input)
        args.utilize_cache = str_to_bool(utilize_cache)
        args.remove_checkbox = remove_checkbox
        args.remove_vanilla_ratio = remove_vanilla_ratio

        new_generation = str_to_bool(new_generation)
        if len(subject_token_idx) != 0:
            subject_token_idx = [int(i.strip()) for i in subject_token_idx.split(",")]
        else:
            subject_token_idx = None
        sigma = float(sigma)
        
        if reset_checkbox is True:
            args.counter = 0
            args.latent_memory = []
            pipe_inpaint.layerstore._empty_all_layers()
        
        if args.counter == 0:
            args.counter += 1
            return run_pipeline(pipe_inpaint, args, prompt, image=None, mask_image=None, inpaint=False, 
                             object_gen=args.object_gen, utilize_cache=args.utilize_cache)
        
        if isinstance(mask, dict):
            mask = Image.fromarray(np.array(mask["composite"])[:, :, 3]).convert("L").resize((args.resolution, args.resolution))

        args.counter += 2
        output_image = run_pipeline(pipe_inpaint, args, prompt, image, mask, inpaint=True, 
                                 cattn_masking=args.cattn_masking, multi_query_disentanglement=args.multi_query_disentanglement, 
                                 alpha=args.alpha, object_gen=args.object_gen, utilize_cache=args.utilize_cache, 
                                 subject_token_idx=subject_token_idx, sigma=sigma, remove_vanilla_ratio=remove_vanilla_ratio, 
                                 new_generation=new_generation, remove_checkbox=args.remove_checkbox)
        return output_image
    
    outputs = []
    output = generate(prompt=prompt, mask=mask, image=None, scheduler=args.scheduler_type, dpms_guidance_scale=args.dpms_guidance_scale, sas_guidance_scale=args.sas_guidance_scale
                      vanilla_ratio_input=args.vanilla_ratio_input, cattn_masking=args.cattn_masking, multi_query_disentanglement_input=args.multi_query_disentanglement_input, alpha_input=args.alpha_input,
                      object_gen_input=args.object_gen_input, utilize_cache=args.utilize_cache, subject_token_idx=args.subject_token_idx, sigma=args.sigma, remove_vanilla_ratio=args.remove_vanilla_ratio,
                      new_generation=args.new_generation, reset_checkbox=args.reset_checkbox, remove_checkbox=args.remove_checkbox)
    
if __name__ == '__main__':
    main()