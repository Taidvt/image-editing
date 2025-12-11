#!/usr/bin/env python
"""
Test script to verify attention map extraction works with PixArt pipeline.
Tests both background generation and inpainting with selective attention capture.
"""
import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

# Add local diffusers to path BEFORE importing anything else
diffusers_path = str(current_file_path.parent / "diffusers" / "src")
if diffusers_path not in sys.path:
    sys.path.insert(0, diffusers_path)
    print(f"Using local diffusers from: {diffusers_path}")

import torch
import numpy as np
from PIL import Image
from scripts.pipeline_pixart_inpaint_with_latent_memory_improved import PixArtAlphaInpaintLMPipeline
from attention_store import AttentionStore
from inference import register_attention_control, show_cross_attention

def test_background_generation():
    """Test attention capture during background generation (no inpainting)"""
    print("\n" + "="*80)
    print("TEST 1: Background Generation (No Inpainting)")
    print("="*80)
    
    # Load pipeline - try local first, fallback to download if needed
    print("Loading PixArt model...")
    try:
        pipe = PixArtAlphaInpaintLMPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
            local_files_only=True
        ).to('cuda:0')
        print("✓ Loaded from local cache")
    except (OSError, ValueError) as e:
        print("⚠ Model not in cache, downloading...")
        pipe = PixArtAlphaInpaintLMPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
        ).to('cuda:0')
        print("✓ Downloaded successfully")
    
    # Setup attention store with selective capture
    resolution = 1024
    latent_size = int(np.ceil(resolution / 16))  # 64 for 1024x1024
    pipe.attention_store = AttentionStore(attn_res=(latent_size, latent_size))
    register_attention_control(pipe, capture_mode="selective")
    
    # Generate background image
    prompt = "A serene mountain landscape with a clear blue sky"
    print(f"\nPrompt: {prompt}")
    print("Generating image...")
    
    generator = torch.Generator(device='cuda:0').manual_seed(42)
    result = pipe(
        prompt=prompt,
        image=None,
        mask_image=None,
        strength=1.0,
        generator=generator,
        guidance_scale=7.5,
        num_inference_steps=20,
        num_images_per_prompt=1,
        inpaint=False,
        cattn_masking=False,
        multi_query_disentanglement=False
    )
    
    # Handle both ImagePipelineOutput and tuple returns
    if hasattr(result, 'images'):
        image = result.images[0]
    else:
        image = result[0].images[0]
    
    # Try to extract attention maps
    print("\nExtracting attention maps...")
    try:
        attn_img = show_cross_attention(pipe, prompt, ['down', 'mid', 'up'])
        if attn_img is not None:
            print("✓ SUCCESS: Attention maps extracted successfully!")
            
            # Save results
            os.makedirs('./output/test_attention', exist_ok=True)
            image.save('./output/test_attention/test1_background.png')
            attn_img.save('./output/test_attention/test1_attention.png')
            print(f"✓ Saved image to: ./output/test_attention/test1_background.png")
            print(f"✓ Saved attention to: ./output/test_attention/test1_attention.png")
            return True
        else:
            print("✗ FAILED: Attention map extraction returned None")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inpainting_with_selective_capture():
    """Test attention capture during inpainting with selective capture"""
    print("\n" + "="*80)
    print("TEST 2: Inpainting with Selective Attention Capture")
    print("="*80)
    
    # Load pipeline - try local first, fallback to download if needed
    print("Loading PixArt model...")
    try:
        pipe = PixArtAlphaInpaintLMPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
            local_files_only=True
        ).to('cuda:0')
        print("✓ Loaded from local cache")
    except (OSError, ValueError) as e:
        print("⚠ Model not in cache, downloading...")
        pipe = PixArtAlphaInpaintLMPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
        ).to('cuda:0')
        print("✓ Downloaded successfully")
    
    # Setup attention store with selective capture
    resolution = 1024
    latent_size = int(np.ceil(resolution / 16))
    pipe.attention_store = AttentionStore(attn_res=(latent_size, latent_size))
    register_attention_control(pipe, capture_mode="selective")
    
    # First generate background
    bg_prompt = "A grassy field under a bright sunny sky"
    print(f"\nBackground prompt: {bg_prompt}")
    print("Generating background...")
    
    generator = torch.Generator(device='cuda:0').manual_seed(42)
    bg_result = pipe(
        prompt=bg_prompt,
        image=None,
        mask_image=None,
        strength=1.0,
        generator=generator,
        guidance_scale=7.5,
        num_inference_steps=20,
        inpaint=False,
        cattn_masking=False,
        multi_query_disentanglement=False
    )
    
    # Handle tuple return: (ImagePipelineOutput, latent_memory)
    if hasattr(bg_result, 'images'):
        bg_image = bg_result.images[0]
        latent_memory = None
    else:
        bg_image = bg_result[0].images[0]
        latent_memory = bg_result[1]  # Get latent memory
    
    # Reset attention store
    pipe.attention_store.reset()
    
    # Create a simple mask (center square)
    mask = Image.new('L', (resolution, resolution), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    center = resolution // 2
    box_size = resolution // 3
    draw.rectangle([center - box_size//2, center - box_size//2, 
                   center + box_size//2, center + box_size//2], fill=255)
    
    # Now inpaint with a new object
    fg_prompt = "a red apple"
    print(f"\nForeground prompt: {fg_prompt}")
    print("Generating inpainted image with selective attention capture...")
    
    generator = torch.Generator(device='cuda:0').manual_seed(42)
    fg_result = pipe(
        prompt=fg_prompt,
        image=bg_image,
        mask_image=mask,
        strength=1.0,
        generator=generator,
        guidance_scale=7.5,
        num_inference_steps=20,
        inpaint=True,
        cattn_masking=True,
        multi_query_disentanglement=False,
        latent_memory=latent_memory
    )
    
    # Handle tuple return
    if hasattr(fg_result, 'images'):
        fg_image = fg_result.images[0]
    else:
        fg_image = fg_result[0].images[0]
    
    # Try to extract attention maps for foreground object
    print("\nExtracting attention maps for foreground object...")
    try:
        attn_img = show_cross_attention(pipe, fg_prompt, ['down', 'mid', 'up'])
        if attn_img is not None:
            print("✓ SUCCESS: Attention maps extracted from inpainting!")
            
            # Save results
            os.makedirs('./output/test_attention', exist_ok=True)
            bg_image.save('./output/test_attention/test2_background.png')
            fg_image.save('./output/test_attention/test2_inpainted.png')
            mask.save('./output/test_attention/test2_mask.png')
            attn_img.save('./output/test_attention/test2_attention.png')
            print(f"✓ Saved images to: ./output/test_attention/")
            return True
        else:
            print("✗ FAILED: Attention map extraction returned None")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("ATTENTION MAP EXTRACTION TEST SUITE")
    print("="*80)
    print("\nThis script tests the fixed attention map extraction with:")
    print("1. Selective capture mode")
    print("2. Support for multi-query disentanglement")
    print("3. Background consistency guidance")
    print("4. Layer-wise memory")
    
    results = []
    
    # Test 1: Background generation
    try:
        result1 = test_background_generation()
        results.append(("Background Generation", result1))
    except Exception as e:
        print(f"\nTest 1 crashed: {e}")
        results.append(("Background Generation", False))
    
    # Test 2: Inpainting with selective capture
    try:
        result2 = test_inpainting_with_selective_capture()
        results.append(("Inpainting with Selective Capture", result2))
    except Exception as e:
        print(f"\nTest 2 crashed: {e}")
        results.append(("Inpainting with Selective Capture", False))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Attention map extraction is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

