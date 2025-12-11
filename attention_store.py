from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import Attention

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, should_capture: bool = True):
        """
        Store attention maps selectively.
        
        Args:
            attn: Attention probabilities tensor
            is_cross: Whether this is cross-attention
            place_in_unet: Location identifier (down/mid/up)
            should_capture: Whether to actually store this attention (for selective capture)
        """
        if self.cur_att_layer >= 0 and is_cross and should_capture:
            key = f"{place_in_unet}"
            # Store cross-attention maps
            self.step_store[key].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                # Only add if both stores have items for this key
                if key in self.step_store and len(self.step_store[key]) > 0:
                    # Handle case where number of items might differ
                    min_len = min(len(self.attention_store[key]), len(self.step_store[key]))
                    for i in range(min_len):
                        # Only add if shapes match (different shapes occur in inpainting vs regular generation)
                        if self.attention_store[key][i].shape == self.step_store[key][i].shape:
                            self.attention_store[key][i] += self.step_store[key][i]
        self.cur_step += 1
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # Avoid division by zero
        divisor = max(self.cur_step, 1)
        average_attention = {key: [item / divisor for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def aggregate_attention(self, from_where: List[str], is_cross: bool, prompts: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        expected_patches = np.prod(self.attn_res)
        
        # Check if we have any attention maps
        if not attention_maps or all(len(attention_maps.get(f"{loc}", [])) == 0 for loc in from_where):
            raise ValueError(f"No attention maps found. Store keys: {list(attention_maps.keys())}, "
                           f"Looking for: {from_where}. Make sure attention maps are being collected.")
        
        for location in from_where:
            key = f"{location}"
            if key not in attention_maps or len(attention_maps[key]) == 0:
                continue  # Skip if no maps for this location
                
            for item in attention_maps[key]:
                # item shape: [batch_size * num_heads, num_patches, num_tokens]
                num_patches = item.shape[1]
                
                # Handle both full and partial (masked) attention maps
                if num_patches == expected_patches:
                    # Full attention map - reshape normally
                    batch_size = len(prompts)
                    num_heads = item.shape[0] // batch_size
                    num_tokens = item.shape[-1]
                    
                    cross_maps = item.reshape(batch_size, num_heads, -1, num_tokens)
                    cross_maps = cross_maps.reshape(batch_size, num_heads, self.attn_res[0], self.attn_res[1], num_tokens)
                    out.append(cross_maps)
                elif num_patches < expected_patches:
                    # Partial attention map (from masked inpainting)
                    # Average over patches to get a per-token attention score
                    batch_size = len(prompts)
                    num_heads = item.shape[0] // batch_size  
                    num_tokens = item.shape[-1]
                    
                    # Average attention across the masked patches: [batch*heads, patches, tokens] -> [batch*heads, tokens]
                    cross_maps = item.mean(dim=1)  # Average over patches
                    # Reshape: [batch*heads, tokens] -> [batch, heads, 1, 1, tokens]
                    cross_maps = cross_maps.reshape(batch_size, num_heads, num_tokens)
                    # Broadcast to spatial dims: [batch, heads, H, W, tokens]
                    cross_maps = cross_maps.unsqueeze(2).unsqueeze(3).expand(batch_size, num_heads, self.attn_res[0], self.attn_res[1], num_tokens)
                    out.append(cross_maps)
                # else: skip attention maps with more patches than expected
        
        if len(out) == 0:
            raise ValueError(f"No valid attention maps collected from {from_where}. "
                           f"Expected {expected_patches} patches or fewer. "
                           f"Make sure attention capture is enabled.")
        
        # Concatenate across layers and average
        out = torch.cat(out, dim=1)  # Concatenate on head dimension
        # Average across batch and heads: [batch_size, total_heads, H, W, num_tokens] -> [H, W, num_tokens]
        out = out.mean(dim=(0, 1))
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.cur_step = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.cur_step = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res


class AttentionProcessor:
    def __init__(self, attnstore, place_in_unet, capture_mode="all"):
        """
        Initialize AttentionProcessor with selective capture support.
        
        Args:
            attnstore: AttentionStore instance
            place_in_unet: Location identifier (down/mid/up)
            capture_mode: "all" to capture all attention, "selective" to only capture marked calls
        """
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.capture_mode = capture_mode
        self.call_count = 0  # Track number of calls within same layer
        self.query_mode = None  # For compatibility with attention_processor features
    
    def set_query_mode(self, mode: str = None, vt: float = 0.9):
        """Set query mode - stub for compatibility with attention_processor features"""
        self.query_mode = mode
        self.vt = vt
    
    def clear_query_cache(self):
        """Clear query cache - stub for compatibility"""
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        capture_attention=None,  # Explicit parameter so it passes through Attention.forward() filtering
        **kwargs,
    ):
        # For cross-attention, use encoder_hidden_states sequence length
        # For self-attention, use hidden_states sequence length
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Determine if we should capture this attention
        if self.capture_mode == "all":
            should_capture = True  # Always capture in "all" mode
        else:  # selective mode
            # Only capture if explicitly requested (default False in selective mode)
            should_capture = capture_attention if capture_attention is not None else False
        
        self.attnstore(attention_probs, is_cross, self.place_in_unet, should_capture=should_capture)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states