#!/usr/bin/env python3
"""
Simple inference script for Qwen3-VL model with image support.
Supports both text-only and multimodal (image + text) generation.
Based on examples/multimodal/run_text_generation.py pattern.
"""

import os
import sys
import torch
import time
from pathlib import Path
from argparse import Namespace
from typing import List, Optional
from PIL import Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__)))
)


from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.vision.qwen3_vit import get_num_image_embeddings
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.inference_request import Qwen3VLInferenceRequest
from megatron.core.inference.text_generation_controllers.vlm_text_generation_controller import (
    Qwen3VLTextGenerationController,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    Qwen3VLInferenceWrapper,
)
from pretrain_qwen3_vl import model_provider


def add_inference_args(parser):
    """Add inference-specific arguments for multimodal models."""
    group = parser.add_argument_group(title='vision language model inference')
    
    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top-k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top-p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=50,
        help='Number of tokens to generate',
    )
    group.add_argument(
        "--prompts",
        metavar='PROMPT',
        type=str,
        nargs='+',
        default=["What is in this image?"],
        help='Input text prompts for generation. Use quotes for multi-word prompts: --prompts "prompt 1" "prompt 2"',
    )
    group.add_argument(
        "--image-paths",
        metavar='N',
        type=str,
        nargs='*',
        default=[],
        help='Paths to input images (optional). If provided, must match number of prompts.',
    )
    group.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help='Maximum batch size for inference',
    )
    group.add_argument(
        "--use-tiling",
        action="store_true",
        default=False,
        help='Use image tiling for higher resolution images',
    )
    group.add_argument(
        "--max-num-tiles",
        type=int,
        default=4,
        help='Maximum number of tiles for image tiling',
    )
    group.add_argument(
        "--use-thumbnail",
        action="store_true",
        default=False,
        help='Use thumbnail for image tiling',
    )
    group.add_argument(
        '--freeze-LM', action='store_true', default=False, help="Freeze language model weights"
    )
    group.add_argument(
        '--freeze-ViT', action='store_true', default=False, help="Freeze vision model (ViT) weights"
    )
    group.add_argument(
        "--disable-vision-class-token",
        action="store_true",
        default=False,
        help="Drop vision model class token",
    )
    group.add_argument("--dataloader-seq-length", type=int, help="Make dataloader to produce sequences of specific length.")
    group.add_argument("--decoder-tp-comm-overlap", action="store_true", default=False, help="Enables the overlap of "
                        "Tensor parallel communication and GEMM kernels in Decoder only. "
                        "Please provide decoder-seq-length when using this feature.")
    group.add_argument(
        "--use-packed-sequence",
        action="store_true",
        default=False,
        help="Use packed sequence",
    )
    return parser


def get_processor():
    """Get the Qwen3VLProcessor from transformers."""
    try:
        from transformers import Qwen3VLProcessor
    except ImportError:
        raise ImportError(
            "transformers package is required for Qwen3-VL inference. "
            "Please install it with: pip install transformers"
        )
    return Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


def load_image(image_path: str) -> Image.Image:
    """Load image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path).convert('RGB')


def load_images(image_paths: List[str]) -> List[Image.Image]:
    """Load multiple images from file paths."""
    images = []
    for path in image_paths:
        if path:  # Skip empty paths
            images.append(load_image(path))
    return images


def process_images_and_text_with_processor(
    images: List[Image.Image], 
    texts: List[str], 
    processor
) -> dict:
    """Process images and text using Qwen3VLProcessor.
    
    This properly handles image and text processing following the pattern in
    MockQwen3VLFinetuningDataset.
    
    Args:
        images: List of PIL Image objects (can be empty for text-only)
        texts: List of text prompts
        processor: Qwen3VLProcessor instance
    
    Returns:
        dict: Processed outputs with pixel_values, image_grid_thw, input_ids, attention_mask, etc.
    """
    # Process through the Qwen3VLProcessor
    processed = processor(
        images=images if images else None,
        text=texts,
        return_tensors="pt",
        padding=True,
    )
    
    return processed


def get_inference_engine(args: Namespace, model, hf_tokenizer):
    """Create and configure the Qwen3VL inference engine.
    
    Args:
        args: Model arguments
        model: The Qwen3VL model
        hf_tokenizer: HuggingFace tokenizer from the processor
    """
    # Configure inference wrapper for Qwen3VL
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_seq_length=args.inference_max_seq_length,
    )
    
    # Use Qwen3VL inference wrapper
    inference_wrapped_model = Qwen3VLInferenceWrapper(model, inference_wrapper_config)
    
    # Create Qwen3VL text generation controller with HF tokenizer
    text_generation_controller = Qwen3VLTextGenerationController(
        inference_wrapped_model=inference_wrapped_model,
        tokenizer=hf_tokenizer
    )
    
    # Return the inference engine
    return MCoreEngine(text_generation_controller=text_generation_controller)


def create_prompt_with_image_token(prompt: str) -> str:
    """Create a text prompt with image tokens for multimodal input.
    
    Follows the Qwen3-VL format with vision tags and chat format.
    This matches the format used in MockQwen3VLFinetuningDataset.
    """
    # Format: <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>
    return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>"


def is_first_rank():
    """Check if current rank is first tensor and pipeline parallel rank."""
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def main():
    """Main inference function for Qwen3-VL with image support."""
    
    # Initialize Megatron with inference defaults
    initialize_megatron(
        extra_args_provider=add_inference_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'inference_batch_times_seqlen_threshold': 40,
        },
    )
    
    args = get_args()

    print_rank_0(f"Loading Qwen3-VL model...")
    
    # Load the model
    def wrapped_model_provider(pre_process, post_process, add_encoder, add_decoder):
        return model_provider(pre_process, post_process, add_encoder, add_decoder, parallel_output=False)
    
    model = get_model(wrapped_model_provider, model_type=ModelType.encoder_and_decoder, wrap_with_ddp=False)
    
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    
    model = model[0]
    model.eval()
    
    print_rank_0(f"Model loaded successfully")
    
    # Initialize the Qwen3VLProcessor
    print_rank_0("Loading Qwen3VLProcessor...")
    processor = get_processor()
    
    # Get the HF tokenizer from the processor (not the Megatron tokenizer)
    hf_tokenizer = processor.tokenizer
    print_rank_0(f"Using HuggingFace tokenizer: {type(hf_tokenizer).__name__}")
    
    # Set up inference engine with HF tokenizer
    print_rank_0(f"Setting up inference engine...")
    inference_engine = get_inference_engine(args, model, hf_tokenizer)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )
    
    # Prepare inputs
    prompts = args.prompts
    image_paths = args.image_paths if args.image_paths else []
    
    # Load images if provided
    images_to_process = []
    has_images = len(image_paths) > 0
    
    if has_images:
        print_rank_0(f"Loading {len(image_paths)} image(s)...")
        if len(image_paths) != len(prompts):
            print_rank_0(f"Warning: Number of images ({len(image_paths)}) != number of prompts ({len(prompts)})")
        
        for img_path in image_paths:
            try:
                img = load_image(img_path)
                images_to_process.append(img)
            except Exception as e:
                print_rank_0(f"Error loading image {img_path}: {e}")
                images_to_process.append(None)
    
    print_rank_0(f"\nStarting inference...")
    print_rank_0(f"Prompts: {prompts}")
    if has_images:
        print_rank_0(f"Images: {image_paths}")
    print_rank_0(f"Generating {args.num_tokens_to_generate} tokens per prompt\n")
    
    # Run inference
    start_time = time.perf_counter()
    results = []
    
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            if is_first_rank():
                # Prepare images and text for this prompt
                if idx < len(images_to_process) and images_to_process[idx] is not None:
                    # Multimodal: image + text
                    prompt_with_image = create_prompt_with_image_token(prompt)
                    processed = process_images_and_text_with_processor(
                        images=[images_to_process[idx]],
                        texts=[prompt_with_image],
                        processor=processor
                    )
                    # Move to GPU
                    for key in processed:
                        if isinstance(processed[key], torch.Tensor):
                            processed[key] = processed[key].to("cuda")
                    
                    # Get the processed outputs
                    imgs = processed.get("pixel_values")
                else:
                    # Text-only inference
                    processed = process_images_and_text_with_processor(
                        images=[],
                        texts=[prompt],
                        processor=processor
                    )
                    for key in processed:
                        if isinstance(processed[key], torch.Tensor):
                            processed[key] = processed[key].to("cuda")
                    
                    imgs = None
                
                # Create Qwen3VL inference request with properly processed inputs
                request = Qwen3VLInferenceRequest(
                    request_id=inference_engine.get_new_request_id(),
                    prompt=prompt,
                    prompt_tokens=processed.get("input_ids").flatten().tolist(),
                    inference_parameters=sampling_params,
                    imgs=imgs,
                )
                
                # Generate
                batch_results = inference_engine.generate(
                    inference_requests=[request]
                )
                results.extend(batch_results)
            else:
                # Non-first rank still needs to participate in distributed inference
                if idx < len(images_to_process) and images_to_process[idx] is not None:
                    prompt_with_image = create_prompt_with_image_token(prompts[idx])
                    processed = process_images_and_text_with_processor(
                        images=[images_to_process[idx]],
                        texts=[prompt_with_image],
                        processor=processor
                    )
                else:
                    processed = process_images_and_text_with_processor(
                        images=[],
                        texts=[prompts[idx]],
                        processor=processor
                    )
                
                for key in processed:
                    if isinstance(processed[key], torch.Tensor):
                        processed[key] = processed[key].to("cuda")
                
                inference_engine.generate(inference_requests=[])
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    # Print results (only from first rank)
    if is_first_rank():
        print_rank_0("\n" + "="*70)
        print_rank_0("INFERENCE RESULTS")
        print_rank_0("="*70)
        
        for idx, result in enumerate(results):
            print_rank_0(f"\n--- RESULT {idx} ---")
            print_rank_0(f"Input prompt: {prompts[idx]}")
            if idx < len(image_paths) and image_paths[idx]:
                print_rank_0(f"Input image: {image_paths[idx]}")
            # filter out special tokens from generated text
            cleaned_text = [t for t in result.generated_text if not t.startswith("<|")]
            print_rank_0(f"Generated text:\n{cleaned_text}")
            print_rank_0(f"Generated tokens: {len(result.generated_tokens)}")
        
        print_rank_0(f"\n--- SUMMARY ---")
        print_rank_0(f"Total latency: {latency:.2f} seconds")
        total_tokens = sum(len(r.generated_tokens) for r in results)
        if latency > 0:
            print_rank_0(f"Tokens per second: {total_tokens / latency:.2f}")
    
    # Cleanup
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
