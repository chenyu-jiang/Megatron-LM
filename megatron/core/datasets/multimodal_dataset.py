# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset


@dataclass
class MultimodalDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core Multimodal datasets.

    Note: This is unused at the moment and may be missing features. Follow-up changes will use this.
    """

    image_h: int = None
    """Image height."""

    image_w: int = None
    """Image width."""

    # Function to preprocess the data sample to a format expected by a specific model. By default, do nothing.
    preprocess_func: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = lambda x: x
    """Optional function to preprocess data samples for a specific model."""

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.image_h is not None
        assert self.image_w is not None


class MockMultimodalDataset(MockGPTDataset):
    """Mock multimodal dataset.


    This is unused at the moment and may be missing features. Follow-up changes will use this.
    """

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample that contains a dummy image, text sequence and the associated labels and cost and attention masks.

        Args:
            idx (int): The integer seed for mock data generation.

        Returns:
            Dict[str, torch.Tensor]: The mock data.
        """
        # Get a text sample.
        sample = super().__getitem__(idx)

        # Add mock input image.
        sample["image"] = torch.zeros(
            (3, self.config.image_h, self.config.image_w), dtype=torch.float32
        )

        # Run optional data preprocessing.
        preprocess_func = self.config.preprocess_func

        return preprocess_func(sample)


@dataclass
class MockQwen3VLDatasetConfig:
    """Configuration object for Mock Qwen3-VL multimodal datasets.

    This config is used for generating mock multimodal data with random images and text,
    processed using the Qwen3VLProcessor.

    Note: This config does NOT inherit from GPTDatasetConfig because:
    1. MockQwen3VLDataset is a standalone torch.utils.data.Dataset, not a MegatronDataset
    2. It uses Qwen3VLProcessor from transformers for tokenization, not MegatronTokenizer
    3. It doesn't use GPT-specific fields like reset_position_ids, eod_mask_loss, etc.
    """

    max_image_h: int = 512
    """Maximum image height. Actual height will be randomly sampled up to this value."""

    max_image_w: int = 512
    """Maximum image width. Actual width will be randomly sampled up to this value."""

    min_image_h: int = 64
    """Minimum image height."""

    min_image_w: int = 64
    """Minimum image width."""

    max_text_length: int = 256
    """Maximum text token length. Actual length will be randomly sampled up to this value."""

    min_text_length: int = 16
    """Minimum text token length."""

    processor_name_or_path: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    """The pretrained processor name or path for Qwen3VLProcessor."""

    seed: int = 42
    """Random seed for reproducibility."""

    num_samples: int = 10000
    """Number of samples in the mock dataset."""

    def __post_init__(self) -> None:
        # Don't call super().__post_init__() as we don't need tokenizer validation
        # The processor will handle tokenization
        assert self.max_image_h >= self.min_image_h
        assert self.max_image_w >= self.min_image_w
        assert self.max_text_length >= self.min_text_length


class MockQwen3VLFinetuningDataset(torch.utils.data.Dataset):
    """Mock multimodal dataset using Qwen3VLProcessor.

    This dataset generates random images and companion text, then processes them
    using Qwen3VLProcessor to produce inputs compatible with Qwen3-VL models.

    Each sample contains:
        - A randomly generated RGB image with random resolution (up to max_image_h x max_image_w)
        - Random companion text with random length (up to max_text_length tokens)
        - Processed outputs from Qwen3VLProcessor including input_ids, attention_mask,
          pixel_values, and image_grid_thw

    Args:
        config (MockQwen3VLDatasetConfig): Configuration for the mock dataset.
    """

    # Vocabulary for generating random text
    MOCK_VOCABULARY: List[str] = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below", "between",
        "this", "that", "these", "those", "what", "which", "who", "whom", "whose",
        "image", "picture", "photo", "photograph", "scene", "view", "landscape",
        "object", "item", "thing", "person", "people", "animal", "plant", "building",
        "shows", "displays", "contains", "features", "depicts", "illustrates",
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
        "large", "small", "big", "tiny", "huge", "medium", "tall", "short", "wide", "narrow",
        "beautiful", "amazing", "wonderful", "interesting", "colorful", "bright", "dark",
        "describe", "explain", "analyze", "identify", "recognize", "detect", "find",
    ]

    def __init__(self, config: MockQwen3VLDatasetConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed=config.seed)

        # Lazy load processor to avoid import issues if transformers is not installed
        self._processor = None

        # Pre-generate random parameters for each sample for reproducibility
        self._image_heights = self._rng.integers(
            low=config.min_image_h, high=config.max_image_h + 1, size=config.num_samples
        )
        self._image_widths = self._rng.integers(
            low=config.min_image_w, high=config.max_image_w + 1, size=config.num_samples
        )
        self._text_lengths = self._rng.integers(
            low=config.min_text_length, high=config.max_text_length + 1, size=config.num_samples
        )

    @property
    def processor(self):
        """Lazy load the Qwen3VLProcessor."""
        if self._processor is None:
            try:
                from transformers import Qwen3VLProcessor
            except ImportError:
                raise ImportError(
                    "transformers package is required to use MockQwen3VLDataset. "
                    "Please install it with: pip install transformers"
                )
            self._processor = Qwen3VLProcessor.from_pretrained(
                self.config.processor_name_or_path
            )
        return self._processor

    def __len__(self) -> int:
        return self.config.num_samples

    def _generate_random_image(self, idx: int) -> Image.Image:
        """Generate a random RGB image with random resolution.

        Args:
            idx (int): Index used for seeding to ensure reproducibility.

        Returns:
            Image.Image: A randomly generated PIL Image.
        """
        height = int(self._image_heights[idx])
        width = int(self._image_widths[idx])

        # Create a random image using the sample-specific RNG
        sample_rng = np.random.default_rng(seed=self.config.seed + idx)
        random_pixels = sample_rng.integers(
            low=0, high=256, size=(height, width, 3), dtype=np.uint8
        )
        image = Image.fromarray(random_pixels, mode="RGB")
        return image

    def _generate_random_text(self, idx: int) -> str:
        """Generate random companion text for the image.

        Args:
            idx (int): Index used for seeding to ensure reproducibility.

        Returns:
            str: A randomly generated text string.
        """
        text_length = int(self._text_lengths[idx])

        # Use sample-specific RNG for reproducibility
        sample_rng = np.random.default_rng(seed=self.config.seed + idx + 1000000)
        word_indices = sample_rng.integers(
            low=0, high=len(self.MOCK_VOCABULARY), size=text_length
        )
        words = [self.MOCK_VOCABULARY[i] for i in word_indices]

        # Generate response text with similar length
        response_rng = np.random.default_rng(seed=self.config.seed + idx + 2000000)
        response_word_indices = response_rng.integers(
            low=0, high=len(self.MOCK_VOCABULARY), size=text_length
        )
        response_words = [self.MOCK_VOCABULARY[i] for i in response_word_indices]

        # Create text with image placeholder for Qwen3-VL format, including assistant response
        text = (
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{' '.join(words)}<|im_end|>\n"
            f"<|im_start|>assistant\n{' '.join(response_words)}<|im_end|>"
        )
        return text

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a processed sample containing image and text.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - input_ids: Tokenized input sequence (tokens)
                - labels: Shifted input_ids for next token prediction
                - attention_mask: Attention mask for the sequence
                - loss_mask: Mask indicating which tokens to compute loss on (1.0 for assistant response, 0.0 for user input)
                - position_ids: Position IDs for the sequence
                - pixel_values: Processed image pixel values
                - image_grid_thw: Image grid information (temporal, height, width)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Generate random image and text
        image = self._generate_random_image(idx)
        text = self._generate_random_text(idx)

        # Process using Qwen3VLProcessor
        processed = self.processor(
            images=[image],
            text=[text],
            return_tensors="pt",
            padding=True,
        )

        # also create position_ids if not present
        if "position_ids" not in processed:
            # create from attention_mask
            attention_mask = processed["attention_mask"]
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            processed["position_ids"] = position_ids

        # Remove batch dimension for single sample
        sample = {}
        for key, value in processed.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.squeeze(0)
            else:
                sample[key] = value

        # Generate labels (shifted input_ids for next token prediction)
        input_ids = sample["input_ids"]
        labels = torch.roll(input_ids, shifts=-1, dims=0)
        labels[-1] = self.processor.tokenizer.pad_token_id or 0
        sample["labels"] = labels

        # Generate loss_mask: only compute loss on assistant response tokens
        # Find the position of <|im_start|>assistant token sequence
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        
        # Find where assistant response starts
        # The assistant token in Qwen3-VL is typically "<|im_start|>assistant\n"
        # We need to mask the loss for everything before the assistant's response
        assistant_start_id = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        
        # Find the position where assistant response begins
        assistant_start_pos = -1
        for i in range(len(input_ids) - len(assistant_start_id) + 1):
            match = True
            for j, token_id in enumerate(assistant_start_id):
                if input_ids[i + j] != token_id:
                    match = False
                    break
            if match:
                assistant_start_pos = i + len(assistant_start_id)
                break
        
        # Set loss_mask to 1.0 for assistant response tokens (after assistant_start_pos)
        if assistant_start_pos >= 0:
            loss_mask[assistant_start_pos:] = 1.0
        
        # Mask out padding tokens
        loss_mask[labels == (self.processor.tokenizer.pad_token_id or 0)] = 0.0
        
        # Also mask the end token if present
        # im_end_id = self.processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        # if len(im_end_id) > 0:
        #     loss_mask[labels == im_end_id[-1]] = 0.0

        sample["loss_mask"] = loss_mask

        return sample

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching samples.

        Args:
            batch (List[Dict[str, torch.Tensor]]): List of samples to batch.

        Returns:
            Dict[str, torch.Tensor]: Batched samples with proper padding.
        """
        # Re-process the batch through the processor for proper padding
        images = []
        texts = []

        for i, sample_idx in enumerate(range(len(batch))):
            # Regenerate images and texts for the batch
            # This is needed because we need the raw inputs for proper batched processing
            pass

        # For simplicity, stack tensors with padding
        result = {}

        # Handle input_ids, attention_mask, labels, loss_mask, and position_ids with padding
        if "input_ids" in batch[0]:
            max_len = max(sample["input_ids"].shape[0] for sample in batch)
            pad_token_id = self.processor.tokenizer.pad_token_id or 0

            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            padded_loss_mask = []
            padded_position_ids = []
            
            for sample in batch:
                seq_len = sample["input_ids"].shape[0]
                padding_len = max_len - seq_len
                
                # Pad input_ids
                padded_input_ids.append(
                    torch.cat([
                        sample["input_ids"],
                        torch.full((padding_len,), pad_token_id, dtype=sample["input_ids"].dtype)
                    ])
                )
                
                # Pad attention_mask
                padded_attention_mask.append(
                    torch.cat([
                        sample["attention_mask"],
                        torch.zeros(padding_len, dtype=sample["attention_mask"].dtype)
                    ])
                )
                
                # Pad labels
                if "labels" in sample:
                    padded_labels.append(
                        torch.cat([
                            sample["labels"],
                            torch.full((padding_len,), pad_token_id, dtype=sample["labels"].dtype)
                        ])
                    )
                
                # Pad loss_mask
                if "loss_mask" in sample:
                    padded_loss_mask.append(
                        torch.cat([
                            sample["loss_mask"],
                            torch.zeros(padding_len, dtype=sample["loss_mask"].dtype)
                        ])
                    )
                
                # Pad position_ids
                if "position_ids" in sample:
                    padded_position_ids.append(
                        torch.cat([
                            sample["position_ids"],
                            torch.zeros(padding_len, dtype=sample["position_ids"].dtype)
                        ])
                    )
            
            result["input_ids"] = torch.stack(padded_input_ids)
            result["attention_mask"] = torch.stack(padded_attention_mask)
            
            if padded_labels:
                result["labels"] = torch.stack(padded_labels)
            if padded_loss_mask:
                result["loss_mask"] = torch.stack(padded_loss_mask)
            if padded_position_ids:
                result["position_ids"] = torch.stack(padded_position_ids)

        # Handle pixel_values - concatenate along batch dimension
        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat([sample["pixel_values"] for sample in batch], dim=0)

        # Handle image_grid_thw - stack as batch
        if "image_grid_thw" in batch[0]:
            result["image_grid_thw"] = torch.stack([sample["image_grid_thw"] for sample in batch])

        return result

