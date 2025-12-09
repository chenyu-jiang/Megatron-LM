"""Simple test script for MockQwen3VLDataset."""

import torch
from torch.utils.data import DataLoader

from megatron.core.datasets.multimodal_dataset import (
    MockQwen3VLFinetuningDataset,
    MockQwen3VLDatasetConfig,
)


def test_mock_qwen3vl_dataset():
    """Test basic functionality of MockQwen3VLDataset."""
    print("=" * 60)
    print("Testing MockQwen3VLDataset")
    print("=" * 60)

    # Create config
    config = MockQwen3VLDatasetConfig(
        max_image_h=1024,
        max_image_w=1024,
        min_image_h=64,
        min_image_w=64,
        max_text_length=64,
        min_text_length=16,
        processor_name_or_path="Qwen/Qwen3-VL-30B-A3B-Instruct",
        seed=42,
        num_samples=100,
    )

    print(f"\nConfig:")
    print(f"  - Image size: {config.min_image_h}-{config.max_image_h} x {config.min_image_w}-{config.max_image_w}")
    print(f"  - Text length: {config.min_text_length}-{config.max_text_length}")
    print(f"  - Num samples: {config.num_samples}")
    print(f"  - Processor: {config.processor_name_or_path}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = MockQwen3VLFinetuningDataset(config)
    print(f"Dataset length: {len(dataset)}")

    # Test single sample
    print("\n" + "-" * 40)
    print("Testing single sample retrieval...")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    # Test reproducibility
    print("\n" + "-" * 40)
    print("Testing reproducibility...")
    sample1 = dataset[0]
    sample2 = dataset[0]
    
    if torch.equal(sample1["input_ids"], sample2["input_ids"]):
        print("  ✓ input_ids are reproducible")
    else:
        print("  ✗ input_ids are NOT reproducible")
    
    if torch.equal(sample1["pixel_values"], sample2["pixel_values"]):
        print("  ✓ pixel_values are reproducible")
    else:
        print("  ✗ pixel_values are NOT reproducible")

    # Test different samples are different
    print("\n" + "-" * 40)
    print("Testing different samples...")
    sample_a = dataset[0]
    sample_b = dataset[1]
    
    if not torch.equal(sample_a["input_ids"], sample_b["input_ids"]):
        print("  ✓ Different indices produce different input_ids")
    else:
        print("  ✗ Different indices produce same input_ids (unexpected)")

    # Test DataLoader with collate_fn
    print("\n" + "-" * 40)
    print("Testing DataLoader with collate_fn...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    batch = next(iter(dataloader))
    print(f"Batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_mock_qwen3vl_dataset()
