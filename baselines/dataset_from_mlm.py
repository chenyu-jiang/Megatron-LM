from dataclasses import dataclass
from typing import Optional, Any

from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.global_vars import _build_tokenizer, get_tokenizer

@dataclass
class TokenizerArgs:
    vocab_file: str
    tensor_model_parallel_size: int = 1
    tokenizer_type: str = "BertWordPieceLowerCase"
    vocab_extra_ids: int = 100
    rank: int = 0
    make_vocab_size_divisible_by: int = 128
    merge_file: Optional[str] = None
    tokenizer_model: Optional[str] = None
    padded_vocab_size: Optional[int] = None

def get_train_ds(data_path, vocab_file,
                 targets_data_path, train_epochs,
                 train_iters, global_batch_size,
                 encoder_seq_length,
                 decoder_seq_length=None,
                 dynamic_batching=True,
                 eval_interval=1000,
                 mask_prob=0.15,
                 short_seq_prob=0.1,
                 seed=1234,
                 eval_iters=5, train_samples=None, split="949,50,1", data_impl="mmap",
                 mmap_warmup=False,
                 print_fn=None,
                 sort_dataset=False,
                 pack_dataset=False):
    # Build tokenizer
    _build_tokenizer(TokenizerArgs(vocab_file))
    # Number of train/valid/test samples.
    if train_samples is None:
        assert train_iters is not None, \
            "Either train_samples or train_iters must be specified."
        train_samples = train_iters * global_batch_size
    eval_iters = (train_iters // eval_interval + 1) * eval_iters
    test_iters = eval_iters
    train_val_test_num_samples = [train_samples,
                                    eval_iters * global_batch_size,
                                    test_iters * global_batch_size]
    if print_fn is not None:
        print_fn(' > datasets target sizes (minimum size):')
        print_fn('    train:      {}'.format(train_val_test_num_samples[0]))
        print_fn('    validation: {}'.format(train_val_test_num_samples[1]))
        print_fn('    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    train_ds, _, _ = build_train_valid_test_datasets(
        data_prefix=[data_path],
        data_impl=data_impl,
        splits_string=split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=encoder_seq_length,
        max_seq_length_dec=decoder_seq_length,
        masked_lm_prob=mask_prob,
        short_seq_prob=short_seq_prob,
        seed=seed,
        skip_warmup=(not mmap_warmup),
        dataset_type='t5_supervised',
        num_epochs=train_epochs,
        sort_samples=sort_dataset,
        pack_samples=pack_dataset,
        targets_data_path=targets_data_path,
        dynamic_batch_size=dynamic_batching,
        offline_build=True)
    return train_ds

@dataclass
class DataCollatorForPackedDataset:
    """
    Data collator that checks the sequence length and concats the samples into a batch.

    Args:
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    expected_length: int
    expected_length_target: int
    pad_token_id: int
    model: Optional[Any] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    warned: bool = False

    def __call__(self, features, return_tensors=None):
        import torch
        if return_tensors is None:
            return_tensors = self.return_tensors
        for feature in features:
            # rename "text_enc" to "input_ids" and "text_dec" to "labels"
            feature["input_ids"] = feature.pop("text_enc")
            feature["labels"] = feature.pop("text_dec")
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # pad tokens
        for feature in features:
            feature["input_ids"] = feature["input_ids"] + [self.pad_token_id] * (self.expected_length - len(feature["input_ids"]))
            feature["labels"] = feature["labels"] + [self.label_pad_token_id] * (self.expected_length_target - len(feature["labels"]))
        # inject attention mask if not already there
            if "attention_mask" not in feature:
                feature["attention_mask"] = [1] * len(feature["input_ids"])
        collated_features = {}
        for key in feature.keys():
            if key not in ["input_ids", "attention_mask", "labels"]:
                raise ValueError(f"Found unexpected key {key} in features.")
            collated_features[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=collated_features["labels"])
            collated_features["decoder_input_ids"] = decoder_input_ids

        if not self.warned:
            for key, tensor in collated_features.items():
                print("Key: {}, shape: {}".format(key, tensor.shape))
            self.warned = True
        return collated_features