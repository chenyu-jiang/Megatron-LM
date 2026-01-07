# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import OrderedDict, List, Tuple, Optional

import torch

from transformers import Qwen2TokenizerFast

from megatron.core.inference.inference_request import InferenceRequest, VLMInferenceRequest, Qwen3VLInferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)


class VLMTextGenerationController(TextGenerationController):
    """The text generation controller for VLMs"""

    def prep_inference_input(
        self, prompts_tokens: torch.Tensor, active_requests: OrderedDict[str, InferenceRequest]
    ):
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Currently only supports batch size 1 inference.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests
        """
        assert len(active_requests) == 1, f"VLM inference currently only supports batch size 1"

        request = list(active_requests.values())[0]

        assert isinstance(
            request, VLMInferenceRequest
        ), f"Found inference request of type {type(request)}, expected VLMInferenceRequest"

        return self.inference_wrapped_model.prep_inference_input(
            prompts_tokens,
            request.num_img_embeddings_per_tile,
            request.imgs,
            request.num_tiles,
            request.decoder_seq_length,
        )

class Qwen3VLTextGenerationController(TextGenerationController):
    """The text generation controller for VLMs"""

    def prep_inference_input(
        self, prompts_tokens: torch.Tensor, active_requests: OrderedDict[str, InferenceRequest]
    ):
        """Preparing input data for inference, using respective wrapper's prep_inference_input method # pylint: disable=line-too-long

        Currently only supports batch size 1 inference.

        Args:
            prompts_tokens (torch.Tensor): A tensor of shape [batch_size, max_sequence_length]
            active_requests (OrderedDict[str, InferenceRequest]): The input active requests
        """
        assert len(active_requests) == 1, f"VLM inference currently only supports batch size 1"

        request = list(active_requests.values())[0]

        assert isinstance(
            request, Qwen3VLInferenceRequest
        ), f"Found inference request of type {type(request)}, expected Qwen3VLInferenceRequest"

        return self.inference_wrapped_model.prep_inference_input(
            prompts_tokens,
            request.imgs,
            request.image_grid_thw,
        )

    def pad_input_prompt_tokens(
        self,
        batch_prompt_tokens_list: List[List[int]],
        max_prompt_length_in_batch: int,
        num_tokens_to_generate: int,
    ) -> torch.Tensor:
        """Method to pad input prompts

        Given a list of prompts, pad them all to uniform length

        Args:
            batch_prompt_tokens_list (List[List[int]]): A list containing the prompt tokens
            max_prompt_length_in_batch (int): Maximum of the length of the input prompt tokens
            num_tokens_togenerate (int): The number of tokens to generate for each prompt

        Returns:
            torch.Tensor: A torch tensor of shape [bs, max_seq_len] (i.e)
            max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate,
        """
        max_seq_len = max_prompt_length_in_batch + num_tokens_to_generate

        self.tokenizer: Qwen2TokenizerFast  # type: ignore

        for prompt_tokens in batch_prompt_tokens_list:
            padding_size = max_seq_len - len(prompt_tokens)
            prompt_tokens.extend([self.tokenizer.pad_token_id] * padding_size)

        return torch.tensor(batch_prompt_tokens_list, device=torch.cuda.current_device())

    def update_generation_status(
        self,
        updated_prompts_tokens: torch.Tensor,
        generation_started: torch.Tensor,
        current_context_end_position: int,
        is_generation_done_tensor: torch.Tensor,
        generated_sequence_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Checks which prompts have reached an end condition

        We check which prompts have reached an end condition and set the corresponding
        flags of the is_generation_done_tensor to True. The generated sequence lengths
        increase as we keep generating, until that prompts hits an end condition. The
        generation_started tensor determines which prompts have started generating.

        Args:
            updated_prompts_tokens (torch.Tensor): The prompts tokens updated with the latest
                generated tokens. A tensor of shape [batch_size, max_seq_len]
                (i.e max_seq_len = max_prompt_len + tokens_to_generate)
            generation_started (torch.Tensor): A boolean tensor of shape [batch_size]. True
                indicates the prompt at that index has started generating tokens.
            current_context_end_position (int): An integer indicating which position to
                extract from the prompts tokens to get the latest generated tokens.
            is_generation_done_tensor (torch.Tensor): A boolean tensor of shape [batch_size].
                True indicates the prompt at that index has reached end condition.
            generated_sequence_lengths (torch.Tensor): A int tensor of shape [batch_size].
                Each value represents the generated sequence lengths for that prompt.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the boolean
                is_generation_done_tensor and the generated_sequence_lengths after updating it
        """
        latest_samples = updated_prompts_tokens[:, current_context_end_position]
        # Make sure we are checking eod criterion only for prompts that have started generating
        # (i.e) We only look at the generated tokenns and not the input tokens.
        reached_eod = (latest_samples == self.tokenizer.eos_token_id) & generation_started
        is_generation_done_tensor = is_generation_done_tensor | reached_eod
        # We increment generated sequence lengths when that prompt has not hit the
        # EOD and generation has started
        generated_sequence_lengths += ~is_generation_done_tensor & generation_started

        return is_generation_done_tensor, generated_sequence_lengths.int()

    def detokenize_generations(
        self,
        tokens_gpu_tensor: torch.Tensor,
        lengths_gpu_tensor: torch.Tensor,
        detokenize_segments: bool,
    ) -> tuple[str, Optional[List[List[str]]]]:
        """Detokenize the generated tokens.

        Args:
            tokens_gpu_tensor (torch.Tensor): Tensor containing the tokens
            lengths_gpu_tensor (torch.Tensor): Tensor containing the lengths of each sequence
            detokenize_segments (bool): If True, returns individually detokenized tokens. If False,
            returns None as second element. Helpful for understanding per-token boundaries in
            generated text.

        Returns:
            tuple[str, List[str] | None]: A tuple containing:
            - str: The complete detokenized text
            - List[str] | None: List of segmented tokens if detokenize_segments is True, else None
        """
        # TODO(helenn): Unify with `detokenize_generations` from legacy textgen path

        if not detokenize_segments:
            tokens = tokens_gpu_tensor.cpu().numpy().tolist()
            return self.tokenizer.batch_decode(tokens), None

        prompts_plus_generations: List[str] = []
        prompts_plus_generations_segments: List[List[str]] = []

        tokens_gpu_tensor = torch.unsqueeze(tokens_gpu_tensor, 0)
        tokens = tokens_gpu_tensor.cpu().numpy().tolist()
        lengths = lengths_gpu_tensor.cpu().numpy().tolist()

        for sequence_tokens, length in zip(tokens, lengths):
            sequence_tokens = sequence_tokens[:length]
            detok_str = self.tokenizer.detokenize(sequence_tokens)
            prompts_plus_generations.append(detok_str)
            offsets = self.tokenizer.offsets(sequence_tokens, detok_str)
            words = [
                detok_str[start:end] for start, end in zip(offsets, offsets[1:] + [len(detok_str)])
            ]

            prompts_plus_generations_segments.append(words)

        text = self.tokenizer.decode(tokens[0])

        return text, prompts_plus_generations_segments