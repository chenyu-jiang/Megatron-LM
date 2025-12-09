# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional

from dataclasses import dataclass

import torch

from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.qwen3_vit import Qwen3ViTModel, Qwen3VisionConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_rank, get_context_parallel_world_size
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import log_single_rank

try:
    import transformer_engine  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.utils import is_te_min_version

    HAVE_TE = True
    try:
        import transformer_engine_torch as tex

        HAVE_TEX = True
    except:
        HAVE_TEX = False
except:
    HAVE_TE = False
    if get_context_parallel_world_size() > 1:
        raise RuntimeError("ContextParallelism requires TransformerEngine support, but not found.")


class _get_data_on_this_cp_rank(torch.autograd.Function):
    """Performs sharding for Context Parallelism in THD format

    In the forward pass, indices are selected for each CP rank and remaining tokens are dropped.
    In the backward pass, this class takes care of managing gradients for dropped tokens on each
    CP rank.
    """

    @staticmethod
    def forward(ctx, batch, packed_seq_params):
        """Context Parallelism forward support for THD format"""
        cp_size = get_context_parallel_world_size()
        cp_rank = get_context_parallel_rank()
        for key, data in batch.items():
            index = tex.thd_get_partitioned_indices(
                packed_seq_params.cu_seqlens_q_padded, data.size(1), cp_size, cp_rank
            )
            if key == "combined_embeddings":
                ctx.decoder_emb_index = index
                ctx.decoder_emb_seqlen = data.size(1)
            batch[key] = data.index_select(1, index)
            batch[key].requires_grad = data.requires_grad

        return batch

    @staticmethod
    def backward(ctx, grad_out, grad_label, grad_loss):
        """Context Parallelism backward support for THD format"""
        seqlen = ctx.decoder_emb_seqlen
        index = ctx.decoder_emb_index
        assert grad_out.size(1) == index.size(
            0
        ), f"Shape mismatch in incoming gradient {grad_out.shape} and \
                index from THD CP sharding {index.shape}"
        grad_in = torch.zeros(
            grad_out.size(0),
            seqlen,
            *grad_out.size()[2:],
            dtype=grad_out.dtype,
            device=grad_out.device,
        )
        grad_in[:, ctx.decoder_emb_index, :] = grad_out

        return (grad_in, None, None, None)

@dataclass
class Qwen3TextModelConfig:
    language_position_embedding_type: str = "mrope"
    language_vocab_size: int = 151936
    language_max_sequence_length: int = 262144
    language_rotary_percent: float = 1.0
    language_rotary_base: int = 10000
    language_rope_scaling: Optional[dict] = None
    image_token_id: int = 151655
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

# Note: This is under development and may be missing features.
class Qwen3VLModel(MegatronModule):
    """Qwen3-VL model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Language model spec.
        qwen3_language_config (Qwen3TextModelConfig): Qwen3 language model config.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Vision model spec.
        qwen3_vision_config (Qwen3VisionConfig): Qwen3 vision model config.
        parallel_output (bool): Keep outputs split across tensor parallel ranks.
            This is typically True for training and False for inference.
        pre_process (bool): Include embedding layer in the decoder (used with pipeline parallel).
        post_process (bool): Include output layer in the decoder (used with pipeline parallel).
        add_encoder (bool): Construct the encoder (used with pipeline parallel).
            When we use pipelining, the encoder will live on only the first stage
        add_decoder (bool): Construct the decoder (used with pipeline parallel).
            When we use pipelining, the decoder will live on every stage after the first one.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        qwen3_language_config: Qwen3TextModelConfig,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        qwen3_vision_config: Qwen3VisionConfig,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "Qwen3-VL is work in progress. Features are missing and methods can change.",
        )

        self.qwen3_language_config = qwen3_language_config
        self.qwen3_vision_config = qwen3_vision_config

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        if self.sequence_parallel_lm or self.context_parallel_lm > 1:
            assert (
                language_transformer_layer_spec.submodules.self_attention.submodules.core_attention
                == TEDotProductAttention
                and HAVE_TE
            ), "Sequence/Context Parallelism is supported only with TE DotProductAttention."
            if self.context_parallel_lm > 1:
                assert is_te_min_version(
                    "1.10.0"
                ), "Context Parallelism in LLaVA requires TE v1.10 or higher"
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        if self.add_decoder:
            if hasattr(
                language_transformer_config, "language_model_type"
            ) and language_transformer_config.language_model_type.startswith("huggingface"):
                from megatron.core.models.huggingface.module import build_hf_model

                self.language_model = build_hf_model(language_transformer_config)
            else:
                self.language_model = GPTModel(
                    config=language_transformer_config,
                    transformer_layer_spec=language_transformer_layer_spec,
                    vocab_size=qwen3_language_config.language_vocab_size,
                    max_sequence_length=qwen3_language_config.language_max_sequence_length,
                    parallel_output=parallel_output,
                    position_embedding_type=qwen3_language_config.language_position_embedding_type,
                    rotary_percent=qwen3_language_config.language_rotary_percent,
                    pre_process=self.pre_process,
                    post_process=self.post_process,
                    rotary_base=qwen3_language_config.language_rotary_base,
                    rope_scaling=qwen3_language_config.language_rope_scaling,
                    scatter_embedding_sequence_parallel=False,
                )

                self.share_embeddings_and_output_weights = (
                    self.language_model.share_embeddings_and_output_weights
                )
            # self._language_max_sequence_length = language_transformer_config.max_sequence_length
            self._language_is_pipeline_parallel = (
                language_transformer_config.pipeline_model_parallel_size > 1
            )

            # Newer Transformer Engine versions add _extra_state keys in state_dict when using FP8.
            # Older models may not have _extra_state and can be ignored.
            self.language_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

        if self.add_encoder:
            assert vision_transformer_config.vision_model_type == "qwen3_vit", (
                "Qwen3-VL vision model type must be 'qwen3_vit' for Qwen3-VL models."
            )
            self.vision_model = Qwen3ViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                qwen3_vision_config,
            )
            self.vision_model.register_load_state_dict_post_hook(
                _load_state_dict_hook_ignore_extra_state
            )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for qwen3-vl'

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _process_embedding_token_parallel(
        self, combined_embeddings, new_labels, new_loss_mask, packed_seq_params
    ):
        """Processes the input data for model parallelism support.

        When using sequence parallelism (SP) or context parallelism (CP), the sequence is sharded
        across different GPUs. This function performs the sharding and distributes the sequence
        across GPUs for SP and CP

        Context Parallelism is a feature that helps improve memory efficiency for
        long sequence training by distributing sequence across CP ranks.
        It requires token length to be divisible by (CP size *2) to ensure proper load balance.

        Sequence Parallelism is a feature that helps improve memory efficiency for
        long sequence training by distributing sequence across TP ranks.
        It requires token length to be divisible by TP size.

        Returns:
            combined_embeddings (torch.Tensor): image and text embeddings combined and distributed.
            new_labels (torch.Tensor): Distributed labels for image and text positions.
            new_loss_mask (torch.Tensor): Distributed loss mask.
            packed_seq_params (PackedSeqParams): Dict with padded token information.

        """

        # No pre or post processing needed with PP middle chunks.
        if not self.pre_process and not self.post_process:
            return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

        shard_factor = seq_dim = None
        if self.pre_process:
            if self.context_parallel_lm > 1 and self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm * self.context_parallel_lm * 2
                seq_dim = 1
            elif self.context_parallel_lm > 1:
                shard_factor = self.context_parallel_lm * 2
                seq_dim = 1
            elif self.sequence_parallel_lm:
                shard_factor = self.tensor_model_parallel_size_lm
                seq_dim = 0

            assert (
                combined_embeddings.shape[seq_dim] % shard_factor == 0
            ), f"Sequence length should be divisible by {shard_factor} for \
                Sequence/Context parallelism"
            if self.sequence_parallel_lm and self.tp_comm_overlap_lm:
                assert (
                    combined_embeddings.shape[seq_dim] == self._language_max_sequence_length
                ), f"TP Comm overlap either requires Vision+Text token length \
                == language_max_sequence_length"

        if self.context_parallel_lm > 1:
            batch = dict()
            if self.pre_process:
                batch["combined_embeddings"] = combined_embeddings
            if self.post_process:
                batch["new_labels"] = new_labels
                batch["new_loss_mask"] = new_loss_mask
            # Distribute sequence across CP ranks
            if packed_seq_params is None or packed_seq_params.qkv_format == 'sbhd':
                from megatron.training.utils import get_batch_on_this_cp_rank

                batch = get_batch_on_this_cp_rank(batch)
            else:
                assert HAVE_TEX and is_te_min_version(
                    "1.10.0"
                ), "Please update Transformer Engine to >= 1.10 to use \
                    Context Parallel with THD format data"
                batch = _get_data_on_this_cp_rank.apply(batch, packed_seq_params)

            if self.pre_process:
                combined_embeddings = batch["combined_embeddings"]  # [B, S/CP, H]
                combined_embeddings = combined_embeddings.transpose(
                    1, 0
                ).contiguous()  # [B,S/CP,H] -> [S/CP,B,H]
            if self.post_process:
                new_labels = batch["new_labels"]
                new_loss_mask = batch["new_loss_mask"]

        if self.sequence_parallel_lm and self.pre_process:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings
            )  # [S/(CP*TP),B,H]

        return combined_embeddings, new_labels, new_loss_mask, packed_seq_params

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        # pixel_values = pixel_values.type(self.vision_model.dtype)
        image_embeds, deepstack_image_embeds = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.language_model.embedding(
                torch.tensor(self.qwen3_language_config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.qwen3_language_config.image_token_id

        # special_image_mask is [b, seqlen], switch to [seqlen, b] to match inputs_embeds
        special_image_mask = special_image_mask.transpose(0, 1)

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        return special_image_mask

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen3-VL model.

        Args:
            images (torch.Tensor): input images of shape [batch_size, num_channels, img_h, img_w].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Language model attention mask
                [batch, 1, 1, combined_seq_len]. NOTE: attention_mask is typically None and
                attn_mask_type in layer specs determines the attention mask used.
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
            packed_seq_params (PackedSeqParams): 1) If using sequence packing, must contain
                subsample length information. 2) If using SP/CP with padding mask type,
                must contain padded token information.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        has_images = images is not None and images.shape[0] > 0

        # If running inference, we can skip image token computation
        # if they were computed already earlier for this sample.
        deepstack_image_embeds = None
        if use_inference_kv_cache:
            input_embeddings = None
        elif self.add_encoder and not has_images:
            # If no images provided, use an empty image embeddings tensor.
            input_embeddings = self.language_model.embedding(input_ids, None)
            # image_embeddings = torch.tensor([], dtype=images.dtype, device=images.device).reshape(
            #     0, 0, 0
            # )
        elif self.add_encoder and has_images:
            input_embeddings = self.language_model.embedding(input_ids, None)
            image_embeddings, deepstack_image_embeds = self.get_image_features(
                images, image_grid_thw=image_grid_thw
            )
            image_embeddings = torch.cat(image_embeddings, dim=0).to(images.device, images.dtype)
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=input_embeddings, image_features=image_embeddings
            )
            input_embeddings = input_embeddings.masked_scatter(image_mask, image_embeddings)

            image_mask = image_mask[..., 0]

            # TODO: Support batched inference.
            # In inference, the language model KV cache will be updated for image token positions.
            # Store the image tokens sequence length to be used as an offset to the KV cache later.
            if inference_params is not None:
                inference_params.key_value_memory_dict["image_tokens_count"] = (
                    image_embeddings.shape[0] * image_embeddings.shape[1]
                )
        else:
            input_embeddings = self.encoder_hidden_state

        if not self.add_decoder:
            return (input_embeddings, deepstack_image_embeds), loss_mask

        # if self.context_parallel_lm > 1 or self.sequence_parallel_lm:
        #     combined_embeddings, new_labels, new_loss_mask, packed_seq_params = (
        #         self._process_embedding_token_parallel(
        #             combined_embeddings, new_labels, new_loss_mask, packed_seq_params
        #         )
        #     )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=input_embeddings,
            labels=labels,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs={
                "visual_pos_masks": image_mask,
                "deepstack_visual_embeds": deepstack_image_embeds,
            },
        )

        return output, loss_mask


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this if you want to load a checkpoint that contains vision and language
    model weights but not the vision projection weights.

    Args:
        param_names (list str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in LlavaModel"
            )
            incompatible_keys.missing_keys.remove(param_name)


def _load_state_dict_hook_ignore_extra_state(
    module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    This is for backwards-compatibility. Newer TE versions add _extra_state keys to the state dict,
    while older models might not have those keys. Those keys can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys,
            which collect the missing and unexpected keys, respectively.
    """
    for name, keys in incompatible_keys._asdict().items():
        for key in keys[::-1]:
            if "extra_state" in key:
                logging.getLogger(__name__).warning(
                    f"_extra_state key {key} being removed from {name}"
                )
                keys.remove(key)


# pylint: disable-next=line-too-long
# Based on https://github.com/OpenGVLab/InternVL/blob/c7c5af1a8930b4862afe8ed14672307082ef61fa/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L218
# Copyright (c) 2023 OpenGVLab.
def pixel_shuffle(x, scale_factor=0.5, version=2):
    """Pixel shuffle based on InternVL but adapted for our use case.

    Args:
        x (torch.Tensor): Vision model outputs [num_tiles, img_seq_len, h_vision]
        version (int): Implementation version.

    Returns:
        Shuffled vision model outputs [num_tiles, (sq ** 2) * (scale ** 2), h_vision / (scale ** 2)]
    """
    h = w = int(x.shape[1] ** 0.5)  # sq
    x = x.reshape(x.shape[0], h, w, -1)  # [num_tiles, sq, sq, h_vision]

    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(
        n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
    )

    if version == 2:
        x = x.permute(0, 2, 1, 3).contiguous()

    x = x.reshape(x.shape[0], -1, x.shape[-1])

    return x
