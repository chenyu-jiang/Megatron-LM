# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain vision language model."""
from copy import deepcopy
from functools import partial
import warnings

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.multimodal_dataset import (
    MockMultimodalDataset,
    MultimodalDatasetConfig,
    MockQwen3VLFinetuningDataset,
    MockQwen3VLDatasetConfig,
)
from megatron.core.enums import ModelType
from megatron.core.models.vision.qwen3_vit import get_num_image_embeddings
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.multimodal.qwen3_vl_model import Qwen3VLModel, Qwen3VisionConfig, Qwen3TextModelConfig
from megatron.core.models.multimodal.qwen3_vl_spec import (
    decoder_model_with_transformer_engine_default_spec,
    decoder_model_with_local_default_spec,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_qwen3_vit_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_batch_on_this_cp_rank
from megatron.core import mpu
from megatron.core.models.multimodal import context_parallel
from pretrain_gpt import loss_func


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> Qwen3VLModel:
    """Builds the model.

    Note: currently, only LLaVA model is supported. Follow-up changes will make this configurable.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable model parallel output.

    Returns:
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): A multimodal model
    """
    args = get_args()
    vision_model_type = "qwen3_vit"

    # assert args.ckpt_format == 'torch', "Only ckpt-format torch is supported for VLM training currently."
    # assert not (args.context_parallel_size > 1 and args.pipeline_model_parallel_size > 1), "PP+CP is not yet supported by this script. \
    # Current mock dataset does not support natively packed sequence dataset required for correct PP comm shapes."

    # old_seq_length = args.seq_length
    # # dataloader-seq-length is required to determine the length of text seq len
    # if args.dataloader_seq_length is None:
    #     args.dataloader_seq_length = args.seq_length

    # # decoder_seq_len denotes the language model sequence length.
    # decoder_seq_len = args.dataloader_seq_length + num_image_embeddings

    # # seq_length and encoder_seq_length denote the vision model sequence length. Override if the user provided something else.
    # args.seq_length = args.encoder_seq_length = num_image_embeddings
    # if torch.distributed.get_rank() == 0 and old_seq_length != args.seq_length:
    #     warnings.warn(
    #         f"Changed seq_length and encoder_seq_length (vision model sequence length) from {old_seq_length} to num_image_tokens ({num_image_embeddings})"
    #     )
    # mp_padding_needed = context_parallel.get_padding(
    #     decoder_seq_len,
    #     args.context_parallel_size,
    #     args.tensor_model_parallel_size,
    #     args.sequence_parallel,
    #     args.decoder_tp_comm_overlap,
    #     args.decoder_seq_length
    # )
    # args.decoder_seq_length = decoder_seq_len + mp_padding_needed
    mp_padding_needed = True

    args.max_position_embeddings = max(args.max_position_embeddings, args.decoder_seq_length)

    print_rank_0('building a multimodal model ...')
    language_transformer_config = core_transformer_config_from_args(get_args())
    if args.decoder_tp_comm_overlap:
        assert args.transformer_impl == "transformer_engine", \
            "TransformerEngine is needed to support Decoder TP Comm overlap"
        language_transformer_config.tp_comm_overlap = args.decoder_tp_comm_overlap

    if args.spec is not None:
        language_transformer_layer_spec = import_module(args.spec)
    elif args.transformer_impl == "transformer_engine":
        language_transformer_layer_spec = decoder_model_with_transformer_engine_default_spec(
            args.num_experts, args.moe_grouped_gemm, qk_layernorm=args.qk_layernorm
        )
    else:  # transformer_impl == "local"
        language_transformer_layer_spec = decoder_model_with_local_default_spec(
            args.num_experts, args.moe_grouped_gemm, qk_layernorm=args.qk_layernorm
        )

    # Prepare mask type for any required padding to support CP/SP sequence sharding.
    # if mp_padding_needed > 0:
    #     if language_transformer_layer_spec.submodules.self_attention.params.get('attn_mask_type', '') == AttnMaskType.causal:
    #         language_transformer_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding_causal
    #     elif language_transformer_layer_spec.submodules.self_attention.params.get('attn_mask_type', '') == AttnMaskType.no_mask:
    #         language_transformer_layer_spec.submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding

    if args.transformer_impl == "transformer_engine":
        vision_transformer_layer_spec = get_qwen3_vit_layer_with_transformer_engine_spec()
    else:  # transformer_impl == "local"
        raise NotImplementedError("Only Transformer Engine implementation is supported for Qwen3-VL currently.")
        # vision_transformer_layer_spec = get_vit_layer_with_local_spec()

    # TODO: Make these configurable via input .yaml config.
    qwen3_vision_config = Qwen3VisionConfig()
    vision_transformer_config = deepcopy(language_transformer_config)
    # vit uses bias
    vision_transformer_config.add_bias_linear = True
    vision_transformer_config.add_qkv_bias = True
    # vit uses layernorm (not rmsnorm)
    vision_transformer_config.normalization = "LayerNorm"
    # make sure this is aligned with the vision model config in Qwen3VisionConfig
    vision_transformer_config.hidden_size = qwen3_vision_config.hidden_size
    vision_transformer_config.activation_func = partial(F.gelu, approximate='tanh')
    vision_transformer_config.ffn_hidden_size = qwen3_vision_config.ffn_hidden_size
    vision_transformer_config.kv_channels = qwen3_vision_config.hidden_size // qwen3_vision_config.num_heads
    vision_transformer_config.num_attention_heads = qwen3_vision_config.num_heads
    # ViT uses full MHA
    vision_transformer_config.num_query_groups = vision_transformer_config.num_attention_heads
    vision_transformer_config.bias_activation_fusion = False
    vision_transformer_config.num_layers = args.encoder_num_layers
    vision_transformer_config.first_pipeline_num_layers = None
    vision_transformer_config.last_pipeline_num_layers = None
    vision_transformer_config.vision_model_type = vision_model_type
    vision_transformer_config.context_parallel_size = 1 # Force CP=1 for Vision Transformer
    if vision_transformer_config.sequence_parallel:
        print_rank_0("> Disabling Sequence parallelism in Vision Transformer. Not yet supported")
        vision_transformer_config.sequence_parallel = False
    if vision_transformer_config.tp_comm_overlap:
        print_rank_0("> Disabling TP Comm overlap in Vision Transformer. Not yet supported")
        vision_transformer_config.tp_comm_overlap = False

    if args.encoder_pipeline_model_parallel_size > 0:
        assert (
            args.encoder_pipeline_model_parallel_size == 1
        ), "ViT can only live on 1 pipeline stage."
        vision_transformer_config.pipeline_model_parallel_size = (
            args.encoder_pipeline_model_parallel_size
        )
        if args.encoder_tensor_model_parallel_size > 0:
            vision_transformer_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )

    if args.virtual_pipeline_model_parallel_size:
        raise NotImplementedError("virtual pipeline model parallelism is not supported yet.")

    language_max_sequence_length = args.decoder_seq_length
    if args.context_parallel_size > 1:
        if args.use_packed_sequence or mp_padding_needed > 0:
            # Use THD data format
            language_max_sequence_length = args.decoder_seq_length * args.micro_batch_size
    # with torch.device("cpu"):
    model = Qwen3VLModel(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        qwen3_language_config=Qwen3TextModelConfig(),
        vision_transformer_config=vision_transformer_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        qwen3_vision_config=Qwen3VisionConfig(),
        parallel_output=parallel_output,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False,
    )

    # load hf model for reference
    # from transformers import AutoModelForImageTextToText
    # hf_model = AutoModelForImageTextToText.from_pretrained(
    #     "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="cpu"
    # )

    # import code
    # code.interact(local=locals())

    # with open("./hf_model_keys.json", "w") as f:
    #     import json
    #     state_dict_keys = list(hf_model.state_dict().keys())
    #     # also get weight shape
    #     for key in state_dict_keys:
    #         state_dict_keys[state_dict_keys.index(key)] = (key, str(hf_model.state_dict()[key].shape))
    #     json.dump(state_dict_keys, f, indent=4)
    
    # with open("./megatron_model_keys.json", "w") as f:
    #     import json
    #     state_dict_keys = list(model.state_dict().keys())
    #     # also get weight shape
    #     for key in state_dict_keys:
    #         if hasattr(model.state_dict()[key], 'shape'):
    #             state_dict_keys[state_dict_keys.index(key)] = (key, str(model.state_dict()[key].shape))
    #         else:
    #             state_dict_keys[state_dict_keys.index(key)] = (key, "N/A")
    #     json.dump(state_dict_keys, f, indent=4)

    # exit(0)
    return model

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, validation, and test datasets using MockQwen3VLDataset.

    This provider creates mock Qwen3-VL datasets with random images and text,
    processed using Qwen3VLProcessor. Useful for testing and development.

    Args:
        train_val_test_num_samples: A list containing the number of samples in train, validation, and test sets.

    Returns:
        train_ds, val_ds, test_ds: Train, validation, and test MockQwen3VLDataset instances.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for Qwen3-VL using mock data ...")

    # Create configurations for train, validation, and test datasets
    train_config = MockQwen3VLDatasetConfig(
        max_image_h=getattr(args, 'img_h', 1024),
        max_image_w=getattr(args, 'img_w', 1024),
        min_image_h=64,
        min_image_w=64,
        max_text_length=getattr(args, 'dataloader_seq_length', 256) if hasattr(args, 'dataloader_seq_length') and args.dataloader_seq_length else 256,
        min_text_length=16,
        processor_name_or_path=getattr(args, 'qwen3_processor_path', "Qwen/Qwen3-VL-30B-A3B-Instruct"),
        seed=args.seed,
        num_samples=train_val_test_num_samples[0],
    )

    valid_config = MockQwen3VLDatasetConfig(
        max_image_h=getattr(args, 'img_h', 1024),
        max_image_w=getattr(args, 'img_w', 1024),
        min_image_h=64,
        min_image_w=64,
        max_text_length=getattr(args, 'dataloader_seq_length', 256) if hasattr(args, 'dataloader_seq_length') and args.dataloader_seq_length else 256,
        min_text_length=16,
        processor_name_or_path=getattr(args, 'qwen3_processor_path', "Qwen/Qwen3-VL-30B-A3B-Instruct"),
        seed=args.seed + 1,  # Different seed for validation
        num_samples=train_val_test_num_samples[1],
    )

    test_config = MockQwen3VLDatasetConfig(
        max_image_h=getattr(args, 'img_h', 1024),
        max_image_w=getattr(args, 'img_w', 1024),
        min_image_h=64,
        min_image_w=64,
        max_text_length=getattr(args, 'dataloader_seq_length', 256) if hasattr(args, 'dataloader_seq_length') and args.dataloader_seq_length else 256,
        min_text_length=16,
        processor_name_or_path=getattr(args, 'qwen3_processor_path', "Qwen/Qwen3-VL-30B-A3B-Instruct"),
        seed=args.seed + 2,  # Different seed for test
        num_samples=train_val_test_num_samples[2],
    )

    # Create datasets
    train_ds = MockQwen3VLFinetuningDataset(train_config)
    valid_ds = MockQwen3VLFinetuningDataset(valid_config) if train_val_test_num_samples[1] > 0 else None
    test_ds = MockQwen3VLFinetuningDataset(test_config) if train_val_test_num_samples[2] > 0 else None

    print_rank_0(f">   train dataset size: {len(train_ds)}")
    if valid_ds:
        print_rank_0(f">   validation dataset size: {len(valid_ds)}")
    if test_ds:
        print_rank_0(f">   test dataset size: {len(test_ds)}")
    print_rank_0("> finished creating Qwen3-VL mock multimodal datasets ...")

    return train_ds, valid_ds, test_ds


def get_batch(data_iterator):
    """Generate a batch.

    Args:
        data_iterator: Iterable dataset.

    Returns:
        sample: A data sample with images, tokens, etc.
    """
    args = get_args()
    cp_size = args.context_parallel_size
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i = tensor_parallel.broadcast_data(["input_ids", "labels", "position_ids", "image_grid_thw",], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["pixel_values", "loss_mask"], data, torch.float32)

    batch = dict()
    packed_seq_params = None
    image_token_mask = None
    # Create batch with tokens and position_ids for CP sharding.
    tokens = data_i["input_ids"].long()
    position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    image_grid_thw = data_i["image_grid_thw"].long()
    loss_mask = data_f["loss_mask"].float()
    images = data_f["pixel_values"].float()

    # if cp_size > 1 or args.sequence_parallel:
    #     vision_model_type = "clip"
    #     # Calculate the number of image embedding tokens will be added to text tokens
    #     num_image_embeddings_per_tile = get_num_image_embeddings(
    #         args.img_h, args.img_w, args.patch_dim, vision_model_type, args.disable_vision_class_token, 1
    #     )
    #     # Pad to make sure the text sequence can be sharded equally by CP chunks.
    #     image_token_mask = tokens == DEFAULT_IMAGE_TOKEN_INDEX
    #     num_images_per_sample = torch.sum(image_token_mask, dim=-1)
    #     img_seq_len = (num_image_embeddings_per_tile * num_images_per_sample - num_images_per_sample).max()
    #     mp_padding_needed_for_text = context_parallel.get_padding(
    #         tokens.shape[1] + img_seq_len,
    #         args.context_parallel_size,
    #         args.tensor_model_parallel_size,
    #         args.sequence_parallel,
    #         args.decoder_tp_comm_overlap,
    #         args.decoder_seq_length
    #     )
    #     if mp_padding_needed_for_text > 0:
    #         tokens, position_ids, labels, loss_mask = [torch.nn.functional.pad(item, (0, mp_padding_needed_for_text)) for item in (tokens, position_ids, labels, loss_mask)]
    #     packed_seq_params = context_parallel.get_packed_seq_params(tokens, img_seq_len, mp_padding_needed_for_text, cp_size, args.use_packed_sequence)

    #     if packed_seq_params.qkv_format == 'thd':
    #         # Reshape from [B,S] to [T,1]
    #         tokens = (
    #             tokens.contiguous()
    #             .view(tokens.shape[0] * tokens.shape[1])
    #             .unsqueeze(0)
    #         )
    #         position_ids = (
    #             position_ids.contiguous()
    #             .view(position_ids.shape[0] * position_ids.shape[1])
    #             .unsqueeze(0)
    #         )
    #         labels = labels.view(labels.shape[0] * labels.shape[1]).unsqueeze(0)
    #         loss_mask = loss_mask.view(
    #             loss_mask.shape[0] * loss_mask.shape[1]
    #         ).unsqueeze(0)

    attention_mask = None  # Use the attention mask type defined in layer spec. Typically no mask for the vision model and causal mask for the vision model.

    return tokens, position_ids, labels, images, image_grid_thw, loss_mask, attention_mask, packed_seq_params


def forward_step(data_iterator, model: Qwen3VLModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, position_ids, labels, images, image_grid_thw, loss_mask, attention_mask, packed_seq_params = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(
        images, tokens, position_ids, attention_mask, labels, loss_mask, packed_seq_params=packed_seq_params, image_grid_thw=image_grid_thw
    )

    return output_tensor, partial(loss_func, loss_mask)


def add_vlm_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='vision language model specific arguments')
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


def qwen3vl_embedding_ranks(pp_ranks):
    """Qwen3-VL's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank:
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]


def qwen3vl_position_embedding_ranks(pp_ranks):
    """Qwen3-VL's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_vlm_extra_args,
        get_embedding_ranks=qwen3vl_embedding_ranks,
        get_position_embedding_ranks=qwen3vl_position_embedding_ranks,
    )
