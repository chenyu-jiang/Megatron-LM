# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataloaders."""

import sys
import os
import random
import bisect
import pickle
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron import get_args
from megatron import mpu
from megatron import get_num_microbatches
from megatron.utils import print_rank_0
from megatron.data.t5_dataset import T5SupervisedDataset, T5UnsupervisedDataset
from megatron.data.data_utils import DynamicBatchingLevel

from plopt.data_opt.optimizer import DataAssignmentOptimizer

MB_WORKER_PATH = (Path(__file__).parent / "gen_microbatch_worker.py").resolve()

def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
        shape_iterator = None
        n_iters_per_epoch = len(dataset) // args.global_batch_size
    elif args.dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )
        shape_iterator = None
        n_iters_per_epoch = len(dataset) // args.global_batch_size
    elif args.dataloader_type == "ordered":
        batch_sampler = MegatronPretrainingOrderedSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
            dynamic_batchsize=args.dynamic_batchsize,
            dynamic_batch_level=args.dynamic_batch_level,
            tokens_per_global_batch=args.tokens_per_global_batch,
            # seq_len_buckets=args.seq_len_buckets,
            # max_truncation_factor=args.max_truncation_factor,
            # min_truncation_seq_len=args.min_truncation_seq_len,
        )
        shape_iterator = batch_sampler.get_shape_iterator()
        n_iters_per_epoch = batch_sampler.n_iters_per_epoch()
    else:
        raise Exception(
            "{} dataloader type is not supported.".format(args.dataloader_type)
        )

    # Torch dataloader.
    if isinstance(dataset, T5SupervisedDataset) and not dataset.padded:
        collate_fn = dataset.dynamic_microbatch_collate_fn
    else:
        collate_fn = None
    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        shape_iterator,
        n_iters_per_epoch,
    )


class MegatronPretrainingSampler:
    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = (
            self.micro_batch_size * data_parallel_size
        )
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(
            self.total_samples
        )
        assert (
            self.consumed_samples < self.total_samples
        ), "no samples left to consume: {}, {}".format(
            self.consumed_samples, self.total_samples
        )
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), "data_parallel_rank should be smaller than data size: {}, " "{}".format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):
    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:
    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = (
            self.micro_batch_size * data_parallel_size
        )
        self.last_batch_size = (
            self.total_samples % self.micro_batch_times_data_parallel_size
        )

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(
            self.total_samples
        )
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), "data_parallel_rank should be smaller than data size: {}, " "{}".format(
            self.data_parallel_rank, data_parallel_size
        )
        if isinstance(dataset, (T5UnsupervisedDataset, T5SupervisedDataset)):
            self.dataset.set_per_sample_seq_len_func(
                lambda _: (self.dataset.max_seq_length, self.dataset.max_seq_length_dec, False)
            )

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (
                self.total_samples // self.micro_batch_size
            ) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[
                self.data_parallel_rank :: self.data_parallel_size
            ]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class ShapeIterator:
    def __init__(self, shape_generator):
        self.shape_generator = shape_generator

    def __iter__(self):
        return self.shape_generator()


class MegatronPretrainingOrderedSampler(MegatronPretrainingRandomSampler):
    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
        dynamic_batchsize=False,
        dynamic_batch_level=DynamicBatchingLevel.BATCH,
        tokens_per_global_batch=None,
    ):
        super().__init__(
            dataset,
            total_samples,
            consumed_samples,
            micro_batch_size,
            data_parallel_rank,
            data_parallel_size,
            data_sharding,
        )
        if not isinstance(dataset, (T5UnsupervisedDataset, T5SupervisedDataset)):
            raise NotImplementedError(
                "Only T5Dataset is supported for ordered sampler for now."
            )
        assert (
            hasattr(dataset, "ordered") and dataset.ordered
        ), "Dataset should be ordered for ordered sampler."
        self._global_batch_size_per_rank = (
            get_num_microbatches() * self.micro_batch_size
        )
        self._global_batch_size = (
            self._global_batch_size_per_rank * self.data_parallel_size
        )
        self._dynamic_batchsize = dynamic_batchsize
        self._dynamic_batch_level = dynamic_batch_level
        if dynamic_batchsize:
            assert (
                tokens_per_global_batch is not None
            ), "tokens_per_global_batch should be provided for dynamic batching"
            self._tokens_per_global_batch = tokens_per_global_batch
            if self._dynamic_batch_level == DynamicBatchingLevel.MICROBATCH:
                args = get_args()
                self._target_compute_efficiency = args.dynamic_batch_min_efficiency
                # hard code model spec for now
                self._stage_time_model = DataAssignmentOptimizer(args.dynamic_batch_profile_path, 4, 3, 36000, 1024, 32, 16384, 128)
        self._shape_consumed_samples = consumed_samples
        self._is_supervised_dataset = isinstance(dataset, T5SupervisedDataset)
        if dynamic_batchsize:
            self._precalc_batches()
        else:
            self.dataset.set_per_sample_seq_len_func(
                lambda _: (self.dataset.max_seq_length, self.dataset.max_seq_length_dec, False)
            )

    def _round_mbs(self, target_mbs):
        mbs_bucket_idx = (
            bisect.bisect_right(self._microbatch_size_buckets, target_mbs) - 1
        )
        if mbs_bucket_idx < 1:
            mbs_bucket_idx = 1
        rounded_mbs = self._microbatch_size_buckets[mbs_bucket_idx]
        return rounded_mbs

    def _get_seq_len_bucket_idx(self, seq_len, decoder=False):
        buckets = self._seq_len_buckets if not decoder else self._dec_seq_len_buckets
        bucket_idx = bisect.bisect_left(buckets, seq_len)
        if bucket_idx == len(buckets):
            bucket_idx -= 1
        if not decoder:
            # see if seq_len is within truncation factor of the previous bucket
            if bucket_idx > 1 and seq_len > self._min_truncation_seq_len:
                prev_bucket_seq_len = buckets[bucket_idx - 1]
                if seq_len <= prev_bucket_seq_len * (1 + self._max_truncation_factor):
                    bucket_idx -= 1
        return bucket_idx

    def _n_microbatches_per_global_batch(self, mbs):
        return self._global_batch_size_per_rank // mbs

    def n_iters_per_epoch(self):
        if self._dynamic_batchsize:
            return len(self._batches)
        else:
            return len(self.dataset) // self._global_batch_size

    def _get_seq_len(self, sample_id):
        args = get_args()
        if args.assume_perfect_batching:
            input_seq_len = args.perfect_batching_seq_len
            target_seq_len = args.perfect_batching_seq_len
        else:
            input_seq_len = min(
                self.dataset.get_seq_len(sample_id), self.dataset.max_seq_length
            )
            target_seq_len = min(
                self.dataset.get_dec_seq_len(sample_id),
                self.dataset.max_seq_length_dec,
            )
        return input_seq_len, target_seq_len

    def _calc_batched_seqlen_for_mb_level_dbs(
            self, sample_input_seqlens, sample_target_seqlens, indices
        ):
            padded_input_seqlens = []
            padded_target_seqlens = []
            for mb_indices in indices:
                max_input_seqlen = -1
                max_target_seqlen = -1
                for samples in mb_indices:
                    input_seqlen = sum([sample_input_seqlens[s] for s in samples])
                    target_seqlen = sum([sample_target_seqlens[s] for s in samples])
                    max_input_seqlen = max(max_input_seqlen, input_seqlen)
                    max_target_seqlen = max(max_target_seqlen, target_seqlen)
                max_target_seqlen += 2  # for bos and eos
                # round up to nearest multiple of 8
                max_input_seqlen = (max_input_seqlen + 7) // 8 * 8
                max_target_seqlen = (max_target_seqlen + 7) // 8 * 8
                padded_input_seqlens.append(max_input_seqlen)
                padded_target_seqlens.append(max_target_seqlen)
            return padded_input_seqlens, padded_target_seqlens

    @staticmethod
    def _append_batch_async_launch(args):
        dynamic_batch_profile_path, target_compute_eff, sample_input_seqlens, sample_target_seqlens, start_idx, curr_idx, batch_idx = args
        # hardcode model spec for now
        dataopt = DataAssignmentOptimizer(dynamic_batch_profile_path, 4, 3, 36000, 1024, 32, 16384, 128)

        _, indices = dataopt.generate_microbatches(
            sample_input_seqlens,
            decoder_sample_sequence_lengths=sample_target_seqlens,
            bottleneck_tsp=False,
            min_compute_efficiency=target_compute_eff,
            satisfy_seqlen_percentile=0.75,
            enable_packing=True,
            uniform_cost=False,
            partition_method="dp",
        )
        return indices, start_idx, curr_idx, batch_idx

    def _precalc_batches(self):
        assert self._dynamic_batchsize
        args = get_args()
        assert (
            args.memory_model == "fixed"
        ), "Only fixed micro-batch size is supported for dynamic batching for now."

        self._batches = []
        current_batch_start_idx = 0
        adjusted_total_samples = 0
        n_global_batches = 0
        num_micro_batches_per_global_batch = []
        # needed for microbatch level dynamic batching
        avg_samples_per_microbatch = []
        # needed for batch level dynamic batching
        consumed_tokens_include_padding = 0
        current_batch_input_padded_seq_len = 0
        current_batch_target_padded_seq_len = 0
        padded_input_sequence_lengths = []
        # needed for assume perfect batching
        current_batch_size = 0

        # needed for per sample seq len func for dataset when using mini-batch
        # level dynamic batching
        global_batch_start_boundaries = []
        per_global_batch_enc_dec_seq_lengths = []
        # needed when using micro-batch level dynamic batching
        per_sample_enc_dec_seq_lengths = [None] * self.total_samples

        if args.assume_perfect_batching:
            total_tokens = 0
            for idx in range(self.total_samples):
                input_seq_len = min(
                    self.dataset.get_seq_len(idx), self.dataset.max_seq_length
                )
                total_tokens += input_seq_len

        def _append_batch(start_idx, curr_idx):
            nonlocal n_global_batches
            nonlocal num_micro_batches_per_global_batch
            nonlocal current_batch_input_padded_seq_len
            nonlocal current_batch_target_padded_seq_len
            nonlocal consumed_tokens_include_padding
            nonlocal per_sample_enc_dec_seq_lengths
            nonlocal avg_samples_per_microbatch
            if self._dynamic_batch_level == DynamicBatchingLevel.MICROBATCH:
                # if dynamic batch level is microbatch, each microbatch is
                # a list of list, outer: sequences, inner: samples packed
                # into that sequence
                sample_input_seqlens, sample_target_seqlens = zip(
                    *[self._get_seq_len(idx) for idx in range(start_idx, curr_idx)]
                )
                sample_input_seqlens = list(sample_input_seqlens)
                sample_target_seqlens = list(sample_target_seqlens)
                _, indices = self._stage_time_model.generate_microbatches(
                    sample_input_seqlens,
                    decoder_sample_sequence_lengths=sample_target_seqlens,
                    bottleneck_tsp=False,
                    min_compute_efficiency=self._target_compute_efficiency,
                )
                (
                    padded_input_seqlens,
                    padded_target_seqlens,
                ) = self._calc_batched_seqlen_for_mb_level_dbs(
                    sample_input_seqlens, sample_target_seqlens, indices
                )
                microbatches = []
                for i, mb_indices in enumerate(indices):
                    mb = []
                    num_samples = 0
                    for samples in mb_indices:
                        sequence = []
                        for sample_idx, s in enumerate(samples):
                            per_sample_enc_dec_seq_lengths[start_idx + s] = (
                                padded_input_seqlens[i],
                                padded_target_seqlens[i],
                                sample_idx != 0
                            )
                            sequence.append(start_idx + s)
                            num_samples += 1
                        mb.append(sequence)
                    microbatches.append(mb)
                    avg_samples_per_microbatch.append(num_samples)
                num_micro_batches_per_global_batch.append(len(microbatches))
            else:
                # if not using dynamic microbatching, each microbatch is
                # just a tuple specifying the start and end sample index.
                # use fixed microbatch size specified in the args without packing
                n_microbatches = (curr_idx - start_idx) // self.micro_batch_size
                n_microbatches_per_rank = n_microbatches // self.data_parallel_size
                num_micro_batches_per_global_batch.append(n_microbatches)
                microbatches = []
                for m in range(n_microbatches_per_rank):
                    mb_start_idx = (
                        start_idx
                        + self.data_parallel_rank * n_microbatches_per_rank
                        + m * self.micro_batch_size
                    )
                    mb_end_idx = mb_start_idx + self.micro_batch_size
                    microbatches.append((mb_start_idx, mb_end_idx))
            if args.benchmark_microbatch_execution_time:
                # we want to measure the time it takes to execute a single microbatch
                microbatches = microbatches[:1]
                num_micro_batches_per_global_batch[-1] = 1
            self._batches.append(microbatches)
            n_global_batches += 1

            if self._dynamic_batch_level == DynamicBatchingLevel.BATCH:
                global_batch_start_boundaries.append(start_idx)
                # round seq len to nearest multiple of 8
                current_batch_input_padded_seq_len = (
                    (current_batch_input_padded_seq_len + 7) // 8 * 8
                )
                current_batch_target_padded_seq_len = (
                    (current_batch_target_padded_seq_len + 7) // 8 * 8
                )
                per_global_batch_enc_dec_seq_lengths.append(
                    (
                        current_batch_input_padded_seq_len,
                        current_batch_target_padded_seq_len,
                    )
                )
                padded_input_sequence_lengths.append(current_batch_input_padded_seq_len)
                consumed_tokens_include_padding += (
                    current_batch_size * current_batch_input_padded_seq_len
                )

        def _append_batch_async_finish(indices, start_idx, curr_idx, batch_idx):
            nonlocal n_global_batches
            nonlocal avg_samples_per_microbatch
            sample_input_seqlens, sample_target_seqlens = zip(
                *[self._get_seq_len(idx) for idx in range(start_idx, curr_idx)]
            )
            sample_input_seqlens = list(sample_input_seqlens)
            sample_target_seqlens = list(sample_target_seqlens)
            (
                padded_input_seqlens,
                padded_target_seqlens,
            ) = self._calc_batched_seqlen_for_mb_level_dbs(
            sample_input_seqlens, sample_target_seqlens, indices
            )
            microbatches = []
            for i, mb_indices in enumerate(indices):
                mb = []
                num_samples = 0
                for samples in mb_indices:
                    sequence = []
                    for sample_idx, s in enumerate(samples):
                        per_sample_enc_dec_seq_lengths[start_idx + s] = (
                            padded_input_seqlens[i],
                            padded_target_seqlens[i],
                            sample_idx != 0
                        )
                        sequence.append(start_idx + s)
                        num_samples += 1
                    mb.append(sequence)
                microbatches.append(mb)
                avg_samples_per_microbatch.append(num_samples)
            num_micro_batches_per_global_batch.append(len(microbatches))
            if args.benchmark_microbatch_execution_time:
                # we want to measure the time it takes to execute a single microbatch
                microbatches = microbatches[:1]
                num_micro_batches_per_global_batch[-1] = 1
            self._batches[batch_idx] = microbatches
            n_global_batches += 1

        multiprocess_args = []
        avg_global_batch_tokens = 0
        avg_input_seq_len = 0
        current_batch_tokens = 0
        if mpu.get_pipeline_model_parallel_rank() == 0:
            # only calculate batches on the first rank
            print_rank_0("Calculating microbatch assignment using {} workers...".format(args.preprocess_workers))
            from tqdm import trange
            for idx in trange(self.total_samples, desc="Separating batches"):
                input_seq_len, target_seq_len = self._get_seq_len(idx)
                avg_input_seq_len += input_seq_len
                if args.assume_perfect_batching:
                    current_batch_tokens = (
                        current_batch_size * current_batch_input_padded_seq_len
                    )
                # create a new batch if the current batch contains a multiple
                # of data_parallel_size * micro_batch_size samples, and
                # contains roughly _tokens_per_global_batch tokens
                if args.assume_perfect_batching:
                    seq_len_if_added = max(
                        current_batch_input_padded_seq_len, input_seq_len
                    )
                    tokens_if_added = (current_batch_size + 1) * seq_len_if_added
                else:
                    tokens_if_added = current_batch_tokens + input_seq_len
                if (
                    # current batch is not empty
                    current_batch_tokens
                    # current batch contains a multiple of data_parallel_size * micro_batch_size samples
                    and (
                        (idx - current_batch_start_idx)
                        % (self.data_parallel_size * self.micro_batch_size)
                        == 0
                    )
                    # contains roughly _tokens_per_global_batch tokens
                    and tokens_if_added >= self._tokens_per_global_batch
                ):
                    # create a new batch
                    # process_info = _append_batch_async_launch(current_batch_start_idx, idx)
                    if self._dynamic_batch_level == DynamicBatchingLevel.MICROBATCH:
                        # if dynamic batch level is microbatch, each microbatch is
                        # a list of list, outer: sequences, inner: samples packed
                        # into that sequence
                        sample_input_seqlens, sample_target_seqlens = zip(
                            *[self._get_seq_len(i) for i in range(current_batch_start_idx, idx)]
                        )
                        sample_input_seqlens = list(sample_input_seqlens)
                        sample_target_seqlens = list(sample_target_seqlens)
                        multiprocess_arg = (args.dynamic_batch_profile_path, self._target_compute_efficiency, 
                                            sample_input_seqlens, sample_target_seqlens, current_batch_start_idx, idx, len(multiprocess_args))
                        multiprocess_args.append(multiprocess_arg)
                    else:
                        _append_batch(current_batch_start_idx, idx)
                    adjusted_total_samples += idx - current_batch_start_idx
                    avg_global_batch_tokens += current_batch_tokens
                    current_batch_start_idx = idx
                    current_batch_size = 0
                    current_batch_tokens = 0
                    current_batch_input_padded_seq_len = 0
                    current_batch_target_padded_seq_len = 0
                    if (
                        args.assume_perfect_batching
                        and consumed_tokens_include_padding >= total_tokens
                    ):
                        self.total_samples = adjusted_total_samples
                        break
                # append to current batch
                current_batch_tokens += input_seq_len
                if self._dynamic_batch_level == DynamicBatchingLevel.BATCH:
                    current_batch_size += 1
                    current_batch_input_padded_seq_len = max(
                        current_batch_input_padded_seq_len, input_seq_len
                    )
                    current_batch_target_padded_seq_len = max(
                        current_batch_target_padded_seq_len, target_seq_len
                    )
            if self._dynamic_batch_level == DynamicBatchingLevel.MICROBATCH:
                self._batches = [None for _ in range(len(multiprocess_args))]
                # launch the worker processes
                import multiprocessing
                from tqdm import tqdm
                with multiprocessing.Pool(args.preprocess_workers) as pool:
                    map_result = pool.imap_unordered(self._append_batch_async_launch, multiprocess_args)
                    for indices, start_idx, curr_idx, batch_idx in tqdm(map_result, total=len(multiprocess_args), desc="Calculating microbatch assignment"):
                        _append_batch_async_finish(indices, start_idx, curr_idx, batch_idx)
            # create the last batch
            if (
                not args.assume_perfect_batching
                and (idx - current_batch_start_idx)
                % (self.data_parallel_size * self.micro_batch_size)
                == 0
            ):
                _append_batch(current_batch_start_idx, idx)
                avg_global_batch_tokens += current_batch_tokens
            # pickle the data
            data_to_broadcast = (self._batches,
                                adjusted_total_samples,
                                n_global_batches,
                                per_sample_enc_dec_seq_lengths)
            # for testing, dump to current directory
            with open("./tmp_batch_data.pickle", "wb") as f:
                pickle.dump(data_to_broadcast, f)
            # barrier
            torch.distributed.barrier()
            torch.distributed.barrier()
            # delete the file
            os.remove("./tmp_batch_data.pickle")
        else:
            torch.distributed.barrier()
            with open("./tmp_batch_data.pickle", "rb") as f:
                (self._batches,
                adjusted_total_samples,
                n_global_batches,
                per_sample_enc_dec_seq_lengths) = pickle.load(f)
            torch.distributed.barrier()

        self.last_batch_size = self.total_samples - adjusted_total_samples
        if args.benchmark_microbatch_execution_time:
            # only execute first benchmark_microbatch_iters iterations
            self._batches = self._batches[: args.benchmark_microbatch_iters]
            n_global_batches = len(self._batches)
        else:
            print_rank_0(
                "INFO: adjusted total samples from {} to {}. Data wasted: {} ({:.2f}%).".format(
                    self.total_samples,
                    adjusted_total_samples,
                    self.last_batch_size,
                    self.last_batch_size / self.total_samples * 100,
                )
            )
        print_rank_0(">> Dynamic batch stats:")
        print_rank_0(
            ">> Target tokens per global batch: {}, Avg. tokens per global batch: {}, avg input seq len: {}".format(
                self._tokens_per_global_batch,
                avg_global_batch_tokens / n_global_batches,
                avg_input_seq_len / adjusted_total_samples,
            )
        )
        n_samples_per_mb = 0
        if mpu.get_pipeline_model_parallel_rank() == 0:
            if self._dynamic_batch_level == DynamicBatchingLevel.MICROBATCH:
                n_samples_per_mb = sum(avg_samples_per_microbatch) / len(
                    avg_samples_per_microbatch
                )
            else:
                n_samples_per_mb = self.micro_batch_size
            print_rank_0(
                ">> {} total global batches, avg. {:.2f} microbatches per global batch, {} samples per microbatch.".format(
                    n_global_batches,
                    sum(num_micro_batches_per_global_batch)
                    / len(num_micro_batches_per_global_batch),
                    n_samples_per_mb,
                )
            )
            if self._dynamic_batch_level == DynamicBatchingLevel.BATCH:
                for print_seq_len in [16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096]:
                    n_batches = 0
                    for x in padded_input_sequence_lengths:
                        if x >= print_seq_len:
                            n_batches += 1
                    if n_batches > 0:
                        print_rank_0(
                            ">> {} batches have sequence length >= {}.".format(
                                n_batches, print_seq_len
                            )
                        )
        if self._dynamic_batch_level == DynamicBatchingLevel.MICROBATCH:
            # sanity check on per_sample_enc_dec_seq_lengths
            for idx in range(adjusted_total_samples):
                if per_sample_enc_dec_seq_lengths[idx] is None:
                    raise ValueError("Invalid per_sample_enc_dec_seq_lengths at idx {}".format(idx))

        def per_sample_seq_len_func(idx):
            if self._dynamic_batch_level == DynamicBatchingLevel.BATCH:
                global_batch_id = (
                    bisect.bisect_right(global_batch_start_boundaries, idx) - 1
                )
                enc_seqlen, dec_seqlen = per_global_batch_enc_dec_seq_lengths[global_batch_id]
                return enc_seqlen, dec_seqlen, False
            else:
                return per_sample_enc_dec_seq_lengths[idx]

        self.dataset.set_per_sample_seq_len_func(per_sample_seq_len_func)
        self.dataset.set_adjusted_num_samples(adjusted_total_samples)
        self._per_sample_seq_len_func = per_sample_seq_len_func

    def _calc_sample_offsets(self, is_data_iterator=True):
        active_total_samples = self.total_samples - self.last_batch_size
        if is_data_iterator:
            self.epoch = self.consumed_samples // active_total_samples
            if isinstance(self.dataset, RandomSeedDataset):
                self.dataset.set_epoch(self.epoch)
            current_epoch_samples = self.consumed_samples % active_total_samples
            epoch = self.epoch
        else:
            current_epoch_samples = self._shape_consumed_samples % active_total_samples
            epoch = self._shape_consumed_samples // active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if not self.data_sharding:
            print_rank_0(
                "WARNING: data sharding must be enabled for sorted sampler. Ignoring setting."
            )
        return current_epoch_samples, epoch

    def _dynamic_size_get_batch_order(self, is_data_iterator=True):
        current_epoch_samples, epoch = self._calc_sample_offsets(
            is_data_iterator=is_data_iterator
        )
        g = torch.Generator()
        g.manual_seed(epoch)
        n_global_batches = len(self._batches)
        batch_order = torch.randperm(n_global_batches, generator=g).tolist()
        cumulative_samples = 0
        for batch_offset in range(len(batch_order)):
            microbatches = self._batches[batch_offset]
            if cumulative_samples < current_epoch_samples:
                for microbatch in microbatches:
                    if isinstance(microbatch, tuple):
                        cumulative_samples += (
                            microbatch[1] - microbatch[0]
                        ) * self.data_parallel_size
                    elif isinstance(microbatch, list):
                        # dynamic microbatching
                        for sample in microbatch:
                            cumulative_samples += len(sample) * self.data_parallel_size
                    else:
                        raise TypeError(
                            "microbatch must be a tuple or list, got {}".format(
                                type(microbatch)
                            )
                        )
            else:
                break
        return batch_order, batch_offset

    def get_shape_iterator(self):
        if not self._dynamic_batchsize:
            return None

        def shape_generator():
            # generates (num_microbatches, mbs, enc_seq_len, dec_seq_len) tuples
            # need to call get_batch_order inside the generator to ensure
            # that shape iterator is in sync with the data iterator upon reset
            batch_order, batch_offset = self._dynamic_size_get_batch_order(
                is_data_iterator=False
            )
            for batch_id in batch_order[batch_offset:]:
                microbatches = self._batches[batch_id]
                for mb in microbatches:
                    if isinstance(mb, tuple):
                        batch_size = mb[1] - mb[0]
                        enc_seq_len, dec_seq_len, _ = self._per_sample_seq_len_func(mb[0])
                        self._shape_consumed_samples += (
                            batch_size * self.data_parallel_size
                        )
                    elif isinstance(mb, list):
                        # dynamic microbatching
                        batch_size = len(mb)
                        enc_seq_len, dec_seq_len, _ = self._per_sample_seq_len_func(
                            mb[0][0]
                        )
                        for sample in mb:
                            self._shape_consumed_samples += (
                                len(sample) * self.data_parallel_size
                            )
                    else:
                        raise TypeError(
                            "microbatch must be a tuple or list, got {}".format(
                                type(mb)
                            )
                        )
                    yield len(microbatches), batch_size, enc_seq_len, dec_seq_len

        return ShapeIterator(shape_generator)

    def __iter__(self):
        if self._dynamic_batchsize:
            batch_order, batch_offset = self._dynamic_size_get_batch_order()
            for batch_id in batch_order[batch_offset:]:
                microbatches = self._batches[batch_id]
                for mb in microbatches:
                    if isinstance(mb, tuple):
                        sample_indices = [idx for idx in range(mb[0], mb[1])]
                        self.consumed_samples += (
                            len(sample_indices) * self.data_parallel_size
                        )
                        yield sample_indices
                    elif isinstance(mb, list):
                        # dynamic microbatching. use -1 to delimit samples packed into
                        # the same sequence
                        sample_indices = []
                        actual_samples_consumed = 0
                        for sample in mb:
                            sample_indices.extend(sample)
                            actual_samples_consumed += len(sample)
                            sample_indices.append(-1)
                        self.consumed_samples += (
                            actual_samples_consumed * self.data_parallel_size
                        )
                        yield sample_indices
                    else:
                        raise TypeError(
                            "microbatch must be a tuple or list, got {}".format(
                                type(mb)
                            )
                        )
        else:
            current_epoch_samples, epoch = self._calc_sample_offsets()
            g = torch.Generator()
            g.manual_seed(epoch)
            # since we are using sorted dataset, data access is strided for each rank
            stride_size_per_batch = self.micro_batch_times_data_parallel_size
            n_batches = self.total_samples // stride_size_per_batch
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_offset = self.data_parallel_rank

            per_batch_start_idx = torch.randperm(n_batches, generator=g).tolist()
            batch_idx_range = [
                start_offset + x * stride_size_per_batch
                for x in per_batch_start_idx[bucket_offset:]
            ]

            batch = []
            # Last batch if not complete will be dropped.
            for idx in batch_idx_range:
                for i in range(self.micro_batch_size):
                    batch.append(idx + i * self.data_parallel_size)
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
