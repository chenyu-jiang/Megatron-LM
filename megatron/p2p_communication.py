# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from functools import reduce
import operator
import torch

from megatron import get_args, core
from megatron.core import mpu


def _communicate_shapes(tensor_send_next, tensor_send_prev,
                        recv_prev, recv_next):
    """Communicate tensor shapes between stages. Used to communicate 
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    args = get_args()
    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty((3),
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
    if recv_next:
        recv_next_shape_tensor = torch.empty((3),
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(tensor_send_prev.size(),
                                              device=torch.cuda.current_device(),
                                              dtype=torch.int64)
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(tensor_send_next.size(),
                                              device=torch.cuda.current_device(),
                                              dtype=torch.int64)

    if args.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(tensor_send_prev=send_prev_shape_tensor,
                                        tensor_recv_prev=recv_prev_shape_tensor,
                                        tensor_send_next=send_next_shape_tensor,
                                        tensor_recv_next=recv_next_shape_tensor,
                                        group=mpu.get_pipeline_model_parallel_group())
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, send_prev_shape_tensor,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_prev_shape_tensor,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, send_next_shape_tensor,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_next_shape_tensor,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                 tensor_shape, recv_prev_shape=None, recv_next_shape=None,
                 dtype_=None):
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        tensor_shape: shape of tensor to receive (this method assumes that all
                      tensors sent and received in a single function call are
                      the same shape).
        dtype_: optional, this is used when the tensor that needs to be
                communicated is different from args.params_dtype.
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if recv_prev_shape is None and recv_next_shape is None:
        # Some legacy inference code doesn't set the tensor shape, do so now
        # for the normal values for gpt/bert. This could be removed if inference
        # code is changed to provide tensor_shape.
        if not args.variable_seq_lengths:
            if tensor_shape is None:
                recv_prev_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
                recv_next_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
            else:
                recv_prev_shape = tensor_shape
                recv_next_shape = tensor_shape
        else:
            recv_prev_shape, recv_next_shape = \
                _communicate_shapes(tensor_send_next,
                                    tensor_send_prev,
                                    recv_prev,
                                    recv_next)
    if not isinstance(tensor_send_next, list):
        tensor_send_next = [tensor_send_next]
    if not isinstance(tensor_send_prev, list):
        tensor_send_prev = [tensor_send_prev]

    unwrap_recv_prev = False
    if not isinstance(recv_prev_shape, list):
        unwrap_recv_prev = True
        recv_prev_shape = [recv_prev_shape]
    unwrap_recv_next = False
    if not isinstance(recv_next_shape, list):
        unwrap_recv_next = True
        recv_next_shape = [recv_next_shape]

    override_scatter_gather_tensors_in_pipeline = False
    recv_prev_chunk_shapes = []
    recv_next_chunk_shapes = []
    for idx in range(len(recv_prev_shape)):
        rpshape = recv_prev_shape[idx]
        if args.scatter_gather_tensors_in_pipeline and \
                not args.sequence_parallel:
            recv_prev_chunk_shape = reduce(operator.mul, rpshape, 1)
            if recv_prev_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:
                recv_prev_chunk_shape = recv_prev_chunk_shape // \
                    mpu.get_tensor_model_parallel_world_size()
            else:
                recv_prev_chunk_shape = rpshape
                override_scatter_gather_tensors_in_pipeline = True
        else:
            recv_prev_chunk_shape = rpshape
        recv_prev_chunk_shapes.append(recv_prev_chunk_shape)
    for idx in range(len(recv_next_shape)):
        rnshape = recv_next_shape[idx]
        if args.scatter_gather_tensors_in_pipeline and \
                not args.sequence_parallel:
            recv_next_chunk_shape = reduce(operator.mul, rnshape, 1)
            if recv_next_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:
                recv_next_chunk_shape = recv_next_chunk_shape // \
                    mpu.get_tensor_model_parallel_world_size()
            else:
                recv_next_chunk_shape = rnshape
                override_scatter_gather_tensors_in_pipeline = True
        else:
            recv_next_chunk_shape = rnshape
        recv_next_chunk_shapes.append(recv_next_chunk_shape)
    assert override_scatter_gather_tensors_in_pipeline == False, \
        "Not currently supporting scatter_gather_tensors_in_pipeline and hetero" \
        "number of tensors"

    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float

    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    tensor_recv_prevs = []
    tensor_recv_nexts = []
    if recv_prev:
        for recv_prev_chunk_shape in recv_prev_chunk_shapes:
            tensor_recv_prevs.append(torch.empty(recv_prev_chunk_shape,
                                                 requires_grad=requires_grad,
                                                 device=torch.cuda.current_device(),
                                                 dtype=dtype))
    else:
        tensor_recv_prevs = [None] * len(recv_prev_chunk_shapes)
    if recv_next:
        for recv_next_chunk_shape in recv_next_chunk_shapes:
            tensor_recv_nexts.append(torch.empty(recv_next_chunk_shape,
                                                 requires_grad=requires_grad,
                                                 device=torch.cuda.current_device(),
                                                 dtype=dtype))
    else:
        tensor_recv_nexts = [None] * len(recv_next_chunk_shapes)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        for idx in range(len(tensor_send_next)):
            if tensor_send_next[idx] is not None:
                tensor_send_next[idx] = core.tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_next[idx])
        for idx in range(len(tensor_send_prev)):
            if tensor_send_prev[idx] is not None:
                tensor_send_prev[idx] = core.tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_prev[idx])

    # Send tensors in both the forward and backward directions as appropriate.
    if args.use_ring_exchange_p2p:
        assert len(tensor_send_next) == 1, "Not currently supporting multiple tensors in ring exchange"
        assert len(tensor_send_prev) == 1, "Not currently supporting multiple tensors in ring exchange"
        assert len(tensor_recv_nexts) == 1, "Not currently supporting multiple tensors in ring exchange"
        assert len(tensor_recv_prevs) == 1, "Not currently supporting multiple tensors in ring exchange"
        torch.distributed.ring_exchange(tensor_send_prev=tensor_send_prev[0],
                                        tensor_recv_prev=tensor_recv_prevs[0],
                                        tensor_send_next=tensor_send_next[0],
                                        tensor_recv_next=tensor_recv_nexts[0],
                                        group=mpu.get_pipeline_model_parallel_group())
    else:
        ops = []
        for tensor in tensor_send_prev:
            if tensor is not None:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor,
                    mpu.get_pipeline_model_parallel_prev_rank())
                ops.append(send_prev_op)
        for tensor in tensor_recv_prevs:
            if tensor is not None:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor,
                    mpu.get_pipeline_model_parallel_prev_rank())
                ops.append(recv_prev_op)
        for tensor in tensor_send_next:
            if tensor is not None:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor,
                    mpu.get_pipeline_model_parallel_next_rank())
                ops.append(send_next_op)
        for tensor in tensor_recv_nexts:
            if tensor is not None:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv, tensor,
                    mpu.get_pipeline_model_parallel_next_rank())
                ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        if recv_prev:
            for idx in range(len(tensor_recv_prevs)):
                tensor_recv_prevs[idx] = core.tensor_parallel.gather_split_1d_tensor(
                    tensor_recv_prevs[idx]).view(recv_prev_shape[idx]).requires_grad_()
                tensor_recv_prev[idx] = core.utils.make_viewless_tensor(tensor_recv_prev[idx],
                                                                requires_grad=True,
                                                                keep_graph=False)

        if recv_next:
            for idx in range(len(tensor_recv_nexts)):
                tensor_recv_nexts[idx] = core.tensor_parallel.gather_split_1d_tensor(
                    tensor_recv_nexts[idx]).view(recv_next_shape[idx]).requires_grad_()
                tensor_recv_next[idx] = core.utils.make_viewless_tensor(tensor_recv_next[idx],
                                                                requires_grad=True,
                                                                keep_graph=False)
    if isinstance(tensor_recv_prevs, list) and len(tensor_recv_prevs) == 0:
        tensor_recv_prevs = None
    if isinstance(tensor_recv_nexts, list) and len(tensor_recv_nexts) == 0:
        tensor_recv_nexts = None
    if unwrap_recv_prev and recv_prev and isinstance(tensor_recv_prevs, list):
        assert len(tensor_recv_prevs) == 1
        tensor_recv_prevs = tensor_recv_prevs[0]
    if unwrap_recv_next and recv_next and isinstance(tensor_recv_nexts, list):
        assert len(tensor_recv_nexts) == 1
        tensor_recv_nexts = tensor_recv_nexts[0]
    return tensor_recv_prevs, tensor_recv_nexts


def recv_forward(tensor_shape=None, dtype_=None, timers=None):
    """Receive tensor from previous rank in pipeline (forward receive)."""

    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('forward-recv', log_level=2).start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype_)
        if timers is not None:
            timers('forward-recv').stop()
    return input_tensor


def recv_backward(tensor_shape=None, timers=None):
    """Receive tensor from next rank in pipeline (backward receive)."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('backward-recv', log_level=2).start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('backward-recv').stop()
    return output_tensor_grad


def send_forward(output_tensor, tensor_shape=None, dtype_=None, timers=None):
    """Send tensor to next rank in pipeline (forward send)."""

    if not mpu.is_pipeline_last_stage():
        if timers is not None:
            timers('forward-send', log_level=2).start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype_)
        if timers is not None:
            timers('forward-send').stop()


def send_backward(input_tensor_grad, tensor_shape=None, timers=None):
    """Send tensor to previous rank in pipeline (backward send)."""
    if not mpu.is_pipeline_first_stage():
        if timers is not None:
            timers('backward-send', log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('backward-send').stop()


def send_forward_recv_backward(output_tensor, tensor_shape=None, timers=None):
    """Batched send and recv with next rank in pipeline."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, tensor_shape=None, timers=None):
    """Batched send and recv with previous rank in pipeline."""
    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(output_tensor, recv_prev, tensor_shape=None, timers=None):
    """Batched recv from previous rank and send to next rank in pipeline."""
    if timers is not None:
        timers('forward-send-forward-recv', log_level=2).start()
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('forward-send-forward-recv').stop()
    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next, tensor_shape=None, timers=None):
    """Batched recv from next rank and send to previous rank in pipeline."""
    if timers is not None:
        timers('backward-send-backward-recv', log_level=2).start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('backward-send-backward-recv').stop()
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor, input_tensor_grad, recv_prev,
        recv_next, tensor_shape=None, receive_prev_shape=None,
        receive_next_shape=None, timers=None):
    """Batched send and recv with previous and next ranks in pipeline."""
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv',
               log_level=2).start()
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        recv_prev_shape=receive_prev_shape,
        recv_next_shape=receive_next_shape)
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
