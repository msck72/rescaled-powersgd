from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Union

import torch

from powersgd.orthogonalization import orthogonalize
from powersgd.utils import allreduce_average, pack, unpack, is_distributed, find_factors

import math

class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        This method also changes its input gradients. It either sets them to zero if there is no compression,
        or to the compression errors, for error feedback.
        """
        pass

    @abstractmethod
    def aggregate_using_rescaled_grads(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

class AllReduce(Aggregator):
    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(gradients) == 0:
            return []
        return gradients
    
    def aggregate_using_rescaled_grads(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(gradients) == 0:
            return []
        return gradients


class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    start_compressing_after_num_steps: int = 100


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: List[torch.Tensor], config: Config):
        self.config = config
        self.device = list(params)[0].device
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]

        self.step_counter = 0
        self.step_counter_rescaled = 0

        compressed_params, _ = self._split(params)
        self._powersgd = BasicPowerSGD(
            compressed_params,
            config=BasicConfig(
                rank=config.rank,
                num_iters_per_step=config.num_iters_per_step,
            ),
        )
        self._allreduce = AllReduce()

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        self.step_counter += 1

        compressed_grads, uncompressed_grads = self._split(gradients)
        return self._merge(
            self._powersgd.aggregate(compressed_grads),
            self._allreduce.aggregate(uncompressed_grads),
        )
    
    def aggregate_using_rescaled_grads(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        self.step_counter_rescaled += 1

        compressed_grads, uncompressed_grads = self._split(gradients)
        return self._merge(
            self._powersgd.aggregate_using_rescaled_grads(compressed_grads),
            self._allreduce.aggregate(uncompressed_grads),
        )
    
    def _split(self, params: List[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: List[torch.Tensor], uncompressed: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )


class BasicConfig(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    num_iters_per_step: int = 1  # lower number => more aggressive compression


class BasicPowerSGD(Aggregator):
    def __init__(self, params: List[torch.Tensor], config: BasicConfig):
        # Configuration
        self.config = config
        self.params = list(params)
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.params_per_shape = self._matrices_per_shape(self.params)

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0
        self.step_counter_rescaled = 0

        # Initilize and allocate the low rank approximation matrices p and q.
        # _ps_buffer and _qs_buffer are contiguous memory that can be easily all-reduced, and
        # _ps and _qs are pointers into this memory.
        # _ps and _qs represent batches p/q for all tensors of the same shape.
        self._ps_buffer, ps_shapes = pack(
            [
                self._init_p_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._ps = unpack(self._ps_buffer, ps_shapes)

        self._qs_buffer, qs_shapes = pack(
            [
                self._init_q_batch(shape, params)
                for shape, params in self.params_per_shape.items()
            ]
        )
        self._qs = unpack(self._qs_buffer, qs_shapes)


        # For rescaled estimate of the gradient
        # get the number of parameters in the params
        grads_count = [torch.numel(p) for p in params]
        total_grads_count = sum(grads_count)
        
        self.num_matrices = len(grads_count)

        num_grads_per_matrix = total_grads_count / self.num_matrices

        self.mat_dim = int(math.sqrt(num_grads_per_matrix))
        self.rem_grads = total_grads_count - (self.num_matrices * (self.mat_dim * self.mat_dim))
        
        # find the factors of the remaining elements such that it forms almost a square matrix, i.e. the difference is minimal
        dim1, dim2 = find_factors(self.rem_grads)
        self.rem_grads_dim = (dim1, dim2)


        # eff_num_grads_in_mat = self.mat_dim * self.mat_dim
        # eff_num_grads_left_out = total_grads_count - (len(grads_count) * eff_num_grads_in_mat)
        self._ps_rescaled_buffer, _ps_rescaled_shapes = pack([self._init_p_batch_with_len((self.mat_dim, self.mat_dim), self.num_matrices), self._init_p_batch_with_len((dim1, dim2), 1)])
        self._ps_rescaled = unpack(self._ps_rescaled_buffer, _ps_rescaled_shapes)
    
        self._qs_rescaled_buffer, qs_rescaled_shapes = pack([self._init_q_batch_with_len((self.mat_dim, self.mat_dim), self.num_matrices), self._init_q_batch_with_len((dim1, dim2), 1)])
        self._qs_rescaled = unpack(self._qs_rescaled_buffer, qs_rescaled_shapes)

        # divide those parameters equally by m (a hyper-parameter)
        # convert those into sqaure matrices and batch them
        # with rank as input now initialize the p and q matrices
        
        
        
        print(f"POWERSGD num_parameters = {torch.numel(self._ps_buffer) + torch.numel(self._qs_buffer)}")
        print(f"RESCALED POWERSGD num_parameters = {torch.numel(self._ps_rescaled_buffer) + torch.numel(self._qs_rescaled_buffer)}")
        

    def aggregate(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Create a low-rank approximation of the average gradients by communicating with other workers.
        Modifies its inputs so that they contain the 'approximation error', used for the error feedback
        mechanism.
        """
        # Allocate memory for the return value of this function
        output_tensors = [torch.empty_like(g) for g in gradients]

        # Group the gradients per shape, and view them as matrices (2D tensors)
        gradients_per_shape = self._matrices_per_shape(gradients)
        outputs_per_shape = self._matrices_per_shape(output_tensors)
        shape_groups = [
            dict(
                shape=shape,
                grads=matrices,
                outputs=outputs_per_shape[shape],
                grad_batch=torch.stack(matrices),
                approximation=torch.zeros(
                    size=(len(matrices), *shape), device=self.device, dtype=self.dtype
                ),
            )
            for shape, matrices in list(gradients_per_shape.items())
        ]

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs, self._ps
                out_buffer = self._qs_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps, self._qs
                out_buffer = self._ps_buffer

            # Matrix multiplication
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(group["grad_batch"])), 
                    in_batch, 
                    out=out_batch
                )

            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["grad_batch"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch), 
                    alpha=-1
                )

            # Construct low-rank reconstruction and update the approximation and error buffer
            for group, in_batch, out_batch in zip(
                shape_groups, in_batches, out_batches
            ):
                maybe_transpose(group["approximation"]).baddbmm_(
                    in_batch, 
                    batch_transpose(out_batch),
                    # alpha=1
                )

        # Un-batch the approximation and error feedback, write to the output
        for group in shape_groups:
            for o, m, approx, mb in zip(
                group["outputs"],
                group["grads"],
                group["approximation"],
                group["grad_batch"],
            ):
                o.copy_(approx)
                m.copy_(mb)

        # Increment the step counter
        self.step_counter += 1

        return output_tensors
    
    def aggregate_using_rescaled_grads(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        print('In aggreagte usnig rescaled grads')
        # flatten the gradient into a single tensor and also remember the shape for re-building in future
        unsqueezed_shapes = []
        squeezed_grads = []
        for g in gradients:
            unsqueezed_shapes.append((torch.numel(g), g.shape))
            squeezed_grads.append(g.view(-1))
        single_dim_grads = torch.cat(squeezed_grads)

        # build a batched rescaled gradients square matrices
        rescaled_grads = []
        left, right = 0, self.mat_dim * self.mat_dim
        for i in range(self.num_matrices):
            rescaled_grads.append(single_dim_grads[left:right].view(self.mat_dim, self.mat_dim))
            left, right = right, right + (self.mat_dim * self.mat_dim)
        
        rem_grad_mat = single_dim_grads[left:].view(self.rem_grads_dim[0], self.rem_grads_dim[1]).unsqueeze(0)

        assert (self.num_matrices * (self.mat_dim * self.mat_dim)) + torch.numel(rem_grad_mat) == torch.numel(single_dim_grads)

        grad_batches = [torch.stack(rescaled_grads), rem_grad_mat]
        
        # for grad_batch in grad_batches:
        #     print(f'grad_batch.shape = ', grad_batch.shape)
        

        num_iters_per_step = self.config.num_iters_per_step
        for it in range(num_iters_per_step):
            # Alternate between left and right matrix multiplications
            iter_is_even = (self.step_counter * num_iters_per_step + it) % 2 == 0
            if iter_is_even:
                maybe_transpose = lambda g: g
                out_batches, in_batches = self._qs_rescaled, self._ps_rescaled
                out_buffer = self._qs_rescaled_buffer
            else:
                maybe_transpose = batch_transpose
                out_batches, in_batches = self._ps_rescaled, self._qs_rescaled
                out_buffer = self._ps_rescaled_buffer

            # Matrix multiplication to estimate the out-batch
            
            for grad_batch, in_batch, out_batch in zip(grad_batches, in_batches, out_batches):
                orthogonalize(in_batch)
                torch.bmm(
                    batch_transpose(maybe_transpose(grad_batch)), 
                    in_batch, 
                    out=out_batch
                )

            # put the error in the grad-batch
            # maybe_transpose(grad_batch).baddbmm_(
            #     in_batch, 
            #     batch_transpose(out_batch), 
            #     alpha=-1
            # )

            for grad_batch, in_batch, out_batch in zip(grad_batches, in_batches, out_batches):
                torch.bmm(
                    in_batch, 
                    batch_transpose(out_batch),
                    out=maybe_transpose(grad_batch)     
                )

        # rebuild the gradient
        output_tensors = []
        single_dim_approx_grads = torch.cat([grad_batches[0].view(-1), grad_batches[1].view(-1)])
        left, right = 0, 0
        for num_elements, shape in unsqueezed_shapes:
            left, right = right, right + num_elements
            output_tensors.append(single_dim_approx_grads[left:right].view(shape))

        # Increment the step counter
        self.step_counter_rescaled += 1

        # output_tensors.append(grad_batches[1].view(self.rem_grads_dim))
        
        # for o in output_tensors:
        #     print(o.shape)
        return output_tensors
    

    def _init_p_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch(
        self, shape: torch.Size, params: List[torch.Tensor]
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [len(params), shape[1], rank], generator=self.generator, device=self.device
        )
    

    def _init_p_batch_with_len(
        self, shape: torch.Size, params_len: int
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [params_len, shape[0], rank], generator=self.generator, device=self.device
        )

    def _init_q_batch_with_len(
        self, shape: torch.Size, params_len: int
    ) -> torch.Tensor:
        rank = min(self.config.rank, min(shape))
        return torch.randn(
            [params_len, shape[1], rank], generator=self.generator, device=self.device
        )

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: List[torch.Tensor],
    ) -> Dict[torch.Size, List[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats



def batch_transpose(batch_of_matrices):
    return batch_of_matrices.permute([0, 2, 1])


def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)


def avg_compressed_size(shape: torch.Size, config: Union[Config, BasicConfig]) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)