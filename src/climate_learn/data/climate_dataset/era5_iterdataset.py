# for ITERATIVE sequence prediction (preferred choice for now). 
# This is a modified version of the iterdataset.py file. It is labelled as 'experimental'

# Standard library
import math
import os
import random
from typing import Union

# Third party
import numpy as np
import torch
from torch.utils.data import IterableDataset


def shuffle_two_list(list1, list2):
    """
    Shuffle two lists in unison, preserving the correspondence between elements.

    Args:
        list1: The first list to shuffle.
        list2: The second list to shuffle.

    Returns:
        A tuple (list1_shuf, list2_shuf) with the lists shuffled in the same order.
    """
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(list1)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf, list2_shuf


class NpyReader(IterableDataset):
    """
    Dataset class for reading data from numpy (.npy) files.

    This class loads inputs and outputs from corresponding .npy files, filtering out any paths
    that contain the substring "climatology". Optionally, the file lists can be shuffled.

    Parameters:
        inp_file_list: List of file paths for input data.
        out_file_list: List of file paths for output data.
        variables: List of variable keys to extract from the input npy files.
        out_variables: List of variable keys to extract from the output npy files. If None, defaults to variables.
        shuffle (bool): If True, shuffles file order before iteration.
    """
    def __init__(
        self,
        inp_file_list,
        out_file_list,
        variables,
        out_variables,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        assert len(inp_file_list) == len(out_file_list)
        # Exclude files with "climatology" from file lists.
        self.inp_file_list = [f for f in inp_file_list if "climatology" not in f]
        self.out_file_list = [f for f in out_file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle

    def __iter__(self):
        """
        Iterates over the list of file paths, loading and yielding data.

        If shuffling is enabled, file lists are shuffled. Moreover, if running in a
        multi-worker/distributed setting, each worker is assigned a distinct subset of the data.
        """
        if self.shuffle:
            self.inp_file_list, self.out_file_list = shuffle_two_list(
                self.inp_file_list, self.out_file_list
            )

        n_files = len(self.inp_file_list)

        # Get worker info for potential multi-threaded or distributed loading
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process scenario
            iter_start = 0
            iter_end = n_files
        else:
            # Multi-worker: split data among workers and (if applicable) among distributed processes
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size
            per_worker = n_files // num_shards
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        # Iterate over the assigned file range
        for idx in range(iter_start, iter_end):
            path_inp = self.inp_file_list[idx]
            path_out = self.out_file_list[idx]
            inp = np.load(path_inp)
            # If input and output paths are identical, use the same data for output.
            if path_out == path_inp:
                out = inp
            else:
                out = np.load(path_out)
            # Yield dictionaries mapping variable names to the corresponding squeezed numpy arrays,
            # along with the list of variables.
            yield {k: np.squeeze(inp[k], axis=1) for k in self.variables}, {
                k: np.squeeze(out[k], axis=1) for k in self.out_variables
            }, self.variables, self.out_variables


class Forecast(IterableDataset):
    """
    Dataset class for iterative sequence prediction tasks.

    This class processes data from an underlying NpyReader instance to prepare input sequences
    for forecasting. It constructs sequences with repeated historical context and shifts them in time
    to simulate prediction scenarios.

    Parameters:
        dataset: An instance of NpyReader providing raw data.
        pred_range: The number of future time steps to predict.
        history: Number of historical time steps to include (repeated for context).
        window: Number of steps to roll/shift the historical data for prediction.
    """
    def __init__(
        self, dataset: NpyReader, pred_range: int = 6, history: int = 3, window: int = 6
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.pred_range = pred_range
        self.history = history
        self.window = window

    def __iter__(self):
        """
        Processes each batch from the underlying dataset to construct forecast sequences.

        For each sample:
          - Converts numpy arrays to torch tensors.
          - Repeats the input data along a new history dimension.
          - Applies a rolling window shift to simulate evolving historical context.
          - Determines the correct slice range for input and the corresponding prediction indices.
          
        Yields:
            A tuple (inp_data, out_data, variables, out_variables) where:
              - inp_data: Processed input sequence tensor (shape: N, T, H, W).
              - out_data: Target data tensor aligned with prediction range.
              - variables: List of variable names.
              - out_variables: List of output variable names.
        """
        for inp_data, out_data, variables, out_variables in self.dataset:
            # Convert input numpy arrays to float32 tensors and add a history dimension by repeating.
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                .unsqueeze(0)  # Add a new dimension for the history
                .repeat_interleave(self.history, dim=0)  # Repeat data along history dimension
                for k in inp_data.keys()
            }
            # Convert output numpy arrays to float32 tensors.
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            # For each variable in the input, shift the data along the time dimension for each historical step.
            for key in inp_data.keys():
                for t in range(self.history):
                    # Roll the tensor to simulate temporal progression
                    inp_data[key][t] = inp_data[key][t].roll(-t * self.window, dims=0)

            # Determine the last index to slice the historical sequence properly.
            last_idx = -((self.history - 1) * self.window + self.pred_range)

            # Transpose the history and time dimensions so that the final shape is (N, T, H, W)
            inp_data = {
                k: inp_data[k][:, :last_idx].transpose(0, 1)
                for k in inp_data.keys()  # N, T, H, W
            }

            inp_data_len = inp_data[variables[0]].size(0)

            # Create a tensor with the prediction range repeated for indexing.
            predict_ranges = torch.ones(inp_data_len).to(torch.long) * self.pred_range
            # Determine the indices for which the output should be sampled.
            output_ids = (
                torch.arange(inp_data_len)
                + (self.history - 1) * self.window
                + predict_ranges
            )
            # Select the output data corresponding to the computed indices.
            out_data = {k: out_data[k][output_ids] for k in out_data.keys()}
            yield inp_data, out_data, variables, out_variables


class Downscale(IterableDataset):
    """
    Dataset class for downscaling tasks.

    This class converts raw numpy array data from an NpyReader into PyTorch tensors without
    generating multiple historical sequences. It is intended for static or non-sequential tasks.

    Parameters:
        dataset: An instance of NpyReader providing raw data.
    """
    def __init__(self, dataset: NpyReader) -> None:
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        """
        Iterates over the underlying dataset, converting numpy arrays to PyTorch tensors.

        Yields:
            A tuple (inp_data, out_data, variables, out_variables) where each dictionary
            maps variable names to their corresponding tensors.
        """
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            yield inp_data, out_data, variables, out_variables


class IndividualDataIter(IterableDataset):
    """
    Iterator that subsamples from a batched dataset.

    This iterator goes through batched data provided by Forecast or Downscale, applies input and 
    output transformations (if provided), and yields individual samples from the batch.

    Parameters:
        dataset: An instance of Forecast or Downscale providing batched data.
        transforms: A dictionary or module of input transformations to apply.
        output_transforms: A dictionary or module of output transformations to apply.
        subsample (int): Step size for subsampling individual items from the batched data.
    """
    def __init__(
        self,
        dataset: Union[Forecast, Downscale],
        transforms: torch.nn.Module,
        output_transforms: torch.nn.Module,
        subsample: int = 6,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.subsample = subsample

    def __iter__(self):
        """
        Iterates over the batched data and yields individual samples.

        It asserts that all inputs (and outputs) in the batch have the same length,
        applies the specified transformations, and subsamples individual elements.
        
        Yields:
            A tuple (x, y, variables, out_variables) for each individual sample.
        """
        for inp, out, variables, out_variables in self.dataset:
            inp_shapes = set([inp[k].shape[0] for k in inp.keys()])
            out_shapes = set([out[k].shape[0] for k in out.keys()])
            # Ensure consistent batch lengths across variables.
            assert len(inp_shapes) == 1
            assert len(out_shapes) == 1
            inp_len = next(iter(inp_shapes))
            out_len = next(iter(out_shapes))
            assert inp_len == out_len
            # Subsample individual items from the batch.
            for i in range(0, inp_len, self.subsample):
                x = {k: inp[k][i] for k in inp.keys()}
                y = {k: out[k][i] for k in out.keys()}
                # Apply input transformations based on the dataset type.
                if self.transforms is not None:
                    if isinstance(self.dataset, Forecast):
                        x = {
                            k: self.transforms[k](x[k].unsqueeze(1)).squeeze(1)
                            for k in x.keys()
                        }
                    elif isinstance(self.dataset, Downscale):
                        x = {
                            k: self.transforms[k](x[k].unsqueeze(0)).squeeze(0)
                            for k in x.keys()
                        }
                    else:
                        raise RuntimeError(f"Not supported task.")
                # Apply output transformations.
                if self.output_transforms is not None:
                    y = {
                        k: self.output_transforms[k](y[k].unsqueeze(0)).squeeze(0)
                        for k in y.keys()
                    }
                yield x, y, variables, out_variables


class ShuffleIterableDataset(IterableDataset):
    """
    Buffer-based shuffling dataset wrapper.

    This class maintains a fixed-size buffer of items from the underlying dataset and
    yields them in random order. It is useful for shuffling iterable datasets that do not 
    support random access.

    Parameters:
        dataset: An instance of IndividualDataIter from which data items are obtained.
        buffer_size (int): Size of the buffer used for shuffling; must be > 0.
    """
    def __init__(self, dataset: IndividualDataIter, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        """
        Iterates over the dataset while performing buffered shuffling.

        Fills a buffer with items from the dataset and, when the buffer is full, randomly selects 
        an item to yield while replacing it with the next item. After the dataset is exhausted,
        remaining items in the buffer are randomly shuffled and yielded.
        """
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()
