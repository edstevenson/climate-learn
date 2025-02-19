'''
for ITERATIVE sequence prediction (preferred choice for now)
'''

# Standard library
import random

# Third party
import numpy as np
import torch
from torch.utils.data import IterableDataset


def shuffle_two_list(list1, list2):
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
        shuffle=False,
    ):
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


class DirectForecast(IterableDataset):
    """
    Dataset for direct (single-step) forecasting tasks.
    
    Handles:
    - Loading historical context (multiple time steps)
    - Rolling window processing for temporal data
    - Different data sources (ERA5, MPI-ESM) with different temporal resolutions
    
    Args:
        dataset: Base dataset providing the data
        src: Source dataset type ('era5' or 'mpi-esm1-2-hr')
        pred_range: Number of time steps to predict ahead
        history: Number of historical time steps to use
        window: Size of rolling window for temporal processing
    """
    def __init__(self, dataset, src, pred_range=6, history=3, window=6):
        super().__init__()
        self.dataset = dataset
        self.history = history
        # Adjust prediction range and window size based on data source
        if src == "era5":
            print("ERA5 case")
            self.pred_range = pred_range
            self.window = window
        elif src == "mpi-esm1-2-hr":
            print("MPI-ESM case")
            assert pred_range % 6 == 0
            assert window % 6 == 0
            self.pred_range = pred_range // 6
            self.window = window // 6
        else:
            raise ValueError(f"Unsupported source: {src}")

    def __iter__(self):
        """
        Iterate over dataset, preparing input sequences and corresponding targets.
        
        For each sample:
        1. Converts data to torch tensors
        2. Creates historical context by repeating and rolling data
        3. Aligns input sequences with target predictions
        """
        for inp_data, out_data, variables, out_variables in self.dataset:
            # Create historical context by repeating input data
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                .unsqueeze(0)
                .repeat_interleave(self.history, dim=0)
                for k in inp_data.keys()
            }
            # Convert output data to tensors
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            # Apply rolling window to create temporal context
            for key in inp_data.keys():
                for t in range(self.history):
                    inp_data[key][t] = inp_data[key][t].roll(-t * self.window, dims=0)

            # Calculate last valid index considering history and prediction range
            last_idx = -((self.history - 1) * self.window + self.pred_range)
            
            # Prepare input sequences
            inp_data = {
                k: inp_data[k][:, :last_idx].transpose(0, 1)
                for k in inp_data.keys()  # Shape: [N, T, H, W]
            }

            # Calculate output indices based on prediction range
            inp_data_len = inp_data[variables[0]].size(0)
            predict_ranges = torch.ones(inp_data_len).to(torch.long) * self.pred_range
            output_ids = (
                torch.arange(inp_data_len)
                + (self.history - 1) * self.window
                + predict_ranges
            )
            # Select corresponding outputs
            out_data = {k: out_data[k][output_ids] for k in out_data.keys()}
            yield inp_data, out_data, variables, out_variables


class ContinuousForecast(IterableDataset):
    """
    Dataset for continuous forecasting with variable lead times.
    
    Similar to DirectForecast but supports:
    - Random or fixed prediction ranges
    - Lead time information in the input
    """
    def __init__(
        self,
        dataset,
        random_lead_time=True,
        min_pred_range=6,
        max_pred_range=120,
        hrs_each_step=1,
        history=3,
        window=6,
    ):
        super().__init__()
        if not random_lead_time:
            assert min_pred_range == max_pred_range
        self.dataset = dataset
        self.random_lead_time = random_lead_time
        self.min_pred_range = min_pred_range
        self.max_pred_range = max_pred_range
        self.hrs_each_step = hrs_each_step
        self.history = history
        self.window = window

    def __iter__(self):
        for inp_data, out_data, variables, out_variables in self.dataset:
            inp_data = {
                k: torch.from_numpy(inp_data[k].astype(np.float32))
                .unsqueeze(0)
                .repeat_interleave(self.history, dim=0)
                for k in inp_data.keys()
            }
            out_data = {
                k: torch.from_numpy(out_data[k].astype(np.float32))
                for k in out_data.keys()
            }
            for key in inp_data.keys():
                for t in range(self.history):
                    inp_data[key][t] = inp_data[key][t].roll(-t * self.window, dims=0)

            last_idx = -((self.history - 1) * self.window + self.max_pred_range)

            inp_data = {
                k: inp_data[k][:, :last_idx].transpose(0, 1)
                for k in inp_data.keys()  # N, T, H, W
            }

            inp_data_len = inp_data[variables[0]].size(0)
            dtype = inp_data[variables[0]].dtype

            if self.random_lead_time:
                predict_ranges = torch.randint(
                    low=self.min_pred_range,
                    high=self.max_pred_range + 1,
                    size=(inp_data_len,),
                )
            else:
                predict_ranges = (
                    torch.ones(inp_data_len).to(torch.long) * self.max_pred_range
                )
            lead_times = self.hrs_each_step * predict_ranges / 100
            lead_times = lead_times.to(dtype)
            output_ids = (
                torch.arange(inp_data_len)
                + (self.history - 1) * self.window
                + predict_ranges
            )

            out_data = {k: out_data[k][output_ids] for k in out_data.keys()}
            yield inp_data, out_data, lead_times, variables, out_variables


class Downscale(IterableDataset):
    """
    Dataset for spatial downscaling tasks.
    
    Simpler than forecasting datasets as it only needs to:
    1. Load low-resolution input and high-resolution target data
    2. Convert to torch tensors
    3. No temporal processing needed
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
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
    def __init__(
        self,
        dataset,
        transforms,
        output_transforms,
        subsample=6,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.subsample = subsample

    def __iter__(self):
        for sample in self.dataset:
            if isinstance(self.dataset, (DirectForecast, Downscale)):
                inp, out, variables, out_variables = sample
            elif isinstance(self.dataset, ContinuousForecast):
                inp, out, lead_times, variables, out_variables = sample
            inp_shapes = set([inp[k].shape[0] for k in inp.keys()])
            out_shapes = set([out[k].shape[0] for k in out.keys()])
            assert len(inp_shapes) == 1
            assert len(out_shapes) == 1
            inp_len = next(iter(inp_shapes))
            out_len = next(iter(out_shapes))
            assert inp_len == out_len
            for i in range(0, inp_len, self.subsample):
                x = {k: inp[k][i] for k in inp.keys()}
                y = {k: out[k][i] for k in out.keys()}
                if self.transforms is not None:
                    if isinstance(self.dataset, (DirectForecast, ContinuousForecast)):
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
                if self.output_transforms is not None:
                    y = {
                        k: self.output_transforms[k](y[k].unsqueeze(0)).squeeze(0)
                        for k in y.keys()
                    }
                if isinstance(self.dataset, (DirectForecast, Downscale)):
                    result = x, y, variables, out_variables
                elif isinstance(self.dataset, ContinuousForecast):
                    result = x, y, lead_times[i], variables, out_variables
                yield result


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
