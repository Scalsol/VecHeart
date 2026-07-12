import bisect

from reimu.runner import get_dist_info
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.separate_eval = separate_eval

        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Total {len(self)} images for using.")

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
            sample = self.datasets[dataset_idx][sample_idx]
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            sample = self.datasets[dataset_idx][sample_idx]
            if 'idx' in sample:
                sample['idx'] += self.cumulative_sizes[dataset_idx - 1]
        return sample

    def evaluate(self, results, metrics="mDice", num_classes=1, **kwargs):
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print(f'Evaluateing {dataset.data_root} with {len(results_per_dataset)} images.')

                eval_results_per_dataset = dataset.evaluate(results_per_dataset, metrics, num_classes, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results
        else:
             raise NotImplementedError("Currently doesn't support unified evaluation!")
