from reimu.runner import get_dist_info

from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from .utils import get_kfold_splitter, get_split, load_data


@DATASETS.register_module()
class SDFDataset(BaseDataset):
    """
    Label 0  # Background
    Label 1  # MYO-LV: myocardium of the left ventricle
    Label 2  # LV: left ventricle blood cavity
    Label 3  # RV: right ventricle blood cavity
    Label 4  # LA: left atrium blood cavity
    Label 5  # RA: right atrium blood cavity
    """
    def __init__(self, pipeline, data_root, nfolds=5, fold=0, test_mode=False, load_slice=False):
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.kfold = get_kfold_splitter(nfolds)
        self.test_mode = test_mode
        self.load_slice = load_slice

        # sdf
        filenames = []
        filenames.extend(load_data(data_root, "*/*sdf.npz", non_empty=False))

        # slice pts
        filenames_slice = []
        if self.load_slice:
            filenames_slice.extend(load_data(data_root, "*/*slice_pts.npz", non_empty=False))

        train_idx, val_idx = list(self.kfold.split(filenames))[fold]
        idx = val_idx if self.test_mode else train_idx

        self.samples = get_split(filenames, idx)
        if self.load_slice:
            self.slices = get_split(filenames_slice, idx)

        rank, _ = get_dist_info()
        if rank == 0:
            print(f"{len(self.samples)} images for using.")

    def __len__(self):
        """Total number of instances in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load image and label."""
        if self.test_mode:
            return self.prepare_test_instance(idx)
        else:
            return self.prepare_train_instance(idx)

    def prepare_train_instance(self, idx):
        sample_info = self.samples[idx]
        results = dict(sample_info=sample_info, idx=idx)
        if self.load_slice:
            results['slice_info'] = self.slices[idx]

        return self.pipeline(results)

    def prepare_test_instance(self, idx):
        sample_info = self.samples[idx]
        results = dict(sample_info=sample_info, idx=idx)
        if self.load_slice:
            results['slice_info'] = self.slices[idx]

        return self.pipeline(results)
