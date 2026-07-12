import glob
import os

import numpy as np
from sklearn.model_selection import KFold


def get_split(data, idx):
    return list(np.array(data)[idx])


def load_data(path, files_pattern, non_empty=True):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    if non_empty:
        assert len(data) > 0, f"No data found in {path} with pattern {files_pattern}"
    return data


def get_kfold_splitter(nfolds):
    return KFold(n_splits=nfolds, shuffle=True, random_state=12345)
