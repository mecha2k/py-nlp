import numpy as np
import os
import json
from glob import glob
from tqdm import tqdm


data_dir = "../data/cnn_daily/cnn_dm/json.gz"
data_types = ["val", "test", "train"]

# for data_type in data_types:
#     inferred_type = "txt"
#     dataset_files = glob(os.path.join(data_dir, "*" + data_type + ".*.txt"))
#
#     data_load = list()
#     for file in tqdm(dataset_files):
#         with open(file, "r", encoding="utf-8") as f:
#             data = [json.loads(line) for line in f.readlines()]
#         data_load.append(data)
#     datasets = np.concatenate(data_load, axis=0)
#     np.save(f"../data/cnn_daily/cnn_dm/datasets_{data_type}.npy", datasets)

for data_type in data_types:
    datasets = np.load(f"../data/cnn_daily/cnn_dm/datasets_{data_type}.npy", allow_pickle=True)
    print(datasets.shape)
    datasets_small = datasets[:1000]
    np.save(f"../data/cnn_daily/cnn_dm/datasets_{data_type}_small.npy", datasets_small)
    print(datasets_small.shape)
