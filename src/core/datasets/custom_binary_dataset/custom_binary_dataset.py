import os

import h5py

from src.core.datasets.ecg_dataset import EcgDataset


class SingleLeadBinaryDataset(EcgDataset):
    """
    Dataset produced by src/scripts/custom_dataset/prepare_binary_datasets.py for the single-lead task.

    Every sample is stored as a single channel, regardless of whether it originally came from lead_0 or lead_1.
    The original lead identity is preserved in the metadata csv under the source_lead column.
    """

    def __init__(self, root_path, **dataset_kwargs):
        self._h5_path = os.path.join(root_path, f"{self.__class__.__name__}.h5")
        self._h5_file = h5py.File(self._h5_path, "r")["tracings"]
        super().__init__(root_path=root_path, **dataset_kwargs)

    @property
    def h5_file(self):
        return self._h5_file

    def __len__(self):
        return len(self._data_y)

    def transform_lead_selection(self, custom_lead_selection):
        # Every sample in this dataset contains exactly one channel.
        return [0]


class TwoLeadBinaryDataset(EcgDataset):
    """
    Dataset produced by src/scripts/custom_dataset/prepare_binary_datasets.py for the two-lead task.

    Channel order for this custom dataset is:
    - channel 0 -> lead II
    - channel 1 -> V2
    """

    LEAD_ALIASES = {
        "lead_0": 0,
        "lead0": 0,
        "ii": 0,
        "lead_1": 1,
        "lead1": 1,
        "v2": 1,
    }

    def __init__(self, root_path, **dataset_kwargs):
        self._h5_path = os.path.join(root_path, f"{self.__class__.__name__}.h5")
        self._h5_file = h5py.File(self._h5_path, "r")["tracings"]
        super().__init__(root_path=root_path, **dataset_kwargs)

    @property
    def h5_file(self):
        return self._h5_file

    def __len__(self):
        return len(self._data_y)

    def transform_lead_selection(self, custom_lead_selection):
        requested = {elem.strip().lower() for elem in custom_lead_selection.split(",")}
        if "all_leads" in requested:
            return [0, 1]

        indices = []
        for lead in requested:
            if lead not in self.LEAD_ALIASES:
                raise AssertionError(
                    "TwoLeadBinaryDataset supports lead selections from {lead_0, lead_1, II, V2, all_leads}"
                )
            indices.append(self.LEAD_ALIASES[lead])
        return sorted(set(indices))
