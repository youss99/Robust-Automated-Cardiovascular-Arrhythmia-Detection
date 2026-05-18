import os

import h5py

from src.core.datasets.ecg_dataset import EcgDataset


class PatientSegmentRegressionDataset(EcgDataset):
    """
    Dataset produced by src/scripts/custom_dataset/prepare_regression_datasets.py.

    The H5 channel order is recorded in modified_dataset.csv under source_leads, and
    regression targets are stored in target_lead{lead_id}_mV columns. Lead selection
    therefore selects both the input channels and the matching target values.
    """

    LEAD_ALIASES = {
        "lead_0": 0,
        "lead0": 0,
        "0": 0,
        "ii": 0,
        "lead_1": 1,
        "lead1": 1,
        "1": 1,
        "v2": 1,
    }

    def __init__(self, root_path, **dataset_kwargs):
        self._h5_path = os.path.join(root_path, f"{self.__class__.__name__}.h5")
        self._h5_file = h5py.File(self._h5_path, "r")["tracings"]
        self._source_leads_cache = None
        super().__init__(root_path=root_path, **dataset_kwargs)

    @property
    def h5_file(self):
        return self._h5_file

    def __len__(self):
        return len(self._data_y)

    def _source_leads(self):
        if self._source_leads_cache is not None:
            return self._source_leads_cache

        if "source_leads" in self._data_y.columns and not self._data_y["source_leads"].dropna().empty:
            raw = str(self._data_y["source_leads"].dropna().iloc[0])
            self._source_leads_cache = [int(elem.strip()) for elem in raw.split(",") if elem.strip() != ""]
        else:
            self._source_leads_cache = list(range(self._h5_file.shape[1]))
        return self._source_leads_cache

    def _parse_requested_lead(self, lead):
        key = lead.strip().lower()
        if key in self.LEAD_ALIASES:
            return self.LEAD_ALIASES[key]
        try:
            return int(key)
        except ValueError as exc:
            raise AssertionError(
                "PatientSegmentRegressionDataset supports lead selections from "
                "{lead_0, lead_1, lead0, lead1, 0, 1, II, V2, all_leads}"
            ) from exc

    def transform_lead_selection(self, custom_lead_selection):
        source_leads = self._source_leads()
        lead_to_channel = {lead_id: index for index, lead_id in enumerate(source_leads)}
        requested = {elem.strip().lower() for elem in custom_lead_selection.split(",")}

        if "all_leads" in requested:
            return list(range(len(source_leads)))

        selected_channels = []
        for lead in requested:
            lead_id = self._parse_requested_lead(lead)
            if lead_id not in lead_to_channel:
                raise AssertionError(
                    f"Requested lead {lead_id} is not present in dataset source_leads={source_leads}"
                )
            selected_channels.append(lead_to_channel[lead_id])
        return sorted(set(selected_channels))

    def get_regression_target_columns(self):
        source_leads = self._source_leads()
        selected_leads = [source_leads[channel] for channel in self._custom_lead_selection]
        target_columns = [f"target_lead{lead_id}_mV" for lead_id in selected_leads]

        missing = [column for column in target_columns if column not in self._data_y.columns]
        if missing:
            raise ValueError(f"Missing regression target columns in metadata: {missing}")
        return target_columns
