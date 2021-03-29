import json
import logging
import math
import os
import zipfile
from collections import Counter

from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.builders.sensetimeVAD.dataset import SensetimeVADDataset

from mmf.utils.general import get_mmf_root


logger = logging.getLogger(__name__)


@registry.register_builder("sensetimeVAD")
class CLEVRBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("sensetimeVAD")
        self.dataset_class = SensetimeVADDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/sensetimeVAD/defaults.yaml"

    def build(self, config, dataset_type):
        pass

    def load(self, config, dataset_type, *args, **kwargs):
        self.dataset = SensetimeVADDataset(config, dataset_type)
        return self.dataset

