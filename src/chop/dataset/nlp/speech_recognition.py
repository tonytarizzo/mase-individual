import logging
from pathlib import Path
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset
import torchaudio
from ..utils import add_dataset_info
import datasets as hf_datasets

logger = logging.getLogger(__name__)

LIBRISPEECH_CONFIG = {
    "sample_rate": 16000,
    "normalize_waveform": True,
    "tokenizer_checkpoint": "facebook/wav2vec2-base-960h",
    "train_size": 0.8,
    "validation_size": 0.1,
    "test_size": 0.1,
    "max_audio_length": 16 * 16000,
}

processor = Wav2Vec2Processor.from_pretrained(LIBRISPEECH_CONFIG["tokenizer_checkpoint"])

class SpeechRecognitionDatasetBase(Dataset):
    info = None  # MaseDatasetInfo

    def __init__(self, split: str, tokenizer, max_token_len: int, num_workers: int, load_from_cache_file: bool = True, auto_setup: bool = True):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        self.data = None

        if auto_setup:
            self.prepare_data()
            self.setup()

    def _download_dataset(self) -> hf_datasets.DatasetDict:
        raise NotImplementedError

    def prepare_data(self):
        self._download_dataset()

    def setup(self):
        self.data = self._download_dataset()[self.split]

    def __len__(self):
        if self.data is None:
            raise ValueError("Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset.")
        return len(self.data)

    def __getitem__(self, index):
        if self.data is None:
            raise ValueError("Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset.")
        data_row = self.data[index]
        # Implement data processing logic here
        return data_row


@add_dataset_info(
    name="condensed_librispeech_asr",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "test"),
    sequence_classification=True,  # Adjust if necessary
)
class LibrispeechASRDataset(SpeechRecognitionDatasetBase):
    def _download_dataset(self) -> hf_datasets.DatasetDict:
        dataset_dict = hf_datasets.load_dataset("nyalpatel/condensed_librispeech_asr", split=self.split)
        return dataset_dict