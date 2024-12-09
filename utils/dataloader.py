import torch
from torch.utils.data import IterableDataset, get_worker_info

import datasets
import datasets.distributed

import itertools
from loguru import logger
from transformers import AutoTokenizer


def setup_dataset(args, global_rank, world_size):
    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
    # data = datasets.load_from_disk('/fsx-storygen/yuandong/c4_processed/train')

    seed_for_shuffle = 42 
    # if args.continue_from is not None:
    #     seed_for_shuffle += int(hashlib.sha256(args.continue_from.encode("utf-8")).hexdigest(), 16)

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)
    return dataloader, tokenizer


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


