from transformers import AutoTokenizer, AutoModel
from argparse import Namespace
from functools import partial
import glob
import logging
import os

from dataset_proc import SentencesProcessor, FSDataset, FSIterableDataset, pad_batch_collate
from helpers import load_json


logger = logging.getLogger(__name__)


def json_to_dataset(tokenizer, hparams, inputs=None, num_files=0, processor=None):
    idx, json_file = inputs
    logger.info("Processing %s (%i/%i)", json_file, idx + 1, num_files)

    # open current json file (which is a set of documents)
    documents, file_path = load_json(json_file)

    sources = [doc["src"] for doc in documents]
    targets = [doc["tgt"] for doc in documents if "tgt" in doc]
    labels = [doc["labels"] for doc in documents]

    processor.add_examples(
        sources,
        labels=labels,
        targets=targets if targets else None,
        overwrite_examples=True,
        overwrite_labels=True,
    )

    processor.get_features(
        tokenizer,
        bert_compatible_cls=hparams.processor_no_bert_compatible_cls,
        create_segment_ids=hparams.create_token_type_ids,
        sent_rep_token_id="cls",
        create_source=targets,  # create the source if targets were present
        max_length=(
            hparams.max_seq_length if hparams.max_seq_length else tokenizer.model_max_length
        ),
        pad_on_left=tokenizer.padding_side == "left",
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        return_type="lists",
        save_to_path=hparams.data_path,
        save_to_name=os.path.basename(file_path),
        save_as_type=hparams.data_type,
    )


hparams = Namespace(
    model_type="bert",
    model_name="bert-base-uncased",
    data_path="../data/cnn_daily/cnn_dm/json.gz",
    data_type="txt",
    dataloader_type="map",
    dataloader_num_workers=4,
    processor_no_bert_compatible_cls=True,
    create_token_type_ids="binary",
    max_seq_length=512,
)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, use_fast=True)
    sentence_proc = SentencesProcessor(name="main_processor")
    print(tokenizer.model_max_length)

    datasets = dict()
    data_types = ["train", "val", "test"]
    for data_type in data_types:
        json_files = glob.glob(os.path.join(hparams.data_path, "*" + data_type + ".*.json*"))
        if len(json_files) == 0:
            logger.error(
                "No JSON dataset files detected for %s split. Make sure the `--data_path` is correct.",
                data_type,
            )
            sys.exit(1)

        # json_files = sorted(json_files)[:3]

        num_files = len(json_files)
        for inputs in enumerate(json_files):
            json_to_dataset(
                tokenizer, hparams, inputs, num_files=num_files, processor=sentence_proc
            )

        inferred_type = "txt"
        dataset_files = glob.glob(
            os.path.join(hparams.data_path, "*" + data_type + ".*." + inferred_type)
        )
        print(dataset_files)
        # always create actual dataset, either after writing the shard  files to disk or by skipping that step (because preprocessed files detected) and going right to loading.
        if hparams.dataloader_type == "map":
            if inferred_type != "txt":
                logger.error(
                    """The `--dataloader_type` is 'map' but the `--data_type` was not
                    inferred to be 'txt'. The map-style dataloader requires 'txt' data.
                    Either set `--dataloader_type` to 'iterable' to use the old data
                    format or process the JSON to TXT by setting `--data_type` to
                    'txt'. Alternatively, you can convert directly from PT to TXT
                    using `scripts/convert_extractive_pt_to_txt.py`."""
                )
                sys.exit(1)
            datasets[data_type] = FSDataset(dataset_files, verbose=True)
        elif hparams.dataloader_type == "iterable":
            # Since `FSIterableDataset` is an `IterableDataset` the `DataLoader` will ask
            # the `Dataset` for the length instead of calculating it because the length
            # of `IterableDatasets` might not be known, but it is in this case.
            datasets[data_type] = FSIterableDataset(dataset_files, verbose=True)
            # Force use one worker if using an iterable dataset to prevent duplicate data
            hparams.dataloader_num_workers = 1

    # Create `pad_batch_collate` function
    # If the model is a longformer then create the `global_attention_mask`
    if hparams.model_type == "longformer":
        pad_batch_collate = partial(pad_batch_collate, modifier=longformer_modifier)
    else:
        # default is to just use the normal `pad_batch_collate` function
        pad_batch_collate = pad_batch_collate
