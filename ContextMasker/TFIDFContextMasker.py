import json
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import DatasetDict, concatenate_datasets

from data_utils.hf_utils import (
    load_dataset_with_split_map,
    vfuse_cols,
    keep_cols,
    hstack_cols,
    get_tokenizer_encoder,
)


class TFIDFContextMasker(object):
    def __init__(
        self,
        corpus,
        tokenizer=None,
        lowercase=False,  # leave it up to tokenizer by default
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 1),
    ):

        self.tokenizer = tokenizer

        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenizer.tokenize,
            lowercase=lowercase,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
        )

        self.vectorizer.fit(corpus)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        corpus_cols,
        use_splits="train",
        tokenizer="bert-base-uncased",
        lowercase=False,  # leave it up to tokenizer by default
        max_df=1.0,
        min_df=1,
        ngram_range=(1, 1),
    ):

        context_corpus = keep_cols(dataset, cols=corpus_cols)

        if isinstance(context_corpus, DatasetDict):
            if isinstance(use_splits, str):
                context_corpus = context_corpus[use_splits]
            else:
                context_corpus = concatenate_datasets(
                    [context_corpus[i] for i in use_splits]
                )

        if isinstance(corpus_cols, str):
            context_corpus_col = corpus_cols
        else:
            context_corpus_col = "context_corpus"

        context_corpus = hstack_cols(
            context_corpus,
            cols=corpus_cols,
            stacked_col_name=context_corpus_col,
        )

        if isinstance(tokenizer, str):
            tokenizer, _ = get_tokenizer_encoder(encoder_model=tokenizer)

        context_masker = cls(
            context_corpus[context_corpus_col],
            tokenizer=tokenizer,
            lowercase=lowercase,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
        )

        return context_masker

    def save(self, path="data/TFIDFContextMasker.jb"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def load(self, path="data/TFIDFContextMasker.jb"):
        return joblib.load(path)

    def insert_tfidf_token_ids(
        self,
        dataset,
        cols=[],
        percentile_cutoff=50,
        vfuse=True,
        del_tok_ids_cols=True,
    ):

        if isinstance(cols, str):
            cols = [cols]

        feat_names = self.vectorizer.get_feature_names_out()

        def get_token_ids(ex, col, cutoff=50):
            vec = self.vectorizer.transform([ex[col]]).toarray()[0]
            nz_feat_ids = np.where(vec)

            selected_feat_ids = np.where(vec >= np.percentile(vec[nz_feat_ids], cutoff))

            selected_feat_names = feat_names[selected_feat_ids]
            ex[f"{col}_tfidf_tok_ids"] = self.tokenizer.convert_tokens_to_ids(
                selected_feat_names
            )

            return ex

        if cols and isinstance(percentile_cutoff, (int, float)):
            percentile_cutoff = [percentile_cutoff] * len(cols)

        for c, cutoff in zip(cols, percentile_cutoff):
            dataset = dataset.map(partial(get_token_ids, col=c, cutoff=cutoff))

        if len(cols) == 1:
            dataset = dataset.rename_column(f"{cols[0]}_tfidf_tok_ids", "tfidf_tok_ids")

        elif vfuse:
            cols_to_vfuse = [f"{c}_tfidf_tok_ids" for c in cols]

            def combiner_fn(ex, cols):
                return list(set.union(*[set(ex[c]) for c in cols]))

            dataset = vfuse_cols(
                dataset,
                cols=cols_to_vfuse,
                fused_col_name="tfidf_tok_ids",
                combiner_fn=combiner_fn,
                del_orig_cols=del_tok_ids_cols,
            )

        return dataset

    def insert_context_mask(self, dataset, cols, percentile_cutoff=50):
        def mask_fn(ex):
            ex["context_mask"] = np.isin(ex["input_ids"], ex["tfidf_tok_ids"]).astype(
                "int8"
            )
            return ex

        dataset = self.insert_tfidf_token_ids(dataset, cols, percentile_cutoff)
        dataset = dataset.map(mask_fn)
        dataset = dataset.remove_columns("tfidf_tok_ids")
        return dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="TFIDF based context masker")

    parser.add_argument(
        "--dataset_name",
        default="super_glue:boolq",
        help="huggingface compatible dataset name to be loaded with its config "
        "name delimited using ':' in case config name is required",
    )

    parser.add_argument(
        "--corpus_cols",
        nargs="+",
        default=["passage", "question"],
        help="columns names in the dataset to form corpus text for TFIDF masker",
    )

    parser.add_argument(
        "--tokenizer",
        default="bert-base-uncased",
        help="model tokenizer to be used",
    )

    parser.add_argument(
        "--split_map",
        type=json.loads,
        default={"train": "train", "validation": "validation", "test": "test"},
        help="Split maps for the dataset splits. Split merging can be specified as "
        'lists e.g. {"train": ["train", "validation"], "test": "test"} will merge '
        "train and validation into a single train split.",
    )

    parser.add_argument(
        "--use_splits",
        nargs="+",
        default=["train"],
        help="dataset splits to be used as context corpus",
    )

    parser.add_argument(
        "--save_path",
        default="data/TFIDFContextMasker.jb",
        help="Path of a context masker to be saved",
    )

    parser.add_argument(
        "--init_kwargs",
        default={},
        help="Initiiazation kwargs for context masker",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name.split(":")
    config_name = dataset_name[1] if len(dataset_name) == 2 else None
    dataset_name = dataset_name[0]

    if len(args.corpus_cols) == 1:
        args.corpus_cols = args.corpus_cols[0]
    if len(args.use_splits) == 1:
        args.use_splits = args.use_splits[0]

    dataset = load_dataset_with_split_map(
        dataset_name,
        config_name=config_name,
        split_map=args.split_map,
    )

    context_masker = TFIDFContextMasker.from_dataset(
        dataset,
        args.corpus_cols,
        use_splits=args.use_splits,
        tokenizer=args.tokenizer,
        **args.init_kwargs,
    )

    context_masker.save(args.save_path)
