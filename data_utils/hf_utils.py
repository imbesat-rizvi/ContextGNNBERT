from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModel


def keep_cols(dataset, cols=[]):

    if isinstance(cols, str):
        cols = [cols]

    def keep_fields(d):
        if cols:
            remove_cols = [c for c in d.column_names if c not in cols]
            return d.remove_columns(remove_cols)
        return d

    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict({k: keep_fields(v) for k, v in dataset.items()})
        # manually set format otherwise concatenation of datasets throws
        # column mismatch error
        dataset.set_format(columns=dataset[next(iter(dataset.keys()))].column_names)

    else:
        dataset = keep_fields(dataset)
        dataset.set_format(columns=dataset.column_names)

    return dataset


def hstack_cols(dataset, cols=[], stacked_col_name=""):

    if isinstance(cols, str):
        cols = [cols]

    if not stacked_col_name:
        stacked_col_name = "_".join(cols)

    def hstack(d):
        col_split_dataset = [
            keep_cols(d, c).rename_column(c, stacked_col_name) for c in cols
        ]
        return concatenate_datasets(col_split_dataset)

    if len(cols) > 1:
        if isinstance(dataset, DatasetDict):
            dataset = DatasetDict({k: hstack(v) for k, v in dataset.items()})
        else:
            dataset = hstack(dataset)

    return dataset


def vfuse_cols(
    dataset,
    cols=[],
    fused_col_name="",
    # default combiner_fn for cols with string type
    combiner_fn=lambda ex, cols: " ".join(ex[c] for c in cols),
    del_orig_cols=True,
):

    if not fused_col_name:
        fused_col_name = "_".join(cols)

    def vfuse(ex):
        ex[fused_col_name] = combiner_fn(ex, cols)
        return ex

    if cols:
        dataset = dataset.map(vfuse)
        if del_orig_cols:
            dataset = dataset.remove_columns(cols)

    return dataset


def get_tokenizer_encoder(
    encoder_model="bert-base-uncased",
    special_tokens=[],
):

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_model, Padding=True, Truncation=True
    )
    encoder = AutoModel.from_pretrained(encoder_model)

    if special_tokens:
        additional_tokens = {"additional_special_tokens": special_tokens}
        tokenizer.add_special_tokens(additional_tokens)
        encoder.resize_token_embeddings(len(tokenizer))

    return tokenizer, encoder


def tokenize_dataset(dataset, tokenizer, tokenizing_fields=[], **tokenizer_kwargs):

    if isinstance(tokenizing_fields, str):
        tokenizing_fields = [tokenizing_fields]

    def tokenize_fn(ex):
        return tokenizer(*[ex[i] for i in tokenizing_fields], **tokenizer_kwargs)

    return dataset.map(tokenize_fn, batched=True)
