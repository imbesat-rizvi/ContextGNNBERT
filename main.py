from datasets import load_dataset

from data_utils.hf_utils import (
    hstack_cols, 
    get_tokenizer_encoder,
    tokenize_dataset,
)

from context_masker import TFIDFContextMasker
from models import ContextGNNBERT


dataset_name = "super_glue"
config_name = "boolq"
model_name = "bert-base-uncased"
num_labels = 2
num_gnn_blocks = 3
hidden_channels = [256,64]

cols_for_context = ["passage", "question"]
truncation_strategy = "only_first"
context_mask_fn_kwargs = dict(percentile_cutoff=(75, 50))
context_corpus_col = "tfidf_corpus"
cols_to_exl_in_model_inp = cols_for_context + ["idx"]

tokenizer, encoder = get_tokenizer_encoder(model_name=model_name)
max_tokenized_length = encoder.config.max_position_embeddings

dataset = load_dataset(dataset_name, config_name)
dataset = tokenize_dataset(
    dataset,
    tokenizer=tokenizer,
    tokenizing_fields=cols_for_context,
    truncation=truncation_strategy,
    padding="max_length",
    max_length=max_tokenized_length,
)

context_corpus = hstack_cols(
    dataset["train"], cols=cols_for_context, stacked_col_name=context_corpus_col
)

context_masker = TFIDFContextMasker(
    context_corpus[context_corpus_col], tokenizer=tokenizer
)

dataset = context_masker.insert_context_mask(
    dataset, cols=cols_for_context, **context_mask_fn_kwargs
)

dataset = dataset.remove_columns(cols_to_exl_in_model_inp)
dataset.set_format("torch")

model = ContextGNNBERT(
    encoder=encoder,
    num_labels=num_labels,
    num_gnn_blocks=num_gnn_blocks,
    hidden_channels=hidden_channels,
)

d = dataset["test"][:5]
l = d.pop("label")

for i in range(3):
    print(f"Loop = {i+1}...")
    o = model(**d)
    print(o)