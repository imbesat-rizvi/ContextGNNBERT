# ContextGNNBERT

To fit and save a ContextMasker such as TFIDFContextMasker on a large dataset, say wikipedia, so that it can be used later on repeatedly, execute:

```
python3 -m ContextMasker.TFIDFContextMasker \
    --dataset_name wikipedia:20220301.en \
    --corpus_cols text \
    --tokenizer bert-base-uncased \
    --save_path data/TFIDFContextMasker.jb
```

The context masker can then be used along with dataset specific training and evaluation on the train, validation and test set, execute:

```
python3 main.py \
    --dataset_name scitail:snli_format \
    --label_col gold_label \
    --pos_label entailment \
    --input_text_cols sentence1 sentence2 \
    --context_masker_load_path data/TFIDFContextMasker.jb \
    --context_mask_fn_kwargs '{"percentile_cutoff": 50}' \
    --truncation_strategy longest_first \
    --classifier_net GATConv \
    --num_train_epochs 50 \
    --batch_size 12 \
    --optimizer_name AdamW \
    --optimizer_kwargs '{"lr": 1e-3, "eps": 1e-8}' \
    --encoder_optimizer_kwargs '{"lr": 1e-5}'
```

To run context masker fit on the specific dataset at hand, and to train and report evaluation on the train, validation and test set, execute:

```
python3 main.py \
    --dataset_name scitail:snli_format \
    --label_col gold_label \
    --pos_label entailment \
    --input_text_cols sentence1 sentence2 \
    --context_corpus_splits train \
    --context_masker TFIDFContextMasker \
    --context_mask_fn_kwargs '{"percentile_cutoff": 50}' \
    --truncation_strategy longest_first \
    --classifier_net GATConv \
    --num_train_epochs 50 \
    --batch_size 12 \
    --optimizer_name AdamW \
    --optimizer_kwargs '{"lr": 1e-3, "eps": 1e-8}' \
    --encoder_optimizer_kwargs '{"lr": 1e-5}'
```

or to run the script without wandb experiment tracking, execute as:
```
WANDB_MODE=disabled python3 main.py ...
```