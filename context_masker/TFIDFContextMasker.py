import numpy as np
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils.hf_utils import vfuse_cols


class TFIDFContextMasker(object):
    
    def __init__(
        self,
        corpus, 
        tokenizer=None, 
        lowercase=False, 
        max_df=1.0, 
        min_df=1, 
        ngram_range=(1,1),
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
            
            selected_feat_ids = np.where(
                vec >= np.percentile(vec[nz_feat_ids], cutoff)
            )

            selected_feat_names = feat_names[selected_feat_ids]
            ex[f"{col}_tfidf_tok_ids"] = (
                self.tokenizer.convert_tokens_to_ids(selected_feat_names)
            )

            return ex

        if cols and isinstance(percentile_cutoff, (int, float)):
            percentile_cutoff = [percentile_cutoff] * len(cols)

        for c, cutoff in zip(cols, percentile_cutoff):
            dataset = dataset.map(partial(get_token_ids, col=c, cutoff=cutoff))

        if len(cols) == 1:
            dataset = dataset.rename_column(
                f"{cols[0]}_tfidf_tok_ids", "tfidf_tok_ids"
            )
        
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
            ex["context_mask"] = np.isin(
                ex["input_ids"], ex["tfidf_tok_ids"]
            ).astype("int8")
            return ex

        dataset = self.insert_tfidf_token_ids(dataset, cols, percentile_cutoff)
        dataset = dataset.map(mask_fn)
        dataset = dataset.remove_columns("tfidf_tok_ids")
        return dataset