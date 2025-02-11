from transformers import AutoTokenizer, AutoModel
from .AbstractDenseModel import AbstractDenseModel
from sentence_transformers import SentenceTransformer
import torch.nn as nn


class generic_tctcolbert(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)


class docs_tctcolbert(generic_tctcolbert):

    def __init__(self, model_name):
        super().__init__(model_name)

    def tokenize(self, input_text):
        inps = self.tokenizer([f'[CLS] [D] {d}' for d in input_text], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True,
                              max_length=512)

        return inps

    def forward(self, inps):
        inps = {k: v for k, v in inps.items()}
        res = self.model(**inps).last_hidden_state
        res = res[:, 4:, :]  # remove the first 4 tokens (representing [CLS] [ D ])
        res = res * inps['attention_mask'][:, 4:].unsqueeze(2)  # apply attention mask
        lens = inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1)
        lens[lens == 0] = 1  # avoid edge case of div0 errors
        res = res.sum(dim=1) / lens  # average based on dim
        # print(res.cpu().numpy())
        # print(res)
        return {"sentence_embedding": res}


class queries_tctcolbert(generic_tctcolbert):

    def __init__(self, model_name):
        super().__init__(model_name)

    def tokenize(self, input_text):
        inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in input_text], add_special_tokens=False, return_tensors='pt',
                              padding=True, truncation=True, max_length=36)
        #inps = self.tokenizer([q for q in input_text], add_special_tokens=False, return_tensors='pt',
        #                      padding=True, truncation=True, max_length=36)

        return inps

    def forward(self, inps):
        inps = {k: v for k, v in inps.items()}
        res = self.model(**inps).last_hidden_state
        res = res[:, 4:, :].mean(dim=1)  # remove the first 4 tokens (representing [CLS] [ Q ]), and average

        return {"sentence_embedding": res}


class TctColbert(AbstractDenseModel):

    def __init__(self, *args, model_hgf='castorini/tct_colbert-v2-msmarco', **kwargs):
        super().__init__(*args, **kwargs)
        self.model_hgf = model_hgf

        docs_model = docs_tctcolbert(model_hgf)

        self.docs_model = SentenceTransformer(modules=[docs_model])

        query_model = queries_tctcolbert(model_hgf)
        self.queries_model = SentenceTransformer(modules=[query_model])

        self.embeddings_dim = 768

        self.name = "tctcolbert"

    def encode_queries(self, texts):
        return self.queries_model.encode(texts)

    def encode_documents(self, texts):
        return self.docs_model.encode(texts)

    def start_multi_process_pool(self):
        return self.docs_model.start_multi_process_pool()

    def stop_multi_process_pool(self, pool):
        self.docs_model.stop_multi_process_pool(pool)

    def get_model(self):
        return self.docs_model


'''

def __init__(self, model_name='castorini/tct_colbert-msmarco', batch_size=32, text_field='text', verbose=False, device=None):
    super().__init__(batch_size, text_field, verbose)
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = torch.device(device)
    self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()


def encode_queries(self, texts, batch_size=None):
    results = []
    with torch.no_grad():
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in chunk], add_special_tokens=False, return_tensors='pt',
                                  padding=True, truncation=True, max_length=36)
            inps = {k: v.to(self.device) for k, v in inps.items()}
            res = self.model(**inps).last_hidden_state
            res = res[:, 4:, :].mean(dim=1)  # remove the first 4 tokens (representing [CLS] [ Q ]), and average
            results.append(res.cpu().numpy())
    if not results:
        return np.empty(shape=(0, 0))
    return np.concatenate(results, axis=0)


def encode_docs(self, texts, batch_size=None):
    results = []
    with torch.no_grad():
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer([f'[CLS] [D] {d}' for d in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True,
                                  max_length=512)
            inps = {k: v.to(self.device) for k, v in inps.items()}
            res = self.model(**inps).last_hidden_state
            res = res[:, 4:, :]  # remove the first 4 tokens (representing [CLS] [ D ])
            res = res * inps['attention_mask'][:, 4:].unsqueeze(2)  # apply attention mask
            lens = inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1)
            lens[lens == 0] = 1  # avoid edge case of div0 errors
            res = res.sum(dim=1) / lens  # average based on dim
            results.append(res.cpu().numpy())
    if not results:
        return np.empty(shape=(0, 0))
    return np.concatenate(results, axis=0)
'''
