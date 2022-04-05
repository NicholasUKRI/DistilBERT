import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import json
from datetime import datetime
import pandas as pd


def mean_pooling(token_embeddings, attention_mask):
    """
    Effectively averages the embeddings of tokens across the vocabulary dimension
    to calculate the vocab-weighted latent representations (embeddings).

    :param token_embeddings: torch.float tensor of size (n_examples, n_vocab, n_latent)
    :param attention_mask: torch.byte tensor of size (n_examples, n_vocab)
    :return: torch.float tensor of size (n_examples, n_latent)
    """

    # return torch.mean(token_embeddings, dim=1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed(query, tokenizer, model):
    """
    Embed `query` using `model` and return it.

    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    """
    token = tokenizer([query], return_tensors='pt', truncation=True, padding=True)
    query_embedding = model(**token, output_hidden_states=True).hidden_states[-1]
    ## use pooling across vocab size
    query_embedding = mean_pooling(query_embedding, token['attention_mask'])
    return query_embedding


def cosine_similarity(v, M):
    """
    L2 similarity between a vector (single query embedding) and a matrix (of embeddings).

    :param v: torch.tensor of size (1, n_latent)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: torch.tensor of size (n_documents,)
    """
    dv = v.norm(p=2)
    dM = M.norm(p=2, dim=1)
    return (M.matmul(v.T)).squeeze().div(dM * dv)


def L2_similarity(v, M):
    """
    Calculate the L2 / Euclidean distance between all rows in M and v.
    The L2 norm is calculated as the square root of the sum of the squared vector values.
    
    :param v: torch.tensor of size (1, n_latent)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: torch.tensor of size (n_documents,)
    """
    return -torch.cdist(M, v).squeeze()


def return_ranked(query, tokenizer, model, M):
    """
    Embed a `query` using `model` and `tokenizer`, and return the
    indices of document embeddings `M` sorted most to least similar.

    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: a list of ints of length `n_documents`
    """
    q = embed(query, tokenizer, model)
    sims = cosine_similarity(q, M)
    rankings = torch.argsort(sims, descending=True)
    sims = sims[rankings].tolist()
    ranks = rankings.tolist()
    return list(zip(ranks, sims))


def return_ranked_by_sentence(query, tokenizer, model, indices, M):
    """
    Embed a `query` using `model` and `tokenizer`, and return the
    indices of sentence embeddings `M` sorted by most to least similar.

    :param query: str of query
    :param tokenizer: HuggingFace tokenizer instance
    :param model: HuggingFace model instance
    :param indices: torch.tensor of size (n_sentences)
    :param M: torch.tensor of size (n_documents, n_latent)
    :return: a list of ints of length `n_documents`
    """
    q = embed(query, tokenizer, model)
    sims = cosine_similarity(q, M)
    rankings = torch.argsort(sims, descending=True)
    doc_rankings = indices[rankings].numpy()
    _, first_doc_rankings = np.unique(doc_rankings, return_index=True)
    final_doc_rankings = doc_rankings[first_doc_rankings]
    matching_sims = sims[doc_rankings][first_doc_rankings]
    ranks = final_doc_rankings.tolist()
    matching_sims = matching_sims.tolist()
    return list(zip(ranks, matching_sims))


# Loads the vector embeddings
def load_embeddings(embeddings_path):
    m = torch.load(embeddings_path)
    return m


# Loads the tokenizer and masking model
def load_model(path_or_name):
    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForMaskedLM.from_pretrained(path_or_name)
    return tokenizer, model


def run_tool(query, min_words):
    # Loads json file containing UKRI grants
    f = open("data\\metadata.json")
    metadata = json.load(f)

    # Loads the vector embeddings
    embeddings = load_embeddings("data\\distilbert3tensor.pt")

    # Loads the tokenizer and masking model
    tokenizer, model = load_model("model\\distilbert3")

    # Fetch results
    results = return_ranked(query, tokenizer, model, embeddings)

    # Subset to min words
    results = [r for r in results if len(metadata[str(r[0])]['abstract'].split()) > min_words]

    # Return data as json file
    meta = [{"reference": metadata[str(i)]["project_reference"],
             "title": metadata[str(i)]["project_title"],
             "abstract": metadata[str(i)]["abstract"],
             "value": int(metadata[str(i)]["value"]),
             "n_words": len(metadata[str(i)]["abstract"].split()),
             "start_year": int(datetime.strptime(metadata[str(i)]['start_date'],
                                                 "%d/%m/%Y").year),
             "end_year": int(datetime.strptime(metadata[str(i)]['end_date'],
                                               "%d/%m/%Y").year),
             "distance": dist}
            for i, dist in results[:-1]]

    # Turn json file into dataframe
    df = pd.DataFrame(meta)

    # Reference column is a list of grants. This will separate those grants out
    df = df.explode('reference')

    return df
