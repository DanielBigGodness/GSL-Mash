import json
import pickle
import string
from collections import Counter
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from transformers import BertModel, BertTokenizer, BertConfig
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from contractions import contractions_dict
import pandas as pd


def split_list_average_n(origin_list, n):
    for i in range(0, len(origin_list), n):
        yield origin_list[i:i + n]


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def text_processes(text: str) -> List[str]:
    """
    process the raw text
    """
    tokens = word_tokenize(text)
    # filter punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # filter stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # replace abbreviation
    tokens = [contractions_dict[token] if token in contractions_dict.keys() else token for token in tokens]
    # lemmatization
    wnl = WordNetLemmatizer()
    tags = pos_tag(tokens)
    res = []
    for t in tags:
        wordnet_pos = get_wordnet_pos(t[1]) or wordnet.NOUN
        res.append(wnl.lemmatize(t[0], pos=wordnet_pos))
    return res


def delete_long_text(words: List[str], length: int):
    if len(words) > length:
        return words[:length]
    return words


class DataReader:
    def __init__(self, api_path: str, mashup_path: str):
        self.api_path = api_path
        self.mashup_path = mashup_path
        self.apis = json.load(open(api_path, 'r', encoding='utf-8'))
        self.mashups = json.load(open(mashup_path, 'r', encoding='utf-8'))
        self.glove_path = '../../data/glove/glove_300d.txt'

    def clean(self, is_save: bool = False):
        r"""
        Delete mashups and APIs that contain empty fields and mashups that have used uncatalogued APIs.

        Args:
            is_save: If true, will save the processed data to the original file.
        """

        apis = []
        mashups = []
        for item in self.apis:
            if item is None or item['title'] == '' or len(item['tags']) == 0 or item['description'] == '':
                continue
            apis.append(item)
        api_titles = [item['title'] for item in apis]
        for item in self.mashups:
            if item is None or item['title'] == '' or item['description'] == '':
                continue
            # delete mashups that have used uncatalogued APIs
            is_legal = 1
            for api in item['related_apis']:
                if api is None or api['title'] not in api_titles:
                    is_legal = 0
                    break
            if is_legal:
                mashups.append(item)
        self.apis = apis
        self.mashups = mashups
        if is_save:
            json.dump(apis, open(self.api_path, 'w', encoding='utf-8'))
            json.dump(mashups, open(self.mashup_path, 'w', encoding='utf-8'))

    def get_mashups(self):
        r"""
        Get mashups.

        """
        return self.mashups

    def get_apis(self, is_total: bool = False):
        r"""
        Get apis.

        Args:
            is_total (bool): If set true, will return the full amount of apis. Else, will return the partial amount of
            APIs without unused APIs. (default: :obj:`False`)
        """
        if is_total:
            return self.apis
        used_apis = []
        for mashup in self.mashups:
            used_apis.extend([item['title'] for item in mashup['related_apis']])
        used_apis = set(used_apis)
        apis = []
        for item in self.apis:
            if item['title'] in used_apis:
                apis.append(item)
        return apis


    def get_mashup_embeddings(self, model: str = 'BERT', num_token: int = 72) -> Tuple[Tensor, Tensor]:
        r"""
        Get the word embeddings of mashups.

        Args:
            model (str): The name of pre-trained language model. The options available are "BERT", "GloVe"
            num_token (int): The number of tokens of each text.

        Returns:
            Word embeddings of mashup, including word-based embeddings and text-based embeddings.
        """
        descriptions = [item['description'] for item in self.mashups]
        if model.lower() == 'glove':
            wv = KeyedVectors.load_word2vec_format(self.glove_path)
            descriptions = [text_processes(item) for item in descriptions]
            embeddings = []
            for des in descriptions:
                embeddings = []
                for word in delete_long_text(des, num_token):
                    try:
                        vector = wv[word]
                        vector = np.expand_dims(vector, axis=0)
                    except KeyError as e:
                        vector = np.zeros(shape=(1, 300), dtype=np.float32)
                    embeddings.append(vector)
                if len(embeddings) == 0:
                    embeddings.append(np.zeros(shape=(1, 300), dtype=np.float32))
                embeddings = np.concatenate(embeddings, axis=0)
                if embeddings.shape[0] < num_token:
                    embeddings = np.pad(embeddings, ((0, num_token - embeddings.shape[0]), (0, 0)),
                                        'constant', constant_values=(0.0, 0.0))
                embeddings.append(np.expand_dims(embeddings, axis=0))
            embeddings = np.concatenate(embeddings, axis=0)
        elif model.lower() == 'bert':
            model_name = 'bert-base-uncased'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            embeddings = []
            for des in descriptions:
                input = tokenizer(des, truncation=True, padding='max_length', max_length=num_token, return_tensors='pt')
                outputs = model(**input)
                embeddings.append(outputs.last_hidden_state.detach().numpy())
            embeddings = np.concatenate(embeddings, axis=0)
        else:
            raise ValueError('Illegal pre-trained model')

        return embeddings, embeddings.mean(axis=1)

    def get_api_embeddings(self, model: str = 'BERT', num_token: int = 72, is_total: bool = False) -> Tuple[
        Tensor, Tensor]:
        r"""
        Get the word embeddings of apis.

        Args:
            model (str): The name of pre-trained language model. The options available are "BERT", "GloVe".
            num_token (int): the number of tokens of each text.
            is_total (bool): If True, will return the full amount of data. Else, will return the partial amount of data
            without unused APIs. (default: :obj:`False`)

        Returns:
            Word embeddings of mashup, including word-based embeddings and text-based embeddings.
        """
        apis = self.get_apis(is_total)
        descriptions = [item['description'] for item in apis]
        if model.lower() == 'glove':
            wv = KeyedVectors.load_word2vec_format(self.glove_path)
            descriptions = [text_processes(item) for item in descriptions]
            embeddings = []
            for des in descriptions:
                embeddings = []
                for word in delete_long_text(des, num_token):
                    try:
                        vector = wv[word]
                        vector = np.expand_dims(vector, axis=0)
                    except KeyError as e:
                        vector = np.zeros(shape=(1, 300), dtype=np.float32)
                    embeddings.append(vector)
                if len(embeddings) == 0:
                    embeddings.append(np.zeros(shape=(1, 300), dtype=np.float32))
                embeddings = np.concatenate(embeddings, axis=0)
                if embeddings.shape[0] < num_token:
                    embeddings = np.pad(embeddings, ((0, num_token - embeddings.shape[0]), (0, 0)),
                                        'constant', constant_values=(0.0, 0.0))
                embeddings.append(np.expand_dims(embeddings, axis=0))
            embeddings = np.concatenate(embeddings, axis=0)
        elif model.lower() == 'bert':
            model_name = 'bert-base-uncased'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            embeddings = []
            for des in descriptions:
                input = tokenizer(des, truncation=True, padding='max_length', max_length=num_token, return_tensors='pt')
                outputs = model(**input)
                embeddings.append(outputs.last_hidden_state.detach().numpy())
            embeddings = np.concatenate(embeddings, axis=0)
        else:
            raise ValueError('Illegal pre-trained model')

        return embeddings, embeddings.mean(axis=1)

    def get_invocation(self, is_total: bool = False) -> pd.DataFrame:
        r"""
        Get invocation between mashups and APIs.

        Args:
            is_total (bool): If True, will return the full amount of data. Else, will return the partial amount of data
            without unused APIs. (default: :obj:`False`)
        """
        apis = self.get_apis(is_total=is_total)
        apis_idx = [api['title'] for api in apis]
        related_apis = [[str(apis_idx.index(ral['title'])) for ral in m['related_apis']] for m in self.mashups]
        Xs = []
        Ys = []
        times = []
        for i in range(len(self.mashups)):
            inv = [str(i)]
            inv.extend(related_apis[i])
            Xs.append(i)
            Ys.append([int(api) for api in related_apis[i]])
            times.append(self.mashups[i]['date'].strip())
        df = pd.DataFrame({
            'index': pd.Series(range(len(self.mashups))),
            'X': pd.Series(Xs),
            'Y': pd.Series(Ys),
            'time': pd.Series(times)
        })
        return df


    def get_service_domain_embeddings(self, is_total: bool = False, service_embeddings: np.ndarray = None):
        r"""
        Get embeddings of service domain. A service domain refers to a collection of Web APIs of the same category.

        Args:
            is_total (bool): If True, will return the service domains created from the full amount of data. Else, will
            return the service domains created from the partial amount of data. (default: :obj:`False`)
            service_embeddings (List[np.ndarray]): Embeddings of service (Web API).

        """
        apis = self.get_apis(is_total)
        api_categories = [a['tags'][0] if len(a['tags']) > 0 else 'None' for a in apis]
        categories = list(set(api_categories))
        domains = [[] for _ in range(len(categories))]
        for idx, api_cate in enumerate(api_categories):
            domains[categories.index(api_cate)].append(idx)
        domain_embeddings = []
        for apis in domains:
            domain_embeddings.append(np.expand_dims(service_embeddings[apis].mean(axis=0), axis=0))
        domain_embeddings = np.concatenate(domain_embeddings, axis=0)
        return domain_embeddings


    def get_invoked_matrix(self, is_total: bool = False) -> np.ndarray:
        r"""
        Get the invoked matrix M between mashups and APIs, whose size is (num_mashup, num_api). $M_{ij}=1$ if the $i$-th
        mashup used the $j$-th API. Else, $M_{ij}=0$

        Args:
            is_total (bool): If True, will return the invoked matrix created from the full amount of data. Else, will
            return the invoked matrix created from the partial amount of data. (default: :obj:`False`)

        """
        num_mashup = len(self.mashups)
        num_api = len(self.get_apis(is_total))
        invoked_df = self.get_invocation(is_total)
        Xs = invoked_df['X'].tolist()
        Ys = invoked_df['Y'].tolist()
        invoked_matrix = np.zeros(shape=(num_mashup, num_api), dtype=np.int64)
        for x, y in zip(Xs, Ys):
            for index in y:
                invoked_matrix[x][index] = 1
        return invoked_matrix


def select_negative_samples(label: torch.Tensor, negative_sample_ratio: int = 5):
    r"""select negative samples in training stage.

    Args:
        label (List[np.ndarray]): Label indicating the APIs called by mashup.
        negative_sample_ratio (int): Ratio of negative to positive in the training stage. (default: :obj:`5`)

    Returns:
        indices of positive samples, indices of negative samples, indices of all samples, and new label.
    """
    num_candidate = label.size(0)
    positive_idx = label.nonzero(as_tuple=True)[0]
    if len(positive_idx) > 0:
        positive_idx = positive_idx.cpu().numpy()
    else:
        positive_idx = torch.tensor([0], dtype=torch.int64)
    negative_idx = np.random.choice(np.delete(np.arange(num_candidate), positive_idx),
                                    size=negative_sample_ratio * len(positive_idx), replace=False)
    sample_idx = np.concatenate((positive_idx, negative_idx), axis=None)
    label_new = torch.tensor([1] * len(positive_idx) + [0] * len(negative_idx), dtype=torch.float32)
    return positive_idx, negative_idx, sample_idx, label_new.cuda()
