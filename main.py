import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter


def get_vocabs(file_path):
    """
    Extract vocabs from given data sets. Return a words_dict and tags_dict.
    Args:
        file_path: a path to the requested file # TODO: should use a paths list

    Returns:
        words_dict, tags_dict: a dictionary - keys:words\tags, items: counts of appearances.
    """
    words_dict = defaultdict(int)
    tags_dict = defaultdict(int)
    word_tag_dict = defaultdict(int)  # TODO: redundant?

    with open(file_path) as f:
        for line in f:
            split_line = line.split('\t')
            if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                continue
            word, pos_tag = split_line[1], split_line[3]
            words_dict[word] += 1
            tags_dict[pos_tag] += 1
            word_tag_dict[(word, pos_tag)] += 1

    return words_dict, tags_dict


class PosDataReader:
    """ Read the data from the requested file and hold it's components. """

    def __init__(self, file_path, words_dict, tags_dict):
        """
        Args:
            file_path (str): holds the path to the requested file.
            words_dict, tags_dict: a dictionary - keys:words\tags, items: counts of appearances.
        """
        self.file_path = file_path
        self.words_dict = words_dict
        self.tags_dict = tags_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        cur_sentence_word_tag = []
        cur_sentence_labels = []

        with open(self.file_path, 'r') as f:
            for line in f:
                split_line = line.split('\t')
                if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                    self.sentences.append((cur_sentence_word_tag, cur_sentence_labels))
                    cur_sentence_word_tag = []
                    cur_sentence_labels = []
                    continue
                within_idx, word, pos_tag, head = (int(split_line[0]), split_line[1], split_line[3], int(split_line[6]))
                cur_sentence_word_tag.append((word, pos_tag))
                cur_sentence_labels.append((within_idx, head))

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)


UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
ROOT_TOKEN = "<root>"
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]


class PosDataset(Dataset):
    """
    Holds version of our data as a PyTorch's Dataset object.
    """
    def __init__(self, words_dict, tags_dict, file_path, padding=False, word_embeddings=None):
        """
        Args:
            file_path (str): The path of the requested file.
            padding (bool): Gets true if padding is required.
            word_embeddings: A set of words mapping.
        """

        super().__init__()
        self.file_path = file_path
        self.data_reader = PosDataReader(self.file_path, word_dict, pos_dict)
        self.vocab_size = len(self.data_reader.words_dict)
        if word_embeddings:
            self.words_idx_mappings, self.idx_words_mappings, self.words_vectors = word_embeddings
        else:
            self.words_idx_mappings, self.idx_words_mappings, self.words_vectors = self.init_word_embeddings(
                self.data_reader.words_dict)
            self.tags_idx_mappings, self.idx_tags_mappings, self.tags_vectors = self.init_tag_embeddings(
                self.data_reader.tags_dict)
        # self.labels_idx_mappings, self.idx_labels_mappings = self.init_labels_vocab(self.data_reader.tags_dict)

        self.pad_idx = self.words_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.words_idx_mappings.get(UNKNOWN_TOKEN)
        # self.word_vector_dim = self.words_vectors.size(-1)
        self.sentence_lens = [len(sentence[0]) for sentence in self.data_reader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len

    @staticmethod
    def init_word_embeddings(words_dict):
        glove = Vocab(Counter(words_dict), vectors=None, specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    @staticmethod
    def init_tag_embeddings(tags_dict):
        glove = Vocab(Counter(tags_dict), vectors=None, specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_words_embeddings(self):
        return self.words_idx_mappings, self.idx_words_mappings, self.words_vectors

    def get_tags_embeddings(self):
        return self.tags_idx_mappings, self.idx_tags_mappings, self.tags_vectors

    def convert_sentences_to_dataset(self, padding):
        sentence_word_tag_idx_list = list()
        sentence_labels_idx_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.data_reader.sentences):
            words_tags_idx_list = []
            labels_idx_list = []
            for word_tag, labels in zip(sentence[0], sentence[1]):
                words_tags_idx_list.append((self.words_idx_mappings.get(word_tag[0]),
                                            self.tags_idx_mappings.get(word_tag[1])))
                labels_idx_list.append(labels)
            sentence_len = len(words_tags_idx_list)
            if padding:
                while len(words_tags_idx_list) < self.max_seq_len:
                    words_tags_idx_list.append((self.words_idx_mappings.get(PAD_TOKEN), self.words_idx_mappings.get(PAD_TOKEN)))
                    labels_idx_list.append((self.words_idx_mappings.get(PAD_TOKEN), self.words_idx_mappings.get(PAD_TOKEN)))
            sentence_word_tag_idx_list.append(words_tags_idx_list)
            sentence_labels_idx_list.append(labels_idx_list)
            sentence_len_list.append(sentence_len)

        if padding:
            all_sentence_word_tag_idx = torch.tensor(sentence_word_tag_idx_list, dtype=torch.long)
            all_sentence_labels_idx = torch.tensor(sentence_labels_idx_list, dtype=torch.long)
            all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
            return TensorDataset(all_sentence_word_tag_idx, all_sentence_labels_idx, all_sentence_len)


if __name__ == '__main__':
    path_train = "Data/train.labeled"

    word_dict, pos_dict = get_vocabs(path_train)
    train = PosDataset(word_dict, pos_dict, path_train, padding=True)
    train_data_loader = DataLoader(train, batch_size=2, shuffle=True)
    for i, data in enumerate(train_data_loader):
        print(data)
        break
