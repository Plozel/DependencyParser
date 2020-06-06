from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader

train_path = "Data/train.labeled"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
ROOT_TOKEN = "<root>"
WORD_VEC_LEN = 100
TAG_VEC_LEN = 25


def embedding(input_type, em_method='rand'):
    """
    embed a string to a numeric vector representation.

    Args:
        input_type (str): word/tag each with different properties.
        em_method (str): type of embedding method to be used. # TODO: right now support only random, should add word2vec/glove

    Returns:
        The embedded vector.
    """
    if em_method == 'rand':
        if input_type == 'word':
            return torch.rand(WORD_VEC_LEN)
        else:
            return torch.rand(TAG_VEC_LEN)

class DataReader:
    """ Read the data from the requested file and hold it's components. """

    def __init__(self, file_path):
        """
        Args:
            file_path (str): holds the path to the requested file.
        """

        self.file_path = file_path
        self.words_dict = {}
        self.tags_dict = {}
        self.word_tag_dict = {}
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """ main reader function. """

        cur_sentence = []

        with open(self.file_path, 'r') as f:
            for line in f:
                split_line = line.split('\t')
                if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                    self.sentences.append(cur_sentence)
                    cur_sentence = []
                    continue
                within_idx, word, pos_tag, head = (split_line[0], split_line[1], split_line[3], split_line[6])
                cur_sentence.append((within_idx, word, pos_tag, head))

                if word not in self.words_dict:
                    self.words_dict[word] = embedding('word')
                if pos_tag not in self.tags_dict:
                    self.tags_dict[pos_tag] = embedding('tag')
                if (word, pos_tag) not in self.word_tag_dict:
                    self.word_tag_dict[(word, pos_tag)] = torch.cat((self.words_dict[word], self.tags_dict[pos_tag]), 0)



    def get_num_sentences(self):
        """ returns the number of sentences in data. """
        return len(self.sentences)


class DependencyDataset(Dataset):
    """ Holds version of our data as a PyTorch's Dataset object. """

    def __init__(self, file_path: str, padding=True):
        """
        Args:
            file_path (str): The path of the requested file.
            padding (bool): Gets true if padding is required.
        """

        super().__init__()
        self.file_path = file_path
        self.data_reader = DataReader(self.file_path)
        self.sentences = self.data_reader.sentences.copy()
        self.sentence_lens = [len(sentence) for sentence in self.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.pre_processing()

    def pre_processing(self):
        words_dict = self.data_reader.words_dict
        tags_dict = self.data_reader.tags_dict
        word_tag_dict = self.data_reader.word_tag_dict
        for i in range(len(self.sentences)):
            # Using word, pos tag concat representation
            self.sentences[i] = [(word_tag_dict[(token[1], token[2])], (token[0], token[3])) for token in self.sentences[i]]
            # while len(self.sentences[i]) < self.max_seq_len:
            #     self.sentences[i].append(torch.tensor([1]))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

if __name__ == '__main__':
    dataset = DependencyDataset("Data/train.labeled")

    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)
    for i, line in enumerate(dataloader):
        print(line)
        if i == 2:
            break