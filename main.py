from torchtext.vocab import Vocab
import random
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

PAD_TOKEN = 0
ROOT_TOKEN = 0
WORD_VEC_LEN = 2
TAG_VEC_LEN = 2


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
            return [random.uniform(0, 1) for i in range(WORD_VEC_LEN)]

        else:
            return [random.uniform(0, 1) for i in range(TAG_VEC_LEN)]

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
                    self.word_tag_dict[(word, pos_tag)] = self.words_dict[word] + self.tags_dict[pos_tag]

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
        # words_dict = self.data_reader.words_dict
        # tags_dict = self.data_reader.tags_dict
        word_tag_dict = self.data_reader.word_tag_dict

        for i, sentence in enumerate(self.sentences):
            words_part = [word_tag_dict[(token[1], token[2])] for token in sentence]
            labels_part = [(int(token[0]), int(token[3])) for token in sentence]
            while len(words_part) < self.max_seq_len:
                words_part.append([PAD_TOKEN for j in range(WORD_VEC_LEN+TAG_VEC_LEN)])
                labels_part.append((PAD_TOKEN, PAD_TOKEN))
            self.sentences[i] = [torch.tensor(words_part), torch.tensor(labels_part)]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        word_embed_idx, labels = self.sentences[idx]
        return word_embed_idx, labels


if __name__ == '__main__':
    dataset = DependencyDataset("Data/train.labeled")

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    for i, images in enumerate(train_loader):
        print(images)

        if i == 0:
            break
