import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from collections import Counter
from chu_liu_edmonds import decode_mst
import time

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
ROOT_TOKEN = "<root>"
SPECIAL_TOKENS = [ROOT_TOKEN, PAD_TOKEN]

torch.manual_seed(1)


def my_cross_entropy(true_headers, score_matrix, max_len):
    """
        A customize CrossEntropy loss used by a dependency parser, based on known headers and a matrix score.

    Args:
        true_headers (list of int tensors): The true headers for the given batch.
        score_matrix (float tensor): A matrix score given by our model - represent the header-modifier index pair probabilities.
        max_len: The maximum sentence length in the batch.

    Returns:
        Loss score(float tensor).

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score_per_batch = []
    for i in range(len(true_headers)):
        score_per_batch.append(F.log_softmax(score_matrix[:, i].view(max_len, max_len), dim=0))

    _loss = torch.tensor(0, dtype=torch.float).to(device)
    for j in range(len(true_headers)):
        for i, head in enumerate(true_headers[j]):
            _loss = _loss.add(score_per_batch[j][head][i]/len(true_headers[j]))
    a = -1*_loss
    return a



class DataReader:
    """ Read the data from the requested file and hold it's components. """

    def __init__(self, file_path):
        """
        Args:
            file_path (str): holds the path to the requested file.
            words_dict, tags_dict: a dictionary - keys:words\tags, items: counts of appearances.
        """
        self.file_path = file_path
        self.words_dict = {PAD_TOKEN, ROOT_TOKEN}
        self.pos_dict = {PAD_TOKEN, ROOT_TOKEN}
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        cur_sentence_word = [ROOT_TOKEN]
        cur_sentence_pos = [ROOT_TOKEN]
        cur_sentence_headers = [-1]

        with open(self.file_path, 'r') as f:
            for line in f:
                split_line = line.split('\t')
                if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                    self.sentences.append((cur_sentence_word, cur_sentence_pos, cur_sentence_headers))
                    cur_sentence_word = [ROOT_TOKEN]
                    cur_sentence_pos = [ROOT_TOKEN]
                    cur_sentence_headers = [-1]
                    continue
                word, pos_tag, head = split_line[1], split_line[3], int(split_line[6])
                cur_sentence_word.append(word)
                cur_sentence_pos.append(pos_tag)
                cur_sentence_headers.append(head)
                self.words_dict.add(word)
                self.pos_dict.add(pos_tag)

    def get_num_sentences(self):
        """returns num of sentences in data."""
        return len(self.sentences)


class DependencyDataset(Dataset):
    """
    Holds version of our data as a PyTorch's Dataset object.
    """

    def __init__(self, file_path, padding=False, word_embeddings=None):
        """
        Args:
            file_path (str): The path of the requested file.
            padding (bool): Gets true if padding is required.
            word_embeddings: A set of words mapping.
        """

        super().__init__()
        self.file_path = file_path
        self.data_reader = DataReader(self.file_path)
        self.vocab_size = len(self.data_reader.words_dict)

        if word_embeddings:
            self.words_idx_mappings, self.idx_words_mappings, self.words_vectors = word_embeddings
        else:
            self.words_idx_mappings, self.idx_words_mappings, self.words_vectors = self.init_word_embeddings(
                self.data_reader.words_dict)
            self.pos_idx_mappings, self.idx_pos_mappings, self.pos_vectors = self.init_pos_embeddings(
                self.data_reader.pos_dict)

        self.pad_idx = self.words_idx_mappings.get(PAD_TOKEN)
        self.sentence_lens = [len(sentence[0]) for sentence in self.data_reader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, head_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, head_embed_idx, sentence_len

    @staticmethod
    def init_word_embeddings(words_dict):
        glove = Vocab(Counter(words_dict), vectors=None, specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    @staticmethod
    def init_pos_embeddings(pos_dict):
        glove = Vocab(Counter(pos_dict), vectors=None, specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_words_embeddings(self):
        return self.words_idx_mappings, self.idx_words_mappings, self.words_vectors

    def get_pos_embeddings(self):
        return self.pos_idx_mappings, self.idx_pos_mappings, self.pos_vectors

    def convert_sentences_to_dataset(self, padding):
        sentence_words_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_headers_idx_list = list()
        sentence_len_list = self.sentence_lens

        for sentence_idx, sentence in enumerate(self.data_reader.sentences):
            words_idx_list = []
            pos_idx_list = []
            headers_idx_list = []

            for word, pos_tag, header in zip(sentence[0], sentence[1], sentence[2]):
                words_idx_list.append(self.words_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos_tag))
                headers_idx_list.append(header)

            if padding:
                while len(words_idx_list) < self.max_seq_len:
                    words_idx_list.append(self.pad_idx)
                    pos_idx_list.append(self.pad_idx)
                    headers_idx_list.append(self.pad_idx)
                sentence_words_idx_list.append(words_idx_list)
                sentence_pos_idx_list.append(pos_idx_list)
                sentence_headers_idx_list.append(headers_idx_list)
            else:
                sentence_words_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
                sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
                sentence_headers_idx_list.append(torch.tensor(headers_idx_list, dtype=torch.long, requires_grad=False))


        if padding:
            all_sentence_words_idx = torch.tensor(sentence_words_idx_list, dtype=torch.long, requires_grad=False)
            all_sentence_tags_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long, requires_grad=False)
            all_sentence_labels_idx = torch.tensor(sentence_headers_idx_list, dtype=torch.long, requires_grad=False)
            all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
            return TensorDataset(all_sentence_words_idx, all_sentence_tags_idx, all_sentence_labels_idx,
                                 all_sentence_len)
        else:
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_words_idx_list,
                                                                         sentence_pos_idx_list,
                                                                         sentence_headers_idx_list,
                                                                         sentence_len_list))}




class DependencyParser(nn.Module):
    def __init__(self, word_emb_dim, tag_emb_dim, word_vocab_size, tag_vocab_size):
        super(DependencyParser, self).__init__()
        torch.manual_seed(1)

        self.emb_dim = word_emb_dim + tag_emb_dim
        torch.manual_seed(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)

        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)

        torch.manual_seed(1)

        self.tag_embedding = nn.Embedding(tag_vocab_size, tag_emb_dim)
        torch.manual_seed(1)
        self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=2, bidirectional=True,
                               batch_first=True)
        torch.manual_seed(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim * 4, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

        torch.manual_seed(1)
        self.fc1 = nn.Linear(self.emb_dim * 4, 100)
        torch.manual_seed(1)
        self.tanh = nn.Tanh()
        torch.manual_seed(1)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, words_idx_tensor, pos_idx_tensor, max_length, lengths):
        torch.manual_seed(1)
        words_embedded = self.word_embedding(words_idx_tensor[:, :max_length].to(self.device))

        torch.manual_seed(1)

        tags_embedded = self.tag_embedding(pos_idx_tensor[:, :max_length].to(self.device))
        torch.manual_seed(1)

        embeds = torch.cat([words_embedded, tags_embedded], 2)

        torch.manual_seed(1)

        # lstm_out, _ = self.encoder(embeds.view(embeds.shape[1], embeds.shape[0], -1))
        lstm_out, _ = self.encoder(embeds)

        torch.manual_seed(1)
        features = []
        for i in range(max_length):
            for j in range(max_length):
                feature = torch.cat([lstm_out[:, i], lstm_out[:, j]], 1)
                features.append(feature)

        features = torch.stack(features, 0)

        # features = []
        # features = torch.cat(
        #     [lstm_out.view(lstm_out.shape[1], lstm_out.shape[2]).unsqueeze(1).repeat(1, max_length, 1),
        #      lstm_out.repeat(max_length, 1, 1)], -1)

        torch.manual_seed(1)
        # features = self.mlp(features)
        edge_scores = self.fc1(features)
        torch.manual_seed(1)
        edge_scores = self.tanh(edge_scores)
        torch.manual_seed(1)
        edge_scores = self.fc2(edge_scores)
        return edge_scores


if __name__ == '__main__':

    start_time = time.time()

    # hyper_parameters
    EPOCHS = 1000
    WORD_EMBEDDING_DIM = 100
    POS_EMBEDDING_DIM = 25
    HIDDEN_DIM = 125
    BATCH_SIZE = 3

    path_train = "Data/train.labeled"

    # Preparing the dataset
    train = DependencyDataset(path_train, padding=True)
    train_data_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    word_vocab_size = len(train.data_reader.words_dict)
    pos_vocab_size = len(train.data_reader.pos_dict)

    model = DependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, word_vocab_size, pos_vocab_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training start
    print("Training Started")

    for epoch in range(EPOCHS):

        for batch_idx, input_data in enumerate(train_data_loader):

            print("batch number -----", batch_idx)

            words_idx_tensor, pos_idx_tensor, headers_idx_tensor, sentence_length = input_data
            headers_idx_tensors = [headers[:sentence_length[i]] for i, headers in enumerate(headers_idx_tensor)]

            max_length = max(sentence_length)


            batched_weights = model(words_idx_tensor, pos_idx_tensor, max_length, sentence_length)

            loss = my_cross_entropy(headers_idx_tensors, batched_weights, max_length)
            # loss = UDNLLLoss(headers_idx_tensor, batched_weights, sentence_length)
            print(loss)

            loss.backward()

            optimizer.step()
            model.zero_grad()

            # Using Chu Liu Edmonds algorithm to infer a parse tree

            weights = batched_weights

            # Using Chu Liu Edmonds algorithm to infer a parse tree
            trees = []
            for i in range(BATCH_SIZE):
                trees.append(decode_mst(np.array(weights[:, i].detach().cpu()).reshape((max_length, max_length))[:sentence_length[i], :sentence_length[i]], sentence_length[i],
                                        has_labels=False)[0])

            correct = 0

            for i in range(BATCH_SIZE):
                for j, header in enumerate(trees[i]):

                    if headers_idx_tensors[i][j] == header:

                        correct += 1

            print("correct:", correct)
            print(torch.sum(sentence_length))

            break


    end_time = time.time()
    print("the training took: ", end_time - start_time)