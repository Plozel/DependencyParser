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
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
ROOT_TOKEN = "<root>"
SPECIAL_TOKENS = [ROOT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]

torch.manual_seed(1)


def OpTyNLLLOSS(true_headers, score_matrix, max_len):
    """
        A customize NLLLOSS loss used by a dependency parser, based on known headers and a matrix score.
    Args:
        true_headers (list of int tensors): The true headers for the given batch.
        score_matrix (float tensor): A matrix score given by our model - represent the header-modifier index pair probabilities.
        max_len: The maximum sentence length in the batch.
    Returns:
        Loss score(float tensor).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    score_per_batch = [] # split the score matrix by batches.
    for i in range(len(true_headers)):
        score_per_batch.append(F.log_softmax(score_matrix[:, i].view(max_len, max_len), dim=0))

    _loss = torch.tensor(0, dtype=torch.float).to(device)
    for j in range(len(true_headers)):
        for i, head in enumerate(true_headers[j]):
            _loss = _loss.add(score_per_batch[j][head][i]/len(true_headers[j]))

    return -1*_loss


def get_vocabs(list_of_paths):
    """
    Creates a POS-tags and words vocabulary dictionaries
    Args:
        list_of_paths (list of string): Contains the files' paths from which we retrieve our data.
    Returns:
         A POS and words indexes dictionaries.
    """

    words_dict = {PAD_TOKEN, ROOT_TOKEN, UNKNOWN_TOKEN}
    pos_dict = {PAD_TOKEN, ROOT_TOKEN, UNKNOWN_TOKEN}
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                split_line = line.split('\t')
                if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                    continue
                word, pos_tag, head = split_line[1], split_line[3], int(split_line[6])
                words_dict.add(word)
                pos_dict.add(pos_tag)

    return words_dict, pos_dict


class DataReader:
    """ Read the data from the requested file and hold it's components. """

    def __init__(self, word_dict, pos_dict, file_path):
        """
        Args:
            file_path (str): holds the path to the requested file.
            words_dict, tags_dict: a dictionary - keys:words\tags, items: counts of appearances.
        """
        self.file_path = file_path
        self.words_dict = word_dict
        self.pos_dict = pos_dict
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



    def get_num_sentences(self):
        """returns num of sentences in data."""
        return len(self.sentences)


class DependencyDataset(Dataset):
    """
    Holds version of our data as a PyTorch's Dataset object.
    """

    def __init__(self, word_dict, pos_dict, file_path, padding=False, word_embeddings=None):
        """
        Args:
            file_path (str): The path of the requested file.
            padding (bool): Gets true if padding is required.
            word_embeddings: A set of words mapping.
        """

        super().__init__()
        self.file_path = file_path
        self.data_reader = DataReader(word_dict, pos_dict, self.file_path)
        self.vocab_size = len(self.data_reader.words_dict)

        if word_embeddings:
            self.words_idx_mappings, self.idx_words_mappings, self.words_vectors = word_embeddings
        else:
            self.words_idx_mappings, self.idx_words_mappings, self.words_vectors = self.init_word_embeddings(
                self.data_reader.words_dict)
            self.pos_idx_mappings, self.idx_pos_mappings, self.pos_vectors = self.init_pos_embeddings(
                self.data_reader.pos_dict)

        self.pad_idx = self.words_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.words_idx_mappings.get(UNKNOWN_TOKEN)
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

                headers_idx_list.append(header)
                if word in words_dict:
                    words_idx_list.append(self.words_idx_mappings.get(word))
                else:
                    words_idx_list.append(self.unknown_idx)
                if pos_tag in pos_dict:
                    pos_idx_list.append(self.pos_idx_mappings.get(pos_tag))
                else:
                    pos_idx_list.append(self.unknown_idx)

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
    def __init__(self, word_emb_dim, pos_emb_dim, hidden_dim, word_vocab_size, tag_vocab_size):
        super(DependencyParser, self).__init__()
        torch.manual_seed(1)

        self.emb_dim = word_emb_dim + pos_emb_dim
        torch.manual_seed(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)

        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)

        torch.manual_seed(1)

        self.tag_embedding = nn.Embedding(tag_vocab_size, pos_emb_dim)
        torch.manual_seed(1)
        self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True,
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

        lstm_out, _ = self.encoder(embeds)

        features = []

        for i in range(lstm_out.shape[0]):
            features.append(torch.cat(
                [lstm_out[i].unsqueeze(1).repeat(1, max_length, 1),
                 lstm_out[i].repeat(max_length, 1, 1)], -1).unsqueeze(1))

        features = torch.cat(features, 1)
        torch.manual_seed(1)
        # features = self.mlp(features)
        edge_scores = self.fc1(features)
        torch.manual_seed(1)
        edge_scores = self.tanh(edge_scores)
        torch.manual_seed(1)
        edge_scores = self.fc2(edge_scores)
        return edge_scores


def get_acc(edge_scores, headers_idx_tensors, batch_size, max_length, sentence_length):
    """
    Uses Chu Liu Edmonds algorithm to infer a parse tree and calculates the current batch accuracy.
    Args:
        edge_scores:
        headers_idx_tensors:
        batch_size:
        max_length:
        sentence_length:
    Returns:
    """
    acc = 0
    trees = []
    for i in range(batch_size):
        trees.append(decode_mst(
            np.array(edge_scores[:, i].detach().cpu()).reshape((max_length, max_length))[:sentence_length[i],
            :sentence_length[i]], sentence_length[i],
            has_labels=False)[0])

    for i in range(batch_size):
        acc += torch.mean(torch.tensor(headers_idx_tensors[i].tolist() == trees[i], dtype=torch.float, requires_grad=False))
    return acc


def evaluate(model, words_dict, pos_dict, batch_size):
    """
    Args:
        words_dict:
        pos_dict:
        batch_size:
    Returns:
    """
    print("Evaluating Started")
    path_test = "Data/test.labeled"
    test = DependencyDataset(words_dict, pos_dict, path_test, padding=True)
    test_data_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    acc = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(test_data_loader):
            words_idx_tensor, pos_idx_tensor, headers_idx_tensor, sentence_length = input_data
            headers_idx_tensors = [headers[:sentence_length[i]] for i, headers in enumerate(headers_idx_tensor)]
            max_length = max(sentence_length)
            batched_scores = model(words_idx_tensor, pos_idx_tensor, max_length, sentence_length)

            acc += get_acc(batched_scores, headers_idx_tensors, batch_size, max_length, sentence_length)

        acc = acc / len(test)
    print("Evaluating Ended")
    return acc


def print_plots(accuracy_list, loss_list):
    """
    Prints two plot that describes our processes of learning through an NLLL loss function and the accuracy measure.
    Args:
        accuracy_list:
        loss_list:
    Returns:
    """
    plt.plot(accuracy_list, c="red", label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(loss_list, c="blue", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

if __name__ == '__main__':

    start_time = time.time()

    # hyper_parameters
    EPOCHS = 200
    WORD_EMBEDDING_DIM = 100
    POS_EMBEDDING_DIM = 25
    HIDDEN_DIM = 125
    BATCH_SIZE = 10
    LEARNING_RATE = 0.007

    path_train = "Data/train.labeled"
    path_test = "Data/test.labeled"
    paths_list = [path_train]

    words_dict, pos_dict = get_vocabs(paths_list)  # Gets all known vocabularies.

    # Preparing the dataset
    train = DependencyDataset(words_dict, pos_dict, path_train, padding=True)
    train_data_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


    word_vocab_size = len(words_dict)
    pos_vocab_size = len(pos_dict)

    OpTyParser = DependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, pos_vocab_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        OpTyParser.cuda()

    optimizer = optim.Adam(OpTyParser.parameters(), lr=LEARNING_RATE)

    # Training start
    print("Training Started")
    accuracy_list = []
    loss_list = []
    for epoch in range(EPOCHS):
        acc = 0  # to keep track of accuracy
        printable_loss = 0  # To keep track of the loss value
        i = 0
        for input_data in tqdm(train_data_loader):
            i += 1

            words_idx_tensor, pos_idx_tensor, headers_idx_tensor, sentence_length = input_data
            headers_idx_tensors = [headers[:sentence_length[i]] for i, headers in enumerate(headers_idx_tensor)]
            max_length = max(sentence_length)

            batched_weights = OpTyParser(words_idx_tensor, pos_idx_tensor, max_length, sentence_length)

            loss = OpTyNLLLOSS(headers_idx_tensors, batched_weights, max_length)
            loss.backward()
            optimizer.step()
            OpTyParser.zero_grad()

            printable_loss += loss.item()

            acc += get_acc(batched_weights, headers_idx_tensors, BATCH_SIZE, max_length, sentence_length)

        printable_loss = printable_loss / len(train)
        acc = acc/len(train)
        accuracy_list.append(float(acc))
        loss_list.append(float(printable_loss))
        test_acc = evaluate(OpTyParser, words_dict, pos_dict, BATCH_SIZE)
        e_interval = i
        print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
                                                                                      np.mean(loss_list[-e_interval:]),
                                                                                      np.mean(
                                                                                          accuracy_list[-e_interval:]),
                                                                                      test_acc))

    print_plots(accuracy_list, loss_list)
    end_time = time.time()
    torch.save(OpTyParser.state_dict(), 'OpTyParser{}.pkl '.format(start_time))
    print("the training took: ", end_time - start_time)