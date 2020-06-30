import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from collections import Counter, OrderedDict
from chu_liu_edmonds import decode_mst
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from timeit import default_timer as timer
import csv

torch.manual_seed(0)
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
ROOT_TOKEN = "<root>"
SPECIAL_TOKENS = [ROOT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]


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

    words_dict = OrderedDict([(PAD_TOKEN, 1), (ROOT_TOKEN, 1), (UNKNOWN_TOKEN, 1)])
    pos_dict = OrderedDict([(PAD_TOKEN, 1), (ROOT_TOKEN, 1), (UNKNOWN_TOKEN, 1)])

    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                split_line = line.split('\t')
                if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                    continue
                word, pos_tag = split_line[1], split_line[3]
                if word in words_dict:
                    words_dict[word] = words_dict[word] + 1
                else:
                    words_dict[word] = 1
                if pos_tag in pos_dict:
                    pos_dict[pos_tag] = pos_dict[pos_tag] + 1
                else:
                    pos_dict[pos_tag] = 1

    return words_dict, pos_dict


class DataReader:
    """ Reads the data from the requested file and hold it's components. """

    def __init__(self, word_dict, pos_dict, file_path, competition=False):
        """
        Args:
            file_path (str): holds the path to the requested file.
            words_dict, tags_dict: a dictionary - keys:words\tags, items: counts of appearances.
        """
        self.competition = competition
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
                if not self.competition:
                    word, pos_tag, head = split_line[1], split_line[3], int(split_line[6])
                else:
                    word, pos_tag, head = split_line[1], split_line[3], -2
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

    def __init__(self, word_dict, pos_dict, file_path, padding=False, word_embeddings=None, competition=False):

        """
        Args:
            word_dict:
            pos_dict:
            file_path: The path of the requested file.
            padding: Gets true if padding is required.
            word_embeddings (str):  A pretrained embedding path.
            competition (bool):  Gets True if it works on a file without gold headers.
        """

        super().__init__()
        self.file_path = file_path
        self.data_reader = DataReader(word_dict, pos_dict, self.file_path, competition)
        self.vocab_size = len(self.data_reader.words_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                if word in self.data_reader.words_dict:
                    words_idx_list.append(self.words_idx_mappings.get(word))
                else:
                    words_idx_list.append(self.unknown_idx)
                if pos_tag in self.data_reader.pos_dict:
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
                sentence_words_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False).to(self.device))
                sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False).to(self.device))
                sentence_headers_idx_list.append(torch.tensor(headers_idx_list, dtype=torch.long, requires_grad=False).to(self.device))

        if padding:
            all_sentence_words_idx = torch.tensor(sentence_words_idx_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            all_sentence_tags_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            all_sentence_labels_idx = torch.tensor(sentence_headers_idx_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False).to(self.device, non_blocking=True)
            return TensorDataset(all_sentence_words_idx, all_sentence_tags_idx, all_sentence_labels_idx,
                                 all_sentence_len)
        else:
            return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_words_idx_list,
                                                                         sentence_pos_idx_list,
                                                                         sentence_headers_idx_list,
                                                                         sentence_len_list))}


class LSTMEncoder(nn.Module):
    """
    Our model encoder, based on LSTM and Contrast.
    """
    def __init__(self, word_emb_dim, pos_emb_dim, hidden_dim, word_vocab_size, tag_vocab_size, _word_tag_dropout, _mlp_dropout, _lstm_dropout):
        """
        Args:
            word_emb_dim: The dimension of the word embedding.
            pos_emb_dim: The dimension of the POS tag embedding.
            hidden_dim: The dimension of the LSTM's hidden size
            word_vocab_size: The number of words in our vocabulary.
            tag_vocab_size: The number of tags in our vocabulary
            _word_tag_dropout: The probability to dropout a complete word\ag and replace it with it's matched word\tag.
            _mlp_dropout: The MLP's neuron dropout probability
            _lstm_dropout: The LSTM's layer dropout probability
        """
        super(LSTMEncoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hidden_dim = hidden_dim
        self.emb_dim = word_emb_dim + pos_emb_dim

        self._word_tag_dropout = _word_tag_dropout
        self._mlp_dropout = _mlp_dropout
        self._lstm_dropout = _lstm_dropout

        self.weight1 = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.weight1.data.uniform_(-1, 1)
        self.weight2 = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.weight2.data.uniform_(-1, 1)
        self.weight3 = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.weight3.data.uniform_(-1, 1)
        self.weight4 = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.weight4.data.uniform_(-1, 1)
        self.weight5 = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.weight5.data.uniform_(-1, 1)

        self.weight_tanh = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.weight_tanh.data.uniform_(-1, 1)
        self.weight_relu = torch.nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.weight_relu.data.uniform_(-1, 1)

        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim)
        self.tag_embedding = nn.Embedding(tag_vocab_size, pos_emb_dim)
        self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, num_layers=5, bidirectional=True,
                               batch_first=True, dropout=self._lstm_dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, 100),
            nn.Dropout(p=self._mlp_dropout),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.fc1 = nn.Linear(self.hidden_dim * 4, 100)
        self.drop_out = nn.Dropout(p=self._mlp_dropout)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(100, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def word_tag_dropout(self, words, postags, p_drop):
        """ Word\tag dropout based on DEEP BIAFFINE ATTENTION FOR NEURAL DEPENDENCY PARSING
              - Christopher D. Manning, Timothy Dozat """
        p_matrix_word = torch.rand(size=words.shape, device=words.device)
        p_matrix_pos = torch.rand(size=words.shape, device=words.device)
        w_dropout_mask = (p_matrix_word > p_drop).long()
        p_dropout_mask = (p_matrix_pos > p_drop).long()
        w_dropout_mask_inv = (p_matrix_word < p_drop).long()
        p_dropout_mask_inv = (p_matrix_pos < p_drop).long()

        return words*w_dropout_mask + postags*w_dropout_mask_inv, postags*p_dropout_mask + p_dropout_mask_inv

    def forward(self, words_idx_tensor, pos_idx_tensor, max_length, _evaluate=False):

        if not _evaluate:
            words_idx_tensor, pos_idx_tensor = self.word_tag_dropout(words_idx_tensor, pos_idx_tensor, self._word_tag_dropout)

        words_embedded = self.word_embedding(words_idx_tensor[:, :max_length].to(self.device, non_blocking=True))
        tags_embedded = self.tag_embedding(pos_idx_tensor[:, :max_length].to(self.device, non_blocking=True))
        embeds = torch.cat([words_embedded, tags_embedded], 2)
        lstm_out, _ = self.encoder(embeds)

        features = []

        for i in range(lstm_out.shape[0]):
            features.append(torch.cat(
                [lstm_out[i].unsqueeze(1).repeat(1, max_length, 1),
                 lstm_out[i].repeat(max_length, 1, 1)], -1).unsqueeze(1))

        features = torch.cat(features, 1)
        features_x2 = features * features
        features_x3 = features_x2 * features
        features_x4 = features_x3 * features
        features_x5 = features_x4 * features

        features1 = self.weight1.mul(features)
        features2 = self.weight2.mul(features_x2)
        features3 = self.weight3.mul(features_x3)
        features4 = self.weight4.mul(features_x4)
        features5 = self.weight5.mul(features_x5)

        features = features1 + features2 + features3 + features4 + features5

        edge_scores = self.mlp(features)

        return edge_scores


def get_acc(edge_scores, headers_idx_tensors, batch_size, max_length, sentence_length):
    """
    Uses Chu Liu Edmonds algorithm to infer a parse tree and calculates the current batch accuracy.
    Args:
        edge_scores: Edge scores matrix, gained our of our chosen model.
        headers_idx_tensors: The gold headers to compare to.
        batch_size: The number of sentences in a batch.
        max_length: The maximum length of a sentence in the batch.
        sentence_length: List of all the sentences length.

    Returns:
        The summed accuracy of the current batch.

    """
    acc = 0
    trees = []
    for i in range(batch_size):
        trees.append(decode_mst(
            np.array(edge_scores[:, i].detach().cpu()).reshape((max_length, max_length))[:sentence_length[i],
            :sentence_length[i]], sentence_length[i],
            has_labels=False)[0])

    for i in range(batch_size):
        acc += torch.sum(torch.tensor(headers_idx_tensors[i][1:].tolist() == trees[i][1:], dtype=torch.float, requires_grad=False))
    return acc


def evaluate(model, words_dict, pos_dict, batch_size, path_test):
    """
    Evaluate our model on a validation set.
    Args:
        model: Our trained model.
        words_dict: The word vocabulary the model trained with.
        pos_dict: The POS tag vocabulary the model trained with.
        batch_size: The number of sentences in a batch.
    Returns:
        The given model's loss and accuracy gained on the validation set.
    """
    print("Evaluating Started")

    model.eval()
    with torch.no_grad():
        test = DependencyDataset(words_dict, pos_dict, path_test, padding=True)
        test_data_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
        num_of_sentences = len(test)
        acc = 0
        num_of_words = 0

        for batch_idx, input_data in enumerate(test_data_loader):
            words_idx_tensor, pos_idx_tensor, headers_idx_tensor, sentence_length = input_data
            headers_idx_tensors = [headers[:sentence_length[i]] for i, headers in enumerate(headers_idx_tensor)]
            max_length = max(sentence_length)
            batched_scores = model(words_idx_tensor, pos_idx_tensor, max_length, _evaluate=True)

            _loss = OpTyNLLLOSS(headers_idx_tensors, batched_scores, max_length).requires_grad_(False).item()
            acc += get_acc(batched_scores, headers_idx_tensors, batch_size, max_length, sentence_length)
            num_of_words += (max_length - 1)

        acc = acc/num_of_words
        print("Evaluating Ended")
        return acc, _loss


def print_plots(train_acc_list, train_loss_list, test_acc_list, test_loss_list, _time=''):
    """
    Prints two plot that describes our processes of learning through an NLLL loss function and the accuracy measure.
    Args:
        train_acc_list: Contains the accuracy measure tracking through the training phase.
        train_loss_list: Contains the loss measure tracking through the training phase.
        test_acc_list: Contains the accuracy measure tracking through the evaluation phase.
        test_loss_list: Contains the loss measure tracking through the evaluation phase.
        _time: The time id to recognize the plot output.
    Returns:
        Saves the plot in a jpeg file.
    """

    # sns.set_style("whitegrid")

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    x_train = [a for a in range(len(train_loss_list))]
    x_test = [a for a in range(len(test_loss_list))]

    ax[0].plot(x_train, train_loss_list, label='Loss Train')
    ax[0].plot(x_test, test_loss_list, label='Loss Test')

    ax[0].legend()
    ax[0].set_title('Loss Convergence')
    ax[0].set_xlabel('Num of Epochs')
    ax[0].set_ylabel('Loss')

    ax[1].plot(x_train, train_acc_list, label='Train UAS')
    ax[1].plot(x_test, test_acc_list, label='Test UAS')
    ax[1].legend()
    ax[1].set_title('UAS')
    ax[1].set_xlabel('Num of Epochs')
    ax[1].set_ylabel('UAS')
    fig.savefig('plots_{}.png'.format(_time))


class DependencyParser:
    """
    A dependency parser model object, includes all the hyper parameters mix and train phase of the selected model.
    """
    def __init__(self, epochs, word_embedding_dim, pos_embedding_dim, hidden_dim, batch_size, batch_accumulate,
                 learning_rate, path_train, path_test, word_tag_dropout, embedding_dropout, lstm_dropout):
        """
        Args:
            epochs: The number of epochs the model is trained on.
            word_embedding_dim: The dimension of the word embedding.
            pos_embedding_dim: The dimension of the POS tag embedding.
            hidden_dim: The LSTM's hidden size dimension.
            batch_size: The batch size - the number of sentences we get out of the data loader.
            batch_accumulate: The accumulate batch size - The practical batch size, the number of  sentences  which we learn on parallel.
            learning_rate: The learning rate of our optimizer.
            path_train: The path to the train file.
            path_test: The path to the test file.
            word_tag_dropout: The probability to dropout a complete word\ag and replace it with it's matched word\tag.
            embedding_dropout: The MLP's neuron dropout probability.
            lstm_dropout: The LSTM's layer dropout probability.

        """
        self.epochs = epochs
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.batch_accumulate = batch_accumulate
        self.learning_rate = learning_rate
        self.path_train = path_train
        self.path_test = path_test
        self.word_tag_dropout = word_tag_dropout
        self.embedding_dropout = embedding_dropout
        self.lstm_dropout = lstm_dropout

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self):
        """ Runs the training phase of our model.
            Returns an accuracy tracking on a validation set and a unique ID to identify this run.
        """

        start_time = timer()
        torch.cuda.empty_cache()

        paths_list = [self.path_train]

        # Prepares the dataset.
        words_dict, pos_dict = get_vocabs(paths_list)  # Gets all known vocabularies.
        train = DependencyDataset(words_dict, pos_dict, self.path_train, padding=True)
        train_data_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        word_vocab_size = len(words_dict)
        pos_vocab_size = len(pos_dict)

        # Initialize an instance of our encoder with the chosen hyper parameters.
        encoder = LSTMEncoder(self.word_embedding_dim, self.pos_embedding_dim, self.hidden_dim, word_vocab_size,
                              pos_vocab_size, self.word_tag_dropout, self.embedding_dropout, self.lstm_dropout)

        if torch.cuda.is_available():
            encoder.cuda()

        # Initialize the chosen optimizer.
        optimizer = optim.Adam(encoder.parameters(), betas=(0.9, 0.9), lr=self.learning_rate, weight_decay=1e-5)

        # Training start
        print("Training Started")

        # To keep track of the loss and accuracy values.
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []

        for epoch in range(self.epochs):

            acc = 0
            printable_loss = 0
            i = 0 # counts the number batches we run through ( = num of sentences for batch size 1)
            num_of_words = 0

            for input_data in tqdm(train_data_loader):
                i += 1

                encoder.train()

                words_idx_tensor, pos_idx_tensor, headers_idx_tensor, sentence_length = input_data

                # In case we use batches (>1 sentences) we need to cut the padding out of the gold headers.
                headers_idx_tensors = [headers[:sentence_length[i]] for i, headers in enumerate(headers_idx_tensor)]
                max_length = max(sentence_length)

                # Feeding our model with the current batch.
                batched_weights = encoder(words_idx_tensor, pos_idx_tensor, max_length)

                loss = OpTyNLLLOSS(headers_idx_tensors, batched_weights, max_length)
                loss.backward()

                if i % self.batch_accumulate == 0:
                    optimizer.step()
                    encoder.zero_grad()

                printable_loss += loss.item()

                acc += get_acc(batched_weights, headers_idx_tensors, self.batch_size, max_length, sentence_length)
                num_of_words += (max_length - 1)  # We don't count the root as we don't count it in the accuracy.

            # Adds up the new tracking measures.
            printable_loss = printable_loss / len(train)
            acc = acc / num_of_words
            train_acc_list.append(float(acc))
            train_loss_list.append(float(printable_loss))
            # Runs a validation phase.
            test_acc, test_loss = evaluate(encoder, words_dict, pos_dict, self.batch_size, self.path_test)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            time_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            with open(r"{}_advanced_model_{}.pkl".format(epoch, time_id), "wb") as output_file:
                torch.save(encoder.state_dict(), output_file)
            print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1, train_loss_list[-1],
                                                                                          train_acc_list[-1], test_acc))

        # Saves our learned model and plot some graphs.
        time_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        print_plots(train_acc_list, train_loss_list, test_acc_list, test_loss_list, time_id)
        end_time = timer()
        with open(r"advanced_model_{}.pkl".format(time_id), "wb") as output_file:
            torch.save(encoder.state_dict(), output_file)
        print("the training took: {} sec ".format(round(end_time - start_time, 2)))
        return test_acc_list, time_id


def get_hyper_parameters():
    """Returns the hyper parameters of the model."""
    path_train = "Data/train.labeled"
    path_test = "Data/test.labeled"
    return (10, 100, 100, 500, 1, 30, 0.002, path_train, path_test, 0.3, 0.3, 0.3)


if __name__ == '__main__':

    hyper_parameters_list = [get_hyper_parameters()]

    for hyper_parameters in hyper_parameters_list:
        EPOCHS, WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, BATCH_ACCUMULATE, LEARNING_RATE, path_train, path_test, WORD_TAG_DROPOUT, EMBEDDING_DROPOUT, LSTM_DROPOUT = hyper_parameters

        parser = DependencyParser(EPOCHS, WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, BATCH_ACCUMULATE,
                                  LEARNING_RATE, path_train, path_test, WORD_TAG_DROPOUT, EMBEDDING_DROPOUT, LSTM_DROPOUT)

        test_acc_list, time_id = parser.train()
        max_test_acc = round(max(test_acc_list).item(), 3)
        epoch_max = np.argmax(test_acc_list)

        # Saves our model hyper parameters settings.
        with open('parser_results_info.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([time_id, max_test_acc, epoch_max, EPOCHS, WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM,
                             BATCH_SIZE, BATCH_ACCUMULATE, LEARNING_RATE, WORD_TAG_DROPOUT, EMBEDDING_DROPOUT, LSTM_DROPOUT])

    print("Finished training the model, based on the following hyper parameters mix: {}".format(hyper_parameters))