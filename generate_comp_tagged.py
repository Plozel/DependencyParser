import AdvancedModel
import BasicModel
from chu_liu_edmonds import decode_mst
import torch
from tqdm import tqdm
import numpy as np

torch.manual_seed(0)


def generate_comp_tagged_file(model_path, target_path, path_comp, model_module):
    """
    Generates a tagged version of the competition file.
    Args:

        model_path (str): The path to the chosen model's pth file.
        target_path (str): The path for the tagged version's file.
        path_comp: The path for the untagged version's file.
        model_module: The module to use.
    """

    with torch.no_grad():
        words_dict, pos_dict = model_module.get_vocabs(trained_on_path_list)
        word_vocab_size, pos_vocab_size = len(words_dict), len(pos_dict)
        comp = model_module.DependencyDataset(words_dict, pos_dict, path_comp, padding=True, competition=True)
        comp_data_loader = model_module.DataLoader(comp, batch_size=1, shuffle=False, num_workers=0)
        word_to_idx, idx_to_word, _ = comp.get_words_embeddings()

        if model_module == BasicModel:
            _, word_embedding_dim, pos_embedding_dim, hidden_dim, batch_size, _, _, _, _, word_tag_dropout \
                                                                = model_module.get_hyper_parameters()

            model = model_module.LSTMEncoder(batch_size, words_dict, word_to_idx, idx_to_word, word_embedding_dim,
                                             pos_embedding_dim, hidden_dim, word_vocab_size, pos_vocab_size)
        else:
            _, word_embedding_dim, pos_embedding_dim, hidden_dim, _, _, _, _, _, word_tag_dropout, embedding_dropout,\
                                                                    lstm_dropout = model_module.get_hyper_parameters()

            model = model_module.LSTMEncoder(word_embedding_dim, pos_embedding_dim, hidden_dim, word_vocab_size,
                                             pos_vocab_size, word_tag_dropout, embedding_dropout, lstm_dropout)

        model.load_state_dict(torch.load(model_path))
        model = model.eval()

        if torch.cuda.is_available():
            model.cuda()

        trees = []

        for input_data in tqdm(comp_data_loader):
            words_idx_tensor, pos_idx_tensor, headers_idx_tensor, sentence_length = input_data
            max_length = max(sentence_length)
            score_matrix = model(words_idx_tensor, pos_idx_tensor, max_length, _evaluate=True)

            trees.append(decode_mst(np.array(score_matrix[:, 0].detach().cpu()).reshape((max_length, max_length))
                                    [:sentence_length[0], :sentence_length[0]], sentence_length[0], has_labels=False)[0])

        current_tree_idx = 0
        current_word_idx = 1

        with open(path_comp, 'r') as f_unlabeled:
            with open(target_path, 'w') as f_labeled:

                for i, line in enumerate(f_unlabeled):
                    split_line = line.split('\t')

                    if len(split_line) == 1:  # the end of a sentence denotes by \n line.
                        current_word_idx = 1
                        current_tree_idx += 1
                        f_labeled.writelines(''.join(split_line))
                        continue

                    split_line[6] = str(trees[current_tree_idx][current_word_idx])
                    current_word_idx += 1

                    f_labeled.writelines('\t'.join(split_line))


if __name__ == '__main__':
    print("Starting to evaluate")
    torch.cuda.empty_cache()

    path_train = "Data/train.labeled"
    path_test = "Data/test.labeled"
    path_comp = "Data/test.labeled"
    trained_on_path_list = [path_train]
    path_comp_m1_labeled = 'comp_m1_203933551.labeled'
    path_comp_m2_labeled = 'comp_m2_203933551.labeled'
    basic_model_path = '2_basic_model_07_01_2020_01_25_11.pkl'
    advanced_model_path = '1_advanced_model_07_01_2020_01_49_49.pkl'

    generate_comp_tagged_file(basic_model_path, path_comp_m1_labeled, path_comp, BasicModel)
    generate_comp_tagged_file(advanced_model_path, path_comp_m2_labeled, path_comp, AdvancedModel)

    print("Evaluate end")
