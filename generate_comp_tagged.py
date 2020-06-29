import AdvancedModel
from chu_liu_edmonds import decode_mst
import torch
from tqdm import tqdm
import numpy as np
from AdvancedModel import LSTMEncoder
torch.manual_seed(0)


def generate_comp_tagged_file(model_path, target_path, path_comp):
    """
    Generates a tagged version of the competition file.
    Args:
        model_path (str): The path to the chosen model's pth file.
        target_path (str): The path for the tagged version's file.
    """
    torch.manual_seed(0)
    words_dict, pos_dict = AdvancedModel.get_vocabs(trained_on_path_list)
    comp = AdvancedModel.DependencyDataset(words_dict, pos_dict, path_comp, padding=True, competition=True)
    comp_data_loader = AdvancedModel.DataLoader(comp, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        model = torch.load(model_path)

        if torch.cuda.is_available():
            model.cuda()

        model = model.eval()

        trees = []

        for input_data in tqdm(comp_data_loader):
            words_idx_tensor, pos_idx_tensor, headers_idx_tensor, sentence_length = input_data
            max_length = max(sentence_length)
            score_matrix = model(words_idx_tensor, pos_idx_tensor, max_length, _evaluate=True)

            trees.append(decode_mst(np.array(score_matrix[:, 0].detach().cpu()).reshape((max_length, max_length))
                                    [:sentence_length[0], :sentence_length[0]], sentence_length[0], has_labels=False)[0])

        current_tree_idx = 0
        current_word_idx = 1

        # with open(target_path, 'w') as f_labeled:
        #     pass

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
    path_comp = "Data/comp.unlabeled"
    trained_on_path_list = [path_train, path_test]
    path_comp_m1_labeled = 'comp_m1_203933551.labeled'
    path_comp_m2_labeled = 'comp_m2_203933551.labeled'
    basic_model_path = 'encoder06_29_2020_19_19_58.pth'
    advanced_model_path = 'encoder06_30_2020_01_22_43.pth'

    # generate_comp_tagged_file(basic_model_path, path_comp_m1_labeled, path_comp)
    generate_comp_tagged_file(advanced_model_path, path_comp_m2_labeled, path_comp)

    print("Evaluate end")

    # count = 0
    # count_words = 0
    # with open(path_comp_m2_labeled, 'r') as f1:
    #     with open(path_test, 'r') as f2:
    #         for line1, line2 in zip(f1, f2):
    #             split_line1 = line1.split('\t')
    #             split_line2 = line2.split('\t')
    #
    #             if len(split_line1) == 1:  # the end of a sentence denotes by \n line.
    #                 continue
    #             if split_line2[6] == split_line1[6]:
    #                 count = count+1
    #             count_words += 1
    #
    # print(count/count_words)