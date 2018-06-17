
import multiprocessing
import json
import sys

from copy import copy
from decisiontree.id3_algorithm import (calculate_total_entropy,
                                        calculate_information_gain)
from decisiontree.utils import read_csv_file
from random import randint

TREE_FILE_NAME = 'id3_tree.json'
TREE_PRUNED_FILE_NAME = 'id3_pruned_tree.json'

SEPARATOR = '__'

DECISION_INDEX = 14
POSITIVE_DECISION = '>50K'
NEGATIVE_DECISION = '<=50K'
TABS_PER_LINE = 4
IFTHEN_FILE_PATH = 'ifthen.txt'
WORKER = multiprocessing.cpu_count() * 2

global id3_tree_global
global headers_data


def _get_attribute_with_max_information_gain(
        attributtes_information_gain_dict
):
    max_information_gain = 0
    attribute_with_max_information_gain = None
    for attribute, information_gain in attributtes_information_gain_dict.items():  # noqa
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            attribute_with_max_information_gain = attribute
    return attribute_with_max_information_gain, max_information_gain


def _update_id3_tree_with_attribute(
        attribute_variations, selected_attribute, id3_tree
):
    for variation in attribute_variations:
        id3_tree_key = _get_id3_tree_key(selected_attribute, variation)
        id3_tree[id3_tree_key] = {}
    return id3_tree


def _get_attribute_variations(attribute_index, lines):
    attribute_variations = set(
        [variation[attribute_index] for variation in lines]
    )
    return attribute_variations


def _get_id3_tree_key(selected_attribute, variation):
    id3_tree_key = selected_attribute + SEPARATOR + variation
    return id3_tree_key


def train_decision_tree(attributes, lines, entropy, id3_tree={}):
    attributtes_information_gain_dict = calculate_information_gain(
        attributes, entropy, lines, POSITIVE_DECISION, DECISION_INDEX
    )
    selected_attribute, max_information_gain = _get_attribute_with_max_information_gain(
        attributtes_information_gain_dict
    )

    if not selected_attribute:
        return lines[0][DECISION_INDEX]

    attribute_index = attributes.index(selected_attribute)
    attribute_variations = _get_attribute_variations(attribute_index, lines)
    id3_tree = _update_id3_tree_with_attribute(
        attribute_variations, selected_attribute, id3_tree
    )
    variations_lines_dict = _create_variation_dict(attribute_variations)
    while len(lines):
        line = lines.pop()
        variations_lines_dict[line[attribute_index]].append(line)
    for attribute_variation, variations_lines in variations_lines_dict.items():
        variation_entropy = calculate_total_entropy(
            variations_lines, POSITIVE_DECISION,
            NEGATIVE_DECISION, DECISION_INDEX
        )
        id3_tree_key = _get_id3_tree_key(
            selected_attribute, attribute_variation
        )
        id3_tree[id3_tree_key] = train_decision_tree(
            attributes, variations_lines,
            variation_entropy, id3_tree[id3_tree_key]
        )
    return id3_tree


def _create_variation_dict(attribute_variations):
    attribute_variations_dict = {}
    for attribute_variation in attribute_variations:
        attribute_variations_dict[attribute_variation] = []
    return attribute_variations_dict


def save_json_to_file(id3_tree, file_path):
    with open(file_path, 'w') as id3_file_path:
        id3_file_path.write(json.dumps(id3_tree))


def read_id3_tree():
    with open(TREE_FILE_NAME, 'r') as id3_file:
        return json.loads(id3_file.read())


def read_id3_tree_pruned():
    with open(TREE_PRUNED_FILE_NAME, 'r') as id3_file:
        return json.loads(id3_file.read())


def get_decision_attribute(id3_tree):
    return list(id3_tree.keys())[0].split(SEPARATOR)[0]


id3_key_regexes = {}


def get_id3_key(id3_tree, decision_attribute, decision_attribute_value):
    key = decision_attribute + SEPARATOR + decision_attribute_value
    keys_list = list(id3_tree.keys())
    if key in keys_list:
        return key
    else:
        #print('Key not found for {} and {}'.format(
        #    decision_attribute, decision_attribute_value
        #))
        return None


def test_data(id3_tree, headers, data):
    if isinstance(id3_tree, str):
        return data[DECISION_INDEX] == id3_tree
    decision_attribute = get_decision_attribute(id3_tree)
    decision_attribute_index = headers.index(decision_attribute)
    decision_attribute_value = data[decision_attribute_index]

    id3_key = get_id3_key(
        id3_tree, decision_attribute,
        decision_attribute_value
    )
    if id3_key is None:
        return False
    return test_data(id3_tree[id3_key], headers, data)


def separate_folds(file_data, fold_number):
    folds = [[] for x in range(fold_number)]
    count = 0
    while len(file_data) > 0:
        i = randint(0, (len(file_data) - 1))
        folds[count].append(file_data.pop(i))
        count += 1
        if count > (fold_number - 1):
            count = 0
    return folds


def calculate_accuracy(id3_tree, file_data, headers):
    success = 0
    errors = 0
    for line in file_data:
        tested_data = test_data(id3_tree, headers, line)
        if tested_data:
            success += 1
        else:
            errors += 1
    return success / (success + errors)


def prune_tree(id3_tree, id3_tree_part, file_data_test, headers, file_data):
    for key, value in id3_tree_part.items():
        if value not in [NEGATIVE_DECISION, POSITIVE_DECISION]:
            old_accuracy = calculate_accuracy(
                id3_tree, file_data_test, headers
            )
            branch_aux = copy(value)

            id3_tree_part[key] = POSITIVE_DECISION
            positive_accuracy = calculate_accuracy(
                id3_tree, file_data_test, headers
            )

            id3_tree_part[key] = NEGATIVE_DECISION
            negative_accuracy = calculate_accuracy(
                id3_tree, file_data_test, headers
            )

            if (old_accuracy >= negative_accuracy
               and old_accuracy >= positive_accuracy):
                id3_tree_part[key] = branch_aux

            else:
                new_accuracy = calculate_accuracy(
                    id3_tree, file_data, headers
                )
                print('accuracy train: {} - accuracy test: {}'.format(
                        new_accuracy, max(
                            positive_accuracy, negative_accuracy
                        )
                    )
                )
                if positive_accuracy > negative_accuracy:
                    id3_tree_part[key] = POSITIVE_DECISION
                else:
                    id3_tree_part[key] = NEGATIVE_DECISION

            prune_tree(id3_tree, value, file_data_test, headers, file_data)
    return id3_tree


def _get_key_with_most_accuracy(accuracy_dict):
    max_accuracy = None
    max_accuracy_key = None
    for tree_key, tree_accuracy in accuracy_dict.items():
        if max_accuracy is None or tree_accuracy > max_accuracy:
            max_accuracy = tree_accuracy
            max_accuracy_key = tree_key
    return max_accuracy_key


def _create_if_then_tree_key(tree_key, then_nodes=False):
    if then_nodes:
        return 'THEN {}\n'.format(tree_key)
    return 'IF {} == {}\n'.format(*tree_key.split(SEPARATOR))


def _get_lines_that_match_key(headers, tree_key, file_data):
    key, value = tree_key.split(SEPARATOR)
    attribute_index = headers.index(key)
    subtree_data = []
    for line in file_data:
        if line[attribute_index] == value:
            subtree_data.append(line)
    return subtree_data


def convert_to_if_then(id3_tree, file_data, headers, tabs_prefix=0):
    tabs_before = ' ' * TABS_PER_LINE * tabs_prefix
    if isinstance(id3_tree, str):
        return (tabs_before
                + _create_if_then_tree_key(id3_tree, then_nodes=True))
    if_then = ''
    keys_list = list(id3_tree.keys())
    accuracy_dict = {}
    for key in keys_list:
        accuracy = calculate_accuracy(id3_tree[key], file_data, headers)
        accuracy_dict[key] = accuracy
    while accuracy_dict:
        tree_key = _get_key_with_most_accuracy(accuracy_dict)
        del accuracy_dict[tree_key]
        if_then += tabs_before + _create_if_then_tree_key(tree_key)
        subtree_data = _get_lines_that_match_key(headers, tree_key, file_data)
        subif = convert_to_if_then(
            id3_tree[tree_key], subtree_data,
            headers, tabs_prefix=tabs_prefix + 1
        )
        if_then += subif
    return if_then


def save_ifthen_to_file(ifthen):
    with open(IFTHEN_FILE_PATH, 'w') as ifthen_file_path:
        ifthen_file_path.write(ifthen)


def print_accuracy(id3_tree, headers, file_data):
    success = 0
    errors = 0
    for line in file_data:
        tested_data = test_data(id3_tree, headers, line)
        if tested_data:
            success += 1
        else:
            errors += 1
    print('Total {} Sucessos {} Erros {} Accuracy {}'.format(
        success+errors, success, errors,
        calculate_accuracy(id3_tree, file_data, headers)
    ))


def run(
    action, input_file_headers, input_file_data,
    fold_number, input_file_test
):
    headers = read_csv_file(input_file_headers)[0]
    file_data = read_csv_file(input_file_data)
    if action == 'training':
        total_entropy_adult = calculate_total_entropy(
            file_data, POSITIVE_DECISION, NEGATIVE_DECISION,
            DECISION_INDEX
        )
        id3_tree = train_decision_tree(
            headers, file_data, total_entropy_adult
        )
        save_json_to_file(id3_tree, TREE_FILE_NAME)

    if action == 'test':
        id3_tree = read_id3_tree()
        print_accuracy(id3_tree, headers, file_data)

    if action == 'test_prune':
        id3_tree = read_id3_tree_pruned()
        print_accuracy(id3_tree, headers, file_data)

    if action == 'validation':
        folds = separate_folds(file_data, fold_number)
        for i in range(fold_number):
            fold_aux = folds.pop(i)

            training_fold = []

            for j in folds:
                training_fold = training_fold + j

            total_entropy_adult = calculate_total_entropy(
                training_fold, POSITIVE_DECISION, NEGATIVE_DECISION,
                DECISION_INDEX
            )
            id3_tree = train_decision_tree(
                headers, training_fold, total_entropy_adult
            )
            print_accuracy(id3_tree, headers, fold_aux)

            folds.insert(i, fold_aux)

    if action == 'ifthen':
        id3_tree = read_id3_tree()
        ifthen = convert_to_if_then(id3_tree, file_data, headers)
        save_ifthen_to_file(ifthen)

    if action == 'prune':
        total_entropy_adult = calculate_total_entropy(
            file_data, POSITIVE_DECISION, NEGATIVE_DECISION,
            DECISION_INDEX
        )
        id3_tree = train_decision_tree(
            headers, file_data, total_entropy_adult
        )
        file_data = read_csv_file(input_file_data)
        file_data_test = read_csv_file(input_file_test)
        id3_pruned_tree = prune_tree(
            id3_tree, id3_tree, file_data_test, headers, file_data
        )
        save_json_to_file(id3_pruned_tree, TREE_PRUNED_FILE_NAME)


if __name__ == '__main__':
    if not len(sys.argv) > 2:
        print('Please provide the headers and data file path.')
    input_file_headers_param = sys.argv[1]
    input_file_data_param = sys.argv[2]
    fold_number = int(sys.argv[4])
    input_file_test = sys.argv[5]
    action = sys.argv[3]
    run(action, input_file_headers_param,
        input_file_data_param, fold_number,
        input_file_test)
