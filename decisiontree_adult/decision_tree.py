import re
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


def get_decision_attribute(id3_tree):
    return list(id3_tree.keys())[0].split(SEPARATOR)[0]


id3_key_regexes = {}


def get_id3_key(id3_tree, decision_attribute, decision_attribute_value):
    key = decision_attribute + SEPARATOR + decision_attribute_value
    keys_list = list(id3_tree.keys())
    if key in keys_list:
        return key
    else:
        print('Key not found for {} and {}'.format(
            decision_attribute, decision_attribute_value
        ))
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
    print(success)
    return success / (success + errors)


def prune_tree(id3_tree, id3_tree_part, file_data, headers):
    for key, value in id3_tree_part.items():
        print(key)
        if value not in [NEGATIVE_DECISION, POSITIVE_DECISION]:
            old_accuracy = calculate_accuracy(id3_tree, file_data, headers)
            positives = int(key.split(SEPARATOR)[2])
            negatives = int(key.split(SEPARATOR)[3])
            if positives > negatives:
                new_leaf = POSITIVE_DECISION
            else:
                new_leaf = NEGATIVE_DECISION

            branch_aux = copy(value)
            id3_tree_part[key] = new_leaf
            new_accuracy = calculate_accuracy(id3_tree, file_data, headers)
            if old_accuracy > new_accuracy:
                id3_tree_part[key] = branch_aux
            prune_tree(id3_tree, value, file_data, headers)
    return id3_tree


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
        success = 0
        errors = 0
        for line in file_data:
            tested_data = test_data(id3_tree, headers, line)
            if tested_data:
                success += 1
            else:
                errors += 1
        print('Total {} Sucessos {} Erros {}'.format(
            success+errors, success, errors)
        )

    if action == 'validation':
        folds = separate_folds(file_data, fold_number)
        for i in range(fold_number):
            fold_aux = folds.pop(i)

            training_fold = []

            for j in fold_aux:
                training_fold.append(j)

            total_entropy_adult = calculate_total_entropy(
                training_fold, POSITIVE_DECISION, NEGATIVE_DECISION,
                DECISION_INDEX
            )
            id3_tree = train_decision_tree(
                headers, training_fold, total_entropy_adult
            )

            success = 0
            errors = 0
            for line in fold_aux:
                tested_data = test_data(id3_tree, headers, line)
                if tested_data:
                    success += 1
                else:
                    errors += 1
            print('Total {} Sucessos {} Erros {}'.format(
                success+errors, success, errors)
            )

            folds.insert(i, fold_aux)

    if action == 'prune':
        total_entropy_adult = calculate_total_entropy(
            file_data, POSITIVE_DECISION, NEGATIVE_DECISION,
            DECISION_INDEX
        )
        id3_tree = train_decision_tree(
            headers, file_data, total_entropy_adult
        )
        file_data = read_csv_file(input_file_test)
        id3_pruned_tree = prune_tree(id3_tree, id3_tree, file_data, headers)
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
