import json
import sys

from decisiontree.id3_algorithm import calculate_total_entropy, calculate_information_gain
from decisiontree.utils import read_csv_file

SEPARATOR = '__'

DECISION_INDEX = 4
JOGA_TENIS = '1'
NAO_JOGA_TENIS = '0'


def _get_attribute_with_max_information_gain(
        attributtes_information_gain_dict
):
    max_information_gain = 0
    attribute_with_max_information_gain = None
    for attribute, information_gain in attributtes_information_gain_dict.items():  # noqa
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            attribute_with_max_information_gain = attribute
    return attribute_with_max_information_gain


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
        attributes, entropy, lines, JOGA_TENIS, DECISION_INDEX
    )
    selected_attribute = _get_attribute_with_max_information_gain(
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
            variations_lines, JOGA_TENIS, NAO_JOGA_TENIS, DECISION_INDEX
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


if __name__ == '__main__':
    training = len(sys.argv) > 1 and sys.argv[1].lower() == 'true'

    headers = read_csv_file('playtennis_headers.txt')[0]
    play_tennis_file_data = read_csv_file('play_tennis.txt')

    if training:
        total_entropy_play_tennis = calculate_total_entropy(
            play_tennis_file_data, JOGA_TENIS, NAO_JOGA_TENIS, DECISION_INDEX
        )
        id3_tree = train_decision_tree(
            headers, play_tennis_file_data, total_entropy_play_tennis
        )
        save_json_to_file(id3_tree, 'id3_tree.json')
    else:
        pass
