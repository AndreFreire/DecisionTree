import math

POSITIVE_INDEX = 'positive'
NEGATIVE_INDEX = 'negative'


def entropy(total, positive, negative):
    value = 0
    if positive:
        value -= (positive / total * math.log(positive / total, 2))
    if negative:
        value -= (negative / total * math.log(negative / total, 2))
    return value


def _init_attributes_information_gain_dict(attributes):
    attributes_information_gain_dict = {}
    for attribute in attributes:
        attributes_information_gain_dict[attribute] = {
            'value': 0,
            POSITIVE_INDEX: 0,
            NEGATIVE_INDEX: 0
        }
    return attributes_information_gain_dict


def calculate_information_gain(
        attributes, total_entropy, file_data,
        positive_flag, decision_index
):
    attribute_index = 0
    attributes_information_gain_dict = _init_attributes_information_gain_dict(attributes)
    total = len(file_data)
    while attribute_index < decision_index:
        attribute_data = _get_attribute_samples_count(
            attribute_index, file_data, positive_flag, decision_index
        )
        value = total_entropy
        positives_count, negatives_count = 0, 0
        for attribute in attribute_data:
            negative = attribute_data[attribute][NEGATIVE_INDEX]
            negatives_count += negative
            positive = attribute_data[attribute][POSITIVE_INDEX]
            positives_count += positive
            attribute_total = positive + negative
            value -= (attribute_total/total) * entropy(
                attribute_total, positive, negative
            )
        attributes_information_gain_dict[attributes[attribute_index]]['value'] = value
        attributes_information_gain_dict[attributes[attribute_index]][POSITIVE_INDEX] = positives_count
        attributes_information_gain_dict[attributes[attribute_index]][NEGATIVE_INDEX] = negatives_count
        attribute_index += 1
    return attributes_information_gain_dict


def calculate_total_entropy(file_data, positive_flag, negative_flag, decision_index):
    decision_index_attribute = _get_attribute_samples_count(
        decision_index, file_data, positive_flag, decision_index
    )
    positive = decision_index_attribute[
        positive_flag
    ][POSITIVE_INDEX] if decision_index_attribute.get(positive_flag) is not None else 0  # noqa
    negative = decision_index_attribute[
        negative_flag
    ][NEGATIVE_INDEX] if decision_index_attribute.get(negative_flag) is not None else 0  # noqa
    total_entropy = entropy(
        len(file_data),
        positive,
        negative
    )
    return total_entropy


def _get_attribute_samples_count(
        attribute_index, file_data, positive_flag, decision_index
):
    initial_value = {
        POSITIVE_INDEX: 0,
        NEGATIVE_INDEX: 0,
    }
    attribute_values = [value[attribute_index] for value in file_data]
    attribute_data = {}
    for attribute in attribute_values:
        attribute_data[attribute] = initial_value.copy()

    for line in file_data:
        if line[decision_index] == positive_flag:
            attribute_data[line[attribute_index]][POSITIVE_INDEX] += 1
        else:
            attribute_data[line[attribute_index]][NEGATIVE_INDEX] += 1
    return attribute_data