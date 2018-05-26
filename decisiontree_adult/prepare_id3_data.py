import sys
import re

from decisiontree.utils import read_csv_file

DIGIT_REGEX = '^\d+'
digit_regex_compiled = re.compile(DIGIT_REGEX)

def _has_exclude_flag(line, exclude_flag):
    for attribute in line:
        if attribute == exclude_flag:
            return True
    return False


def transform_continuos_data(
        continuous_attribute_indexes, discrete_values_dict,
        input_file, exclude_flag
):
    for line in input_file:
        if not _has_exclude_flag(line, exclude_flag):
            for attribute_index in continuous_attribute_indexes:
                continuous_attribute = line[attribute_index]
                discrete_value = discrete_values_dict[attribute_index]
                if continuous_attribute > discrete_value:
                    new_value = '>{}'.format(discrete_value)
                else:
                    new_value = '<={}'.format(discrete_value)
                line[attribute_index] = new_value
            yield ','.join(line)


def calculate_discrete_values(continuous_attribute_indexes, exclude_flag,
                              input_file):
    discrete_dict = {}
    for attributes_index in continuous_attribute_indexes:
        discrete_dict[attributes_index] = []
    excluded_lines = 0
    for line in input_file:
        if not _has_exclude_flag(line, exclude_flag):
            for continuous_attribute_index in continuous_attribute_indexes:
                discrete_dict[continuous_attribute_index].append(
                    line[continuous_attribute_index]
                )
        else:
            excluded_lines += 1
    for attribute in discrete_dict.keys():
        discrete_dict[attribute].sort()
        middle_index = int(len(discrete_dict[attribute])/2)
        discrete_dict[attribute] = discrete_dict[attribute][middle_index]
    return discrete_dict, excluded_lines


def write_output_file(output_file_path, output_file_generator):
    with open(output_file_path, 'w') as output:
        for line in output_file_generator:
            output.write(line + '\n')


def run(input_file_path, output_file_path, unknown_flag):
    adult_data = read_csv_file(input_file_path)
    continuous_attributes = [0, 2, 4, 10, 11, 12]
    discrete_values_dict, excluded_lines = calculate_discrete_values(
        continuous_attributes, unknown_flag, adult_data
    )
    output_file_generator = transform_continuos_data(
        continuous_attributes, discrete_values_dict, adult_data, unknown_flag
    )
    write_output_file(output_file_path, output_file_generator)


if __name__ == '__main__':
    if not len(sys.argv) > 3:
        print('Please provide the input, output and report file path.')
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    unknown_flag = sys.argv[3]

    run(input_file_path, output_file_path, unknown_flag)

