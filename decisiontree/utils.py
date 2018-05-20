

def read_csv_file(file_name):
    with open(file_name) as csv_file:
        return [line.replace('\n', '').split(',') for line in csv_file]