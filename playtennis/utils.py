

def read_csv_file(file_name):
    lines = []
    with open(file_name) as csv_file:
        for line in csv_file:
            if line.strip():
                new_line = []
                for value in line.replace('\n', '').split(','):
                    value = value.strip()
                    new_line.append(value)
                if new_line:
                    lines.append(new_line)
    return lines
