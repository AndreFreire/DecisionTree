#!/bin/bash
input_file="$1"
temp_file="temp_file.data"
temp_file_30="temp_file_30.data"
file_lines=$(wc -l < $input_file)
output_file_data="$2"
output_file_test="$3"
output_file_prune="$4"
python -c "import random, sys; x = open(sys.argv[1]).readlines(); random.shuffle(x); print(''.join(x))," "$input_file" >> "$temp_file"
lines=$(($file_lines*70/100))
head -n $lines "$temp_file" > "$output_file_data"
lines=$(($file_lines*30/100))
tail -n $lines "$temp_file" > "$temp_file_30"
file_lines=$(wc -l < $temp_file_30)
lines=$(($file_lines*50/100))
head -n $lines "$temp_file_30" > "$output_file_prune"
tail -n $lines "$temp_file_30" > "$output_file_test"
rm -f "$temp_file"
rm -f "$temp_file_30"