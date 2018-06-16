#!/bin/bash
input_file="$1"
temp_file="temp_file.data"
file_lines=$(wc -l < $input_file)
output_file_data="$2"
output_file_test="$3"
python -c "import random, sys; x = open(sys.argv[1]).readlines(); random.shuffle(x); print ''.join(x)," "$input_file" >> "$temp_file"
lines=$(($file_lines*80/100))
head -n $lines "$temp_file" > "$output_file_data"
lines=$(($file_lines*20/100))
tail -n $lines "$temp_file" > "$output_file_test"
rm -f "$temp_file"
