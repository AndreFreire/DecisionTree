export PYTHONPATH=$(PWD)/..

clean:
	rm -f adult_new.data
	rm -f id3_tree.json

clean_test:
	rm -f adult_new.test

clean_validate:
	rm -f adult_new.data

prepare_data:
	python3 prepare_id3_data.py adult.complete adult_new.complete ?

train:
	python3 decision_tree.py adult_headers.txt adult_new.complete training 0

validate:
	python3 decision_tree.py adult_headers.txt adult_new.data validation 10

test:
	python3 decision_tree.py adult_headers.txt adult_new.test test 0

divide_prepare_file: clean prepare_data
	./divide_file.sh adult_new.complete adult.data adult.test

run_training_complete: divide_prepare_file train

run_valid_fold_complete: divide_prepare_file validate

run_test_complete: divide_prepare_file test
