export PYTHONPATH=$(PWD)/..

clean:
	rm -f adult_new.complete
	rm -f adult.data
	rm -f adult.test
	rm -f id3_tree.json

clean_test:
	rm -f adult.test

clean_validate:
	rm -f adult.data

prepare_data:
	python3 prepare_id3_data.py adult.complete adult_new.complete ?

train:
	python3 decision_tree.py adult_headers.txt adult.data training 0 0 

validate:
	python3 decision_tree.py adult_headers.txt adult_new.complete validation 10 0

test:
	python3 decision_tree.py adult_headers.txt adult.data test 0 0

prune:
	python3 decision_tree.py adult_headers.txt adult.data prune 0 adult.test 


divide_prepare_file: prepare_data
	./divide_file.sh adult_new.complete adult.data adult.test

#run_training_complete: clean divide_prepare_file train

run_training_complete: train

run_valid_fold_complete: divide_prepare_file validate

run_test_complete: divide_prepare_file test

run_training_prune_complete: prune
