export PYTHONPATH=$(PWD)/..

clean:
	rm -f adult_new.data
	rm -f id3_tree.json

clean_test:
	rm adult_new.test

clean_validate:
	rm adult_new.data	

prepare_data:
	python3 prepare_id3_data.py adult.data adult_new.data ?

prepare_data_test:
	python3 prepare_id3_data.py adult.test adult_new.test ?

train:
	python3 decision_tree.py adult_headers.txt adult_new.data training 0

validate:
	python3 decision_tree.py adult_headers.txt adult_new.data validation 10

test:
	python3 decision_tree.py adult_headers.txt adult_new.test test 0

run_training_complete: clean prepare_data train

run_valid_fold_complete: clean_validate prepare_data validate

run_test_complete: clean_test prepare_data_test test
