export PYTHONPATH=$(PWD)/..
TRAINING_FILE_RAW := adult.complete
TRAINING_FILE_PROCESSED := adult_new.complete
TEST_FILE_PROCESSED := adult.test
HEADERS_FILE := adult_headers.txt

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
	python3 prepare_id3_data.py $(TRAINING_FILE_RAW) $(TRAINING_FILE_PROCESSED) ?

train:
	python3 decision_tree.py $(HEADERS_FILE) adult.data training 0 0

validate:
	python3 decision_tree.py $(HEADERS_FILE) $(TRAINING_FILE_PROCESSED) validation 10 0

test:
	python3 decision_tree.py $(HEADERS_FILE) $(TEST_FILE_PROCESSED) test 0 0

test_prune:
	python3 decision_tree.py $(HEADERS_FILE) $(TEST_FILE_PROCESSED) test_prune 0 0

prune:
	python3 decision_tree.py $(HEADERS_FILE) adult.data prune 0 $(TEST_FILE_PROCESSED)

ifthen:
	python3 decision_tree.py $(HEADERS_FILE) adult.data ifthen 0 $(TEST_FILE_PROCESSED)


divide_prepare_file: prepare_data
	./divide_file.sh $(TRAINING_FILE_PROCESSED) adult.data adult.test

run_training_complete: clean divide_prepare_file train

run_valid_fold_complete: validate

run_test_complete: test

run_prune_test_complete: test_prune

run_training_prune_complete: prune
