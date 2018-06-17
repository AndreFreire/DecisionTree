export PYTHONPATH=$(PWD)/..
TRAINING_FILE_RAW := adult.complete
TRAINING_FILE_PROCESSED := adult_new.complete
TEST_FILE_PROCESSED := adult.test
VALIDATION_FILE_PROCESSED := adult.validation
HEADERS_FILE := adult_headers.txt

clean:
	rm -f adult_new.complete
	rm -f adult.data
	rm -f adult.test
	rm -f id3_tree.json

clean_test:
	rm -f adult.test

clean_validation:
	rm -f adult.validation

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
	python3 decision_tree.py $(HEADERS_FILE) adult.data prune 0 $(VALIDATION_FILE_PROCESSED)

ifthen:
	python3 decision_tree.py $(HEADERS_FILE) adult_new.complete ifthen 0 $(TEST_FILE_PROCESSED)

ifthen-prune:
	python3 decision_tree.py $(HEADERS_FILE) adult_new.complete ifthen-prune 0 $(TEST_FILE_PROCESSED)

divide_prepare_file: prepare_data
	./divide_file.sh $(TRAINING_FILE_PROCESSED) adult.data adult.test adult.validation

divide_prepare_file_validation: prepare_data 
	./divide_file_validation.sh adult_new.complete adult.data adult.test adult.validation

run_training_complete: clean divide_prepare_file train

run_valid_fold_complete: clean prepare_data validate

run_test_complete: test

run_prune_test_complete: test_prune

run_training_prune_complete: clean_validation divide_prepare_file_validation prune
