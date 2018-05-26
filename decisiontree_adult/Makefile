export PYTHONPATH=$(PWD)/..

clean:
	rm adult_new.data
	rm id3_tree.json

prepare_data:
	python3 prepare_id3_data.py adult.data adult_new.data ?

train:
	python3 decision_tree.py adult_headers.txt adult_new.data True

run_training_complete: clean prepare_data train

