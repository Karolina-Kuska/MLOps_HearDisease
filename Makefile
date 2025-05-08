.PHONY: preprocess train all

# Przetwarzanie danych
preprocess:
	python test_preprocessing.py

# Trening modelu
train:
	python test_train_model.py

# Cały pipeline
all: preprocess train
