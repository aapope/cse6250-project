import csv
import pickle
from gensim.models import Word2Vec

interim_dataset_names = ["dev_50", "test_50", "train_50"]
interim_path = "../../data/interim/"
processed_path = "../../data/processed/"


def load_data(dataset_name):
    """
    Loads dataset from interim data folder

    :param dataset_name: Name of CSV file for data to be transformed

    :return: A list with a tuple for each discharge summary where the first tuple element is a list of
    the tokens and the second tuple element is a list of the labels/ICD codes
    """
    with open(f"{interim_path}{dataset_name}.csv") as discharge_summaries_csv:
        csv_reader = csv.reader(discharge_summaries_csv, delimiter=",")
        # Skip header
        next(csv_reader)

        discharge_summaries_tokenized_with_label = []
        for discharge_summary in csv_reader:
            tokens = discharge_summary[2].split(' ')
            labels = discharge_summary[3].split(';')
            discharge_summaries_tokenized_with_label.append((tokens, labels))

    return discharge_summaries_tokenized_with_label


def integrate_word_embeddings(dataset):
    """
    Integrates word embeddings into the data for the discharge summaries

    Each token for a discharge summary will be replaced by its word embedding

    :param dataset: Data processed by load_data

    :return: A list with a tuple for each discharge summary where the first tuple element is a list of
    word embeddings (each word embedding is a numpy array) and the second tuple element is a list of
    the labels/ICD codes
    """
    model = Word2Vec.load("saved_embedding_models/word_embeddings.model")
    total_word_embeddings_with_labels = []
    for tokens, labels in dataset:
        discharge_summary_word_embeddings = [model.wv[token] for token in tokens]
        total_word_embeddings_with_labels.append((discharge_summary_word_embeddings, labels))

    return total_word_embeddings_with_labels


def create_label_map():
    """
    Creates a dictionary of labels to indices so we can one-hot encode labels

    :return: Dictionary of labels to indices
    """
    with open(f"{interim_path}top_50_codes.csv") as top_50_codes:
        csv_reader = csv.reader(top_50_codes, delimiter=",")
        # Skip header
        codes = [code[0] for code in csv_reader]

    label_map = dict(zip(codes, range(len(codes))))
    pickle.dump(label_map, open(f"{processed_path}/code_map.p", "wb"))
    return label_map


def one_hot_encode_labels(dataset, label_map):
    """
    Replaces the labels with a one hot encoded list of labels, using the label map

    :param label_map: Dictionary of labels to indices

    :return: Data set with one hot encoded labels
    """
    word_embeddings_with_encoded_labels = []
    for word_embeddings, labels in dataset:
        one_hot_list = [0] * len(label_map)

        for label in labels:
            one_hot_list[label_map[label]] = 1

        word_embeddings_with_encoded_labels.append((word_embeddings, one_hot_list))

    return word_embeddings_with_encoded_labels


def apply_formatting(dataset):
    """
    Applies formatting to the word embeddings to prepare the data for modeling

    The processed dataset doesn't have any punctuation, so from what I can tell, it seems like
    the authors capped the size of a document at 2500 tokens and determined that a sentence is
    a group of 25 tokens. Not completely sure, though

    We can also applying padding here or whatever else is needed to get the data ready for
    PyTorch modeling

    The format of the word embeddings will be a 3x nested list where the outermost list contains all of the
    sentences for the discharge summary. The next lists contain the word embeddings for each word in the
    sentence. Each word embedding is a 100 length list

    :param dataset:

    :return:
    """
    sentence_length = 25
    document_cap = 2500
    formatted_dataset = []

    for word_embeddings, labels in dataset:
        capped_document_length = len(word_embeddings) if len(word_embeddings) < document_cap else document_cap
        grouped_word_embeddings = [word_embeddings[index: index + sentence_length] for index in range(0, capped_document_length, sentence_length)]
        formatted_dataset.append((grouped_word_embeddings, labels))

    return formatted_dataset


if __name__ == '__main__':
    labels_map = create_label_map()

    for dataset_name in interim_dataset_names:
        tokens_with_labels = load_data(dataset_name)
        word_embeddings_with_labels = integrate_word_embeddings(tokens_with_labels)
        word_embeddings_with_encoded_labels = one_hot_encode_labels(word_embeddings_with_labels, labels_map)
        final_dataset = apply_formatting(word_embeddings_with_encoded_labels)

        pickle.dump(final_dataset, open(f"{processed_path}/{dataset_name}.p", "wb"))

