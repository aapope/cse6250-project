import pickle
import logging
import os
import sys
from pprint import pprint

import torch
from torch.utils.data import Dataset
import numpy as np

from hlan_model import HLAN


PROCESSED_DATA_PATH = '../../data/processed'
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)


class HLANDataset(Dataset):
    def __init__(self, input_data):
        """
        :param input_data: List of tuples (sequence, label), where each
            sequence is a List (sentences in doc) of Lists (words in sentence)
            of List of floats (embedding of word).
        """
        self.labels = [sample[1] for sample in input_data]
        self.sequence = [sample[0] for sample in input_data]
        # print(len(self.sequence))
        # print(len(self.sequence[0]))
        # print(len(self.sequence[0][0]))
        # print(self.sequence[0][0][0].shape)
        self.doc_lengths = [len(sample) for sample in self.sequence]
        self.sentence_lengths = [
            [len(sentence) for sentence in sample]
            for sample in self.sequence
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequence[index], self.labels[index], self.doc_lengths[index], self.sentence_lengths[index]


def document_collate_function(batch):
    """
    Creates a batch of Tensors from the input data. The returned value
    contains (data, document_lengths, sentence_lengths), labels.

    let:
    - N = number of samples in the batch
    - D = maximum number of sentences in a document in this batch
    - S = 25 = maximum number of words in a sentence in this batch
    - W = 100 = word embedding size
    - L = 50 = number of labels

    the returned values are of shape:
    data => (N, D, S, W)
    document_lengths => (N,)
    sentence_lengths => (N, D)
    labels => (N, L)

    The type of data is FloatTensor; the others are LongTensors.
    """
    max_doc_length = max([document[2] for document in batch])
    max_sentence_length = 25
    embedding_length = 100

    data = []
    labels = []
    doc_lengths = []
    sentence_lengths = []
    for document in batch:
        labels.append(document[1])
        doc_lengths.append(document[2])

        # pad sentences within document
        sentences = []
        for sentence in document[0]:
            array = np.array(sentence)
            sentences.append(np.pad(
                array,
                # pad rows (ie words) to get to max sentence length
                [(0, max_sentence_length - array.shape[0]), (0, 0)],
                'constant'
            ))

        # pad document
        doc = np.array(sentences)
        data.append(np.pad(
            doc,
            [(0, max_doc_length - doc.shape[0]), (0, 0), (0, 0)],
            'constant'
        ))

        sentence_lengths.append(
            np.pad(
                np.array(document[3]),
                [(0, max_doc_length - len(document[3]))],
                'constant'
            )
        )

    # from UserWarning: Creating a tensor from a list of numpy.ndarrays is
    # extremely slow. so creating np arrays first
    data_tensor = torch.tensor(np.array(data)).float()
    doc_lengths_tensor = torch.tensor(doc_lengths).long()
    sent_lengths_tensor = torch.tensor(np.array(sentence_lengths)).long()
    label_tensor = torch.tensor(labels).long()
    
    return (data_tensor, doc_lengths_tensor, sent_lengths_tensor), label_tensor


def load_dataset(dataset_path):
    """
    Loads a preprocessed dataset (i.e., the one expored from
    features.apply_formatting) pickle file and returns a TensorDataset.

    :param dataset_path: path to the processed dataset, in pickle format
    :return: a HLANDataset
    """
    logging.info(f'Loading data from file {dataset_path}...')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    return HLANDataset(dataset)

def run():
    # train_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'train_50.p'))
    dev_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'dev_50.p'))
    # test_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'test_50.p'))

    batch = document_collate_function([dev_dataset[i] for i in range(8)])

    model = HLAN()
    model(batch[0])


if __name__ == '__main__':
    run()
