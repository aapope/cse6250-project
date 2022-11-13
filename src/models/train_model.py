import pickle
import logging
import os
import sys
from pprint import pprint
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from gensim.models import Word2Vec

from hlan_model import HLAN


PROCESSED_DATA_PATH = '../../data/processed'
MODEL_OUTPUT_PATH = '../../models/HLANModel.pth'

NUM_EPOCHS = 30
BATCH_SIZE = 32
USE_CUDA = True


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


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


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
    label_tensor = torch.tensor(labels).float()
    
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



def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
        
        loss.backward()
        # TODO: gradient clipping, if needed: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        losses.update(loss.item(), target.size(0))
        
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(data_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
            
    return losses.avg


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")


def get_label_embeddings():
    model = Word2Vec.load('../features/saved_embedding_models/code_embeddings.model')
    with open(os.path.join(PROCESSED_DATA_PATH, 'code_map.p'), 'rb') as f:
        code_map = pickle.load(f)
    sorted_codes = sorted(code_map.items(), key=lambda x: x[1])
    embeddings = np.array([model.wv[code[0]] for code in sorted_codes])
    return embeddings.T
    

def run():
    train_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'train_50.p'))
    dev_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'dev_50.p'))
    test_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'test_50.p'))
    label_embeddings = get_label_embeddings()

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=document_collate_function,
        num_workers=0
    )
    dev_loader = DataLoader(
        dataset=dev_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=document_collate_function,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=document_collate_function,
        num_workers=0
    )

    model = HLAN(label_embeddings=label_embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    device = get_device()
    torch.manual_seed(10)
    model.to(device)
    criterion.to(device)

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, criterion, optimizer, epoch)
        torch.save(model, MODEL_OUTPUT_PATH)



if __name__ == '__main__':
    run()
