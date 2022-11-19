from train_model import document_collate_function, load_dataset
import torch
from torch.utils.data import DataLoader
import os


PROCESSED_DATA_PATH = '../../data/processed'
MODEL_OUTPUT_PATH = '../../models/HLANModel.pth'
BATCH_SIZE = 32


def evaluate():
    dev_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'dev_50_sample.p'))
    dev_loader = DataLoader(
        dataset=dev_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=document_collate_function,
        num_workers=0
    )

    saved_model = torch.load(MODEL_OUTPUT_PATH)

    for batch in dev_loader:
        pass


if __name__ == '__main__':
    evaluate()