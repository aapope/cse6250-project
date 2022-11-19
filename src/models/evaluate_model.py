from train_model import document_collate_function, load_dataset
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import torch
from torch.utils.data import DataLoader
import os


PROCESSED_DATA_PATH = '../../data/processed'
MODEL_OUTPUT_PATH = '../../models/HLANModel.pth'
BATCH_SIZE = 32
THRESHOLD = 0.5


def evaluate():
    dev_dataset = load_dataset(os.path.join(PROCESSED_DATA_PATH, 'dev_50.p'))
    dev_loader = DataLoader(
        dataset=dev_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=document_collate_function,
        num_workers=0
    )

    saved_model = torch.load(MODEL_OUTPUT_PATH)
    saved_model.eval()

    with torch.no_grad():
        predicted_probabilities_batches = []
        actual_labels_batches = []

        for batch, labels in dev_loader:
            sigmoid_func = torch.nn.Sigmoid()
            predicted_probabilities_batches.append(sigmoid_func(saved_model(batch)))
            actual_labels_batches.append(labels)

        predicted_probabilities = torch.cat(predicted_probabilities_batches).numpy()
        actual_labels = torch.cat(actual_labels_batches).numpy()
        predicted_labels = (predicted_probabilities > THRESHOLD) * 1

        print(f"Micro AUC is: {roc_auc_score(actual_labels, predicted_probabilities, average='micro')}")
        print(f"Macro AUC is: {roc_auc_score(actual_labels, predicted_probabilities, average='macro')}")

        print(f"Micro F1 is: {f1_score(actual_labels, predicted_labels, average='micro')}")
        print(f"Macro F1 is: {f1_score(actual_labels, predicted_labels, average='macro')}")

        print(f"Micro Recall is: {recall_score(actual_labels, predicted_labels, average='micro')}")
        print(f"Macro Recall is: {recall_score(actual_labels, predicted_labels, average='macro')}")

        print(f"Micro Precision is: {precision_score(actual_labels, predicted_labels, average='micro')}")
        print(f"Macro Precision is: {precision_score(actual_labels, predicted_labels, average='macro')}")


if __name__ == '__main__':
    evaluate()