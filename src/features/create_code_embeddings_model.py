import csv
from gensim.models import Word2Vec


def create_code_embeddings_model():
    # Code embeddings are based on the labels for the training discharge summaries, not just those with top 50 codes
    with open("../../data/interim/train_full.csv") as discharge_summaries_csv:
        csv_reader = csv.reader(discharge_summaries_csv, delimiter=",")
        # Skip header
        next(csv_reader)

        discharge_summaries_codes = []
        for discharge_summary in csv_reader:
            codes = discharge_summary[3].split(';')
            discharge_summaries_codes.append(codes)

        model = Word2Vec(discharge_summaries_codes, size=200, window=5, min_count=0)
        model.save("saved_embedding_models/code_embeddings_200.model")


if __name__ == '__main__':
    create_code_embeddings_model()
