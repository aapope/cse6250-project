import csv
from gensim.models import Word2Vec


def create_word_embeddings_model():
    # Word embeddings is based on all preprocessed discharge summaries, not just those with top 50 ICD codes
    with open("../../data/interim/disch_full.csv") as discharge_summaries_csv:
        csv_reader = csv.reader(discharge_summaries_csv, delimiter=",")
        # Skip header
        next(csv_reader)

        discharge_summaries_tokenized = []
        for discharge_summary in csv_reader:
            tokens = discharge_summary[3].split(' ')
            discharge_summaries_tokenized.append(tokens)

        model = Word2Vec(discharge_summaries_tokenized, size=100, window=5, min_count=0)
        model.save("saved_embedding_models/word_embeddings.model")


if __name__ == '__main__':
    create_word_embeddings_model()
