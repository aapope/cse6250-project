CSE6250 Project
==============================

This project is a replication of the HLAN model from [Explainable automated coding of clinical notes using hierarchical label-wise attention networks and label embedding initialisation](https://www.sciencedirect.com/science/article/pii/S1532046421000575). 

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── requirements.txt   <- The requirements file for reproducing the environment
    |
    ├── training_logs      <- The saved logs from training runs
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                    predictions
--------


Dependencies
------------
These are also listed in the requirements.txt
- Python 3.7.9
- gensim 3.8.3
- matplotlib 3.3.2
- nltk 3.5
- numpy 1.20.3
- pandas 1.1.4
- scikit_learn 1.0.1
- scipy 1.5.4
- spacy 2.3.2
- tensorflow 1.13.1
- tflearn 0.5.0
- tqdm 4.49.0
- pytorch 1.10.2
- ipykernel 6.18.2

Install
-----
Prepare your environment by creating a Python 3.7.9 environment with either conda or virtaulenv and installing the packages in requirements.txt:
```
pip install -r requirements.txt
```

Download and Preprocess Data
----------------------------
Download the source data from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-II](https://archive.physionet.org/physiobank/database/mimic2cdb/). The preprocessing is done using the code from the [caml-mimic repo](https://github.com/jamesmullenbach/caml-mimic). Follow the [Data preprocessing instructions in that repo](https://github.com/jamesmullenbach/caml-mimic#data-processing), running the two notebooks, to create the train/dev/test sets for both the full set of codes and the top 50 codes. Copy the resulting CSV files into `data/interim/`.

Generate Embeddings and Features
--------------------
First, generate the Word2Vec models for the word embeddings:
```
cd src/features && python create_word_embeddings_model.py
```

Then generate the Word2Vec models for code embeddings, which are used to initialize the attention weights and projection matrix in the model. You must do this twice, once for embeddings of size 200 (the current setting in the file), and again for embeddings of size 400. Edit [this line](https://github.com/aapope/cse6250-project/blob/master/src/features/create_code_embeddings_model.py#L17) to change the embedding size.
```
cd src/features && python create_code_embeddings_model.py
```

Now you can use these embeddings models to generate the final datasets for training and eval:
```
cd src/features && build_features.py
```
This script loads the data, applies word embeddings to the tokenized data set, one-hot encodes the labels, and restructures the data into the appropriate document, sentence, word dimensions.

Train the Model
--------------
Train the model by running:
```
cd src/models && python train_model.py
```

This script is cofigured to run for 100 epochs, which will take quite a long time. Loss on the training set is printed to stdout after every 10 batches, and loss on the dev set is printed after every epoch. After each epoch, the model is checkpointed to a separate file `models/HLANModel-interpretable-epoch{epoch}.pth`. It's advised to use the loss on the dev set to determine which model to select after training is complete.

Evaluate the Model
------------------
Evaluate the model on the dev set by running:
```
cd src/models && python evaluate_model.py
```
This will print the micro and macro metrics for each of AUC, F1, recall, and precision.

To see the per-label attentions you can run the notebook in `notebooks/label_visualization.ipynb`. This notebook allows you to select a particular sample and see what the sentence- and word-level attention weights were for each token in the input.
    

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------
