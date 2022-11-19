import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HLAN(nn.Module):
    def __init__(self, max_document_length=100, sentence_length=25, embedding_length=100, num_labels=50, word_level_hidden_size=100, sentence_level_hidden_size=200, use_sentence_attention_per_label=True, use_word_attention_per_label=True, label_embeddings=None):
        super(HLAN, self).__init__()
        
        self.max_document_length = max_document_length
        self.sentence_length = sentence_length
        self.embedding_length = embedding_length
        self.num_labels = num_labels
        self.word_level_hidden_size = word_level_hidden_size
        self.sentence_level_hidden_size = sentence_level_hidden_size
        self.use_sentence_attention_per_label = use_sentence_attention_per_label
        self.use_word_attention_per_label = use_word_attention_per_label

        self.word_level_gru = nn.GRU(
            embedding_length, word_level_hidden_size, num_layers=1, batch_first=True,
            bidirectional=True
        )
        self.word_level_attention = nn.Linear(
            2 * word_level_hidden_size,
            2 * word_level_hidden_size
        )

        if label_embeddings is not None:
            print('Using label embeddings to initialize attention matrix')
            label_embeddings = torch.Tensor(label_embeddings).float()
            # TODO: in the paper's code, this is (200, 50), so we need to regenerate these
            # and remove this line
            label_embeddings = label_embeddings[:2 * word_level_hidden_size, :] 
            assert label_embeddings.shape == (2 * word_level_hidden_size, num_labels)

        if use_word_attention_per_label:
            if label_embeddings is not None:
                self.context_vector_word = nn.Parameter(label_embeddings.T)
            else:
                self.context_vector_word = nn.Parameter(torch.ones(num_labels, word_level_hidden_size * 2))
        else:
            self.context_vector_word = nn.Parameter(torch.ones(self.word_level_hidden_size * 2))

        self.sentence_level_gru = nn.GRU(
            2 * word_level_hidden_size, sentence_level_hidden_size, num_layers=1, batch_first=True,
            bidirectional=True
        )
            
        self.sentence_level_attention = nn.Linear(
            2 * sentence_level_hidden_size,
            sentence_level_hidden_size
        )

        
        if use_sentence_attention_per_label:
            # initialize weights the same way that nn.Linear would
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            if label_embeddings is not None:
                self.context_vector_sentence = nn.Parameter(label_embeddings.T)
            else:
                self.context_vector_sentence = nn.Parameter(torch.ones(num_labels, 2 * self.word_level_hidden_size))
                
            bound = math.sqrt(1 / (2 * self.sentence_level_hidden_size))
            self.W_output = nn.Parameter(
                torch.FloatTensor(2 * self.sentence_level_hidden_size, num_labels).uniform_(-bound, bound)
            )
            self.b_output = nn.Parameter(
                torch.FloatTensor(num_labels).uniform_(-bound, bound)
            )
        else:
            self.context_vector_sentence = nn.Parameter(torch.ones(2 * self.sentence_level_hidden_size))
            self.output_layer = nn.Linear(
                2 * sentence_level_hidden_size,
                num_labels
            )

    def forward(self, x):
        data, document_lengths, sentence_lengths = x
        document_length = document_lengths.max().item()
        # data is of shape (batch_size, num_sentences, sentence_length, embed_length)
        # print(f'Input data shape: {data.shape}')

        # reshape to (batch_size * num_sentences, sentence_length, embed_length)
        words = data.reshape(-1, self.sentence_length, self.embedding_length)
        # print(f'Shape into Word GRU: {words.shape}')

        # shape: (batch_size * num_sentences, sentence_length, 2 * hidden)
        gru_output, _ = self.word_level_gru(words)
        # print(f'Shape of Word GRU output: {gru_output.shape}')

        if self.use_word_attention_per_label:
            sentences = self.word_attention_per_label(gru_output)
            sentences = sentences.reshape(
                self.num_labels, -1, document_length, 2 * self.word_level_hidden_size
            )
            # print(f'Shape of word attention output: {sentences.shape}')

            gru_output = self.stack_gru(sentences)
            # print(f'Shape of GRU output: {gru_output.shape}')
        else:
            # shape: (batch_size, num_sentences, 2 * hidden)
            sentences = self.word_attention(gru_output)
            sentences = sentences.reshape(
                -1, document_length, 2 * self.word_level_hidden_size
            )
            # print(f'Shape of word attention output: {sentences.shape}')

            # shape (batch_size, num_sentences, 2 * sentence_hidden)
            gru_output, _ = self.sentence_level_gru(sentences)
            # print(f'Shape of Sentence GRU output: {gru_output.shape}')

        if self.use_sentence_attention_per_label:
            documents = self.sentence_attention_per_label(gru_output, document_length)
            # TODO: the paper's code uses dropout here. we might want to try that
            documents = documents.permute(dims=(1, 2, 0))
            # print(documents.shape)

            logits = torch.mul(documents, self.W_output).sum(dim=1) + self.b_output
        else:
            documents = self.sentence_attention(gru_output, document_length)
            logits = self.output_layer(documents)
            
        # print(logits.shape)
        return logits

    def stack_gru(self, sentences):
        # intresestingly, this method is ~1/3 faster than doing exactly
        # this same thing in a for loop
        return torch.stack([
            self.sentence_level_gru(sentences[0, :, :, :])[0],
            self.sentence_level_gru(sentences[1, :, :, :])[0],
            self.sentence_level_gru(sentences[2, :, :, :])[0],
            self.sentence_level_gru(sentences[3, :, :, :])[0],
            self.sentence_level_gru(sentences[4, :, :, :])[0],
            self.sentence_level_gru(sentences[5, :, :, :])[0],
            self.sentence_level_gru(sentences[6, :, :, :])[0],
            self.sentence_level_gru(sentences[7, :, :, :])[0],
            self.sentence_level_gru(sentences[8, :, :, :])[0],
            self.sentence_level_gru(sentences[9, :, :, :])[0],
            self.sentence_level_gru(sentences[10, :, :, :])[0],
            self.sentence_level_gru(sentences[11, :, :, :])[0],
            self.sentence_level_gru(sentences[12, :, :, :])[0],
            self.sentence_level_gru(sentences[13, :, :, :])[0],
            self.sentence_level_gru(sentences[14, :, :, :])[0],
            self.sentence_level_gru(sentences[15, :, :, :])[0],
            self.sentence_level_gru(sentences[16, :, :, :])[0],
            self.sentence_level_gru(sentences[17, :, :, :])[0],
            self.sentence_level_gru(sentences[18, :, :, :])[0],
            self.sentence_level_gru(sentences[19, :, :, :])[0],
            self.sentence_level_gru(sentences[20, :, :, :])[0],
            self.sentence_level_gru(sentences[21, :, :, :])[0],
            self.sentence_level_gru(sentences[22, :, :, :])[0],
            self.sentence_level_gru(sentences[23, :, :, :])[0],
            self.sentence_level_gru(sentences[24, :, :, :])[0],
            self.sentence_level_gru(sentences[25, :, :, :])[0],
            self.sentence_level_gru(sentences[26, :, :, :])[0],
            self.sentence_level_gru(sentences[27, :, :, :])[0],
            self.sentence_level_gru(sentences[28, :, :, :])[0],
            self.sentence_level_gru(sentences[29, :, :, :])[0],
            self.sentence_level_gru(sentences[30, :, :, :])[0],
            self.sentence_level_gru(sentences[31, :, :, :])[0],
            self.sentence_level_gru(sentences[32, :, :, :])[0],
            self.sentence_level_gru(sentences[33, :, :, :])[0],
            self.sentence_level_gru(sentences[34, :, :, :])[0],
            self.sentence_level_gru(sentences[35, :, :, :])[0],
            self.sentence_level_gru(sentences[36, :, :, :])[0],
            self.sentence_level_gru(sentences[37, :, :, :])[0],
            self.sentence_level_gru(sentences[38, :, :, :])[0],
            self.sentence_level_gru(sentences[39, :, :, :])[0],
            self.sentence_level_gru(sentences[40, :, :, :])[0],
            self.sentence_level_gru(sentences[41, :, :, :])[0],
            self.sentence_level_gru(sentences[42, :, :, :])[0],
            self.sentence_level_gru(sentences[43, :, :, :])[0],
            self.sentence_level_gru(sentences[44, :, :, :])[0],
            self.sentence_level_gru(sentences[45, :, :, :])[0],
            self.sentence_level_gru(sentences[46, :, :, :])[0],
            self.sentence_level_gru(sentences[47, :, :, :])[0],
            self.sentence_level_gru(sentences[48, :, :, :])[0],
            self.sentence_level_gru(sentences[49, :, :, :])[0],
        ], dim=0)


    def sentence_attention(self, hidden_input, document_length):
        hidden_input_reshaped = hidden_input.reshape(-1, 2 * self.sentence_level_hidden_size)
        # print(hidden_input_reshaped.shape)

        sentence_level_attention = torch.tanh(self.sentence_level_attention(hidden_input_reshaped))
        # todo: in the paper's code, this is reshaped to (-1, doc length, sent level hidden), which
        # results in duplicating the batch size. I couldn't get it to work that way.
        sentence_level_attention = sentence_level_attention.reshape(-1, document_length, 2 * self.sentence_level_hidden_size)
        # print(sentence_level_attention.shape)

        x = torch.mul(sentence_level_attention, self.context_vector_sentence)
        attention_logits = torch.sum(x, dim=2)
        attention_logits_max, _ = torch.max(attention_logits, dim=1, keepdim=True)

        p_attention = F.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = torch.unsqueeze(p_attention, dim=-1)

        document_representation = torch.sum(
            torch.mul(p_attention_expanded, hidden_input),
            dim=1
        )
        # print(document_representation.shape)
        return document_representation
        

    def word_attention(self, hidden_input):
        # reshape to (batch_size * num_sentences * sentence_length, 2 * hidden)
        hidden_input_reshaped = hidden_input.reshape(-1, 2 * self.word_level_hidden_size)
        # print(f'Shape of input to word attention: {hidden_input.shape}')

        # same shape as above: (batch_size * num_sentences * sentence_length, 2 * hidden)
        word_level_attention = torch.tanh(self.word_level_attention(hidden_input_reshaped))
        # print(f'Shape of word level attention output: {word_level_attention.shape}')

        # back to (batch_size * num_sentences, sentence_length, 2 * hidden)
        x = word_level_attention.reshape(-1, self.sentence_length, 2 * self.word_level_hidden_size)
        # print(x.shape)

        # (batch_size * num_sentences, sentence_length, 2 * hidden)
        x = torch.mul(x, self.context_vector_word)
        # print(x.shape)

        # (batch_size * num_sentences, sentence_length)
        attention_logits = torch.sum(x, dim=2)
        # print(attention_logits.shape)

        # (batch_size * num_sentences, 1)
        attention_logits_max, _ = torch.max(attention_logits, dim=1, keepdim=True)
        # print(attention_logits_max.shape)

        # (batch_size * num_sentences, sentence_length)
        p_attention = F.softmax(attention_logits - attention_logits_max)
        # print(p_attention.shape)

        # (batch_size * num_sentences, sentence_length, 1)
        p_attention_expanded = torch.unsqueeze(p_attention, dim=-1)
        # print(p_attention_expanded.shape)

        # (batch_size * num_sentences, 2 * hidden)
        sentence_representation = torch.sum(
            torch.mul(p_attention_expanded, hidden_input),
            dim=1
        )
        # print(sentence_representation.shape)

        return sentence_representation
        
        
    def word_attention_per_label(self, hidden_input):
        # reshape to (batch_size * num_sentences * sentence_length, 2 * hidden)
        hidden_input_reshaped = hidden_input.reshape(-1, 2 * self.word_level_hidden_size)
        # print(f'Shape of input to word attention: {hidden_input.shape}')

        # same shape as above: (batch_size * num_sentences * sentence_length, 2 * hidden)
        word_level_attention = torch.tanh(self.word_level_attention(hidden_input_reshaped))
        # print(f'Shape of word level attention output: {word_level_attention.shape}')

        # back to (batch_size * num_sentences, sentence_length, 2 * hidden)
        x = word_level_attention.reshape(-1, self.sentence_length, 2 * self.word_level_hidden_size)
        # print(x.shape)

        x = x.unsqueeze(dim=0)
        # print(f'After reshape: {x.shape}')

        expanded_context_vector = self.context_vector_word.unsqueeze(dim=1).unsqueeze(dim=1)
        # print(f'Context vector: {expanded_context_vector.shape}')

        x = torch.mul(x, expanded_context_vector)
        # print(f'Similarity: {x.shape}')
        
        attention_logits = x.sum(dim=-1)
        # print(f'After sum: {attention_logits.shape}')

        attention_logits_max, _ = attention_logits.max(dim=-1, keepdim=True)
        # print(attention_logits_max.shape)

        p_attention = F.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = p_attention.unsqueeze(dim=-1)
        # print(p_attention_expanded.shape)

        sentence_representation = torch.sum(
            torch.mul(p_attention_expanded, hidden_input),
            dim=2
        )
        # print(sentence_representation.shape)

        return sentence_representation

    def sentence_attention_per_label(self, hidden_input, document_length):
        if self.use_word_attention_per_label:
            hidden_input_reshaped = hidden_input.reshape(self.num_labels, -1, 2 * self.sentence_level_hidden_size)
        else:
            hidden_input_reshaped = hidden_input.reshape(-1, 2 * self.sentence_level_hidden_size)

        # print(f'Reshaped: {hidden_input_reshaped.shape}')

        sentence_level_attention = torch.tanh(self.sentence_level_attention(hidden_input_reshaped))
        # print(f'After feed forward: {sentence_level_attention.shape}')
        if self.use_word_attention_per_label:
            sentence_level_attention = sentence_level_attention.reshape(self.num_labels, -1, document_length, self.sentence_level_hidden_size)
        else:
            sentence_level_attention = sentence_level_attention.reshape(-1, document_length, self.sentence_level_hidden_size)
            sentence_level_attention = sentence_level_attention.unsqueeze(dim=0)
        # print(f'After expansion: {sentence_level_attention.shape}')

        expanded_context_vector = self.context_vector_sentence.unsqueeze(dim=1).unsqueeze(dim=1)
        # print(f'Context vector: {expanded_context_vector.shape}')

        x = torch.mul(sentence_level_attention, expanded_context_vector)
        # print(f'Similarity: {x.shape}')

        attention_logits = x.sum(dim=-1)
        attention_logits_max, _ = torch.max(attention_logits, dim=-1, keepdim=True)
        p_attention = F.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = p_attention.unsqueeze(dim=-1)
        # print(p_attention_expanded.shape)

        document_representation = torch.sum(
            torch.mul(p_attention_expanded, hidden_input),
            dim=2
        )
        # print(document_representation.shape)

        return document_representation
        
