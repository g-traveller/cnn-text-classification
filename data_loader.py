import pandas as pd
import json
import jieba
import numpy as np

import word2vec_helper


class DataLoader(object):

    def __init__(self, training_data_path):
        self.word2vec_model = word2vec_helper.load()
        self.training_data_path = training_data_path
        self.padding_token = '<PADDING>'
        self.intent_mapping = json.load(open('./data/intent_mapping.json'))
        self.intent_size = len(self.intent_mapping)

    def load(self, max_sentence_length=None):
        df = pd.read_csv(self.training_data_path, header=None)

        # word split
        sentences = df[0].apply(lambda x: jieba.lcut(x, cut_all=False))
        sentences, max_sentence_length = self.padding_sentences(sentences, max_sentence_length)

        # embedding words in sentences
        sentences = self.embedding_sentences(sentences)
        training_x = np.array(sentences)

        # set y for training format [0, 1, 0, 0, ]...
        training_y = df[1].apply(self.process_intent)
        training_y = np.array(training_y.tolist())

        return [training_x, training_y], max_sentence_length

    def process_intent(self, intent):
        processed_y = [0] * self.intent_size
        processed_y[self.intent_mapping[intent] - 1] = 1
        return processed_y

    def padding_sentences(self, sentences, padding_sentence_length=None):
        max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
            [len(sentence) for sentence in sentences])
        for sentence in sentences:
            if len(sentence) > max_sentence_length:
                sentence = sentence[:max_sentence_length]
            else:
                sentence.extend([self.padding_token] * (max_sentence_length - len(sentence)))
        return sentences, max_sentence_length

    def embedding_sentences(self, sentences):
        all_vectors = []
        embedding_unknown = [0 for i in range(self.word2vec_model.vector_size)]
        for sentence in sentences:
            this_vector = []
            for word in sentence:
                if word in self.word2vec_model.wv.vocab:
                    this_vector.append(self.word2vec_model[word])
                else:
                    this_vector.append(embedding_unknown)
            all_vectors.append(this_vector)
        return all_vectors


def test():
    data_loader = DataLoader('./data/train_data.csv')
    print(data_loader.load())


if __name__ == '__main__':
    test()
