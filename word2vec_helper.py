from gensim.models import KeyedVectors


def load():
    return KeyedVectors.load_word2vec_format('./data/news12g_bdbk20g_nov90g_dim64.bin', binary=True)


def test():
    model = load()
    print(model.most_similar(['男人']))
    print(model.most_similar(positive=['女人', '国王'], negative=['男人'], topn=2))


if __name__ == '__main__':
    test()
