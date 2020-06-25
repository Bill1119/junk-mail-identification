import numpy as np


class CountVectorizer:
    """

    """
    def fit(self, corpus):
        self.vocabulary = {}
        for sentence in corpus:
            # for word in sentence.split(" "):
            for word in sentence:
                if word not in self.vocabulary.keys():
                    self.vocabulary[word] = 1
                else:
                    self.vocabulary[word] += 1
        self.vocabulary_count = len(self.vocabulary)  # this approach was used to avoid re-computing the valuse
        self.sorted_vocabulary_keys = list(sorted(self.vocabulary))

    def test(self):
        v = {}
        params = self.get_feature_params()
        for word in params:
            v[word] = params.index(word)
        return v

    def get_feature_params(self):
        return self.sorted_vocabulary_keys

    def get_vocabulary_keys(self):
        return self.vocabulary.keys()

    def vocabulary_(self):
        return self.vocabulary

    def get_vocabulary_count(self):
        return self.vocabulary_count

    def fit_transform(self, samples):

        all_dict = []
        vocab = self.get_feature_params()
        for row in samples:
            dic = {}
            for word in vocab:
                dic[word] = 0

            for word in row:
                if word in dic.keys():
                    dic[word] += 1
            all_dict.append(list(dic.values()))
        return np.asarray(all_dict)
