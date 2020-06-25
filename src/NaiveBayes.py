import numpy as np


class NaiveBayes:

    def __init__(self):
        pass

    def model(self, X, y, smoothing, vocab_size, vocabulary):
        """
        The model method is specifically used to generate the parameters required to generate the model.txt file
        :param X: an array of training samples in form of a sparse matrix which contains the frequency of each word in the vocabulary
        :param y: class labels of respective training sample
        :param smoothing: a number(int or float) used to avoid getting a zeroth posterior for a class
        :param vocab_size: size of vocabulary
        :param vocabulary: documents vocabulary
        :return: returns a dictionary that has its key as each word in the vocabulary and value as a list containing
                 the index of the word in the vocabulary, the word itself, the frequency of the word in class ham,
                 the conditional probability of the word in class ham, the frequency of the word in class spam, and
                 the conditional probability of the word in class spam.
        """
        n_samples, n_features = X.shape
        _classes = np.unique(y)
        word_conditional_prob = {}
        _priors = np.zeros(len(_classes), dtype=np.float64)

        for c in _classes:
            word_conditional_prob[c] = np.zeros(n_features)
            X_c = X[c == y]  # get all rows where the label is same as class c
            _priors[int(c)] = X_c.shape[0] / float(n_samples)  # probability of each class
            word_conditional_prob[c] = (np.sum(X_c, axis=0) + smoothing) / (np.sum(X_c) + (smoothing * vocab_size))

            if int(c) == 0:
                word_conditional_prob_spam = list(np.sum(X_c, axis=0) + smoothing) / (np.sum(X_c) + (smoothing * vocab_size))
                freq_spam = (np.sum(X_c, axis=0) + smoothing)
            elif int(c) == 1:
                word_conditional_prob_ham = list(np.sum(X_c, axis=0) + smoothing) / (np.sum(X_c) + (smoothing * vocab_size))
                freq_ham = np.sum(X_c, axis=0) + smoothing

        # word  freq_wi_in_ham  p(wi|ham)  freq_wi_spam  p(wi|spam)
        util = {}
        for idx, word in enumerate(vocabulary):
            if len(word) > 0:
                util[word] = str(idx+1) + "  " + str(word) + "  " + str(round(freq_ham[idx], 8)) + "  " + str(
                    round(word_conditional_prob_ham[idx], 8)) + "  " + str(round(freq_spam[idx], 8)) + "  " + str(
                    round(word_conditional_prob_spam[idx], 8))+"\n"

        return util

    def fit(self, X, y, smoothing, vocab_size):
        """
        The fit method is used to train a training samples which can be used for prediction
        :param X: an array of training samples in form of a sparse matrix which contains the frequency of each word in the vocabulary
        :param y: class labels of respective training sample
        :param smoothing: a number(int or float) used to avoid getting a zeroth posterior for a class
        :param vocab_size: size of vocabulary
        :return: no return type
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        self.word_conditional_prob = {}
        self._priors = np.zeros(len(self._classes), dtype=np.float64)

        # for each word, compute their frequencies and probabilities for each class (class ham
        # and class spam)
        for c in self._classes:
            self.word_conditional_prob[c] = np.zeros(n_features)
            X_c = X[c == y]  # get all rows where the label is same as class c
            self._priors[int(c)] = X_c.shape[0] / float(n_samples)  # probability of each class
            self.word_conditional_prob[c] = (np.sum(X_c, axis=0) + smoothing) / (np.sum(X_c) + (smoothing * vocab_size))  # conditional probability for each word in the vocabulary

    def predict(self, X):
        """
        The prdict method is used to make predictions for test data
        :param X: an array of test samples in form of a sparse matrix which contains the frequency of each word in the vocabulary
        :return: returns a list of predicted classes and the posteriors used in class prediction
        """
        # predicted_classes = [self.predict_(x) for x in X]
        predicted_classes = []
        posteriors = []
        for x in X:
            predicted_class, post = self.predict_(x)
            predicted_classes.append(predicted_class)
            posteriors.append(post)
        return predicted_classes, posteriors

    def predict_(self, x):
        """
        A utility function used by the predict function to compute the priors of each classes, the conditional
        probabilities of each word in the test samples and the posteriors(score) of each class
        :param x: a sparse matrix of the words in the test data
        :return: returns the class of a predicted posterior and a list of the posteriors of each class classes
        """
        posteriors = []
        for c in self._classes:
            prior = np.log(self._priors[int(c)])
            word_conditional_prob = self.word_conditional_prob[c]
            class_conditional_probs = self.conditional_prob(x, word_conditional_prob)
            posteriors.append(prior + class_conditional_probs)
        return self._classes[np.argmax(posteriors)], posteriors  # return class with maximum posterior (0/1)

    def conditional_prob(self, x, word_conditional_prob):
        """
        This method is used to compute the conditional probability of each word in the test sample
        :param x: a sparse matrix of the words in the test data
        :param word_conditional_prob: computed conditional probability of each word in x
        :return: returns
        """
        product = np.multiply(x, word_conditional_prob)
        return np.sum(np.log(product[product > 0]))  # return the sum of log of numbers greater than 0
