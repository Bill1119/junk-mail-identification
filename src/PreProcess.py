import re


class PreProcess:

    def __init__(self):
        pass

    def clean_sentence(self, string):
        """
        clean_sentence method takes all letters to small letters, remove multiple spaces and remove all special characters from doc.
        :param string:
        :return:
        """
        cleaned_str = string.lower()
        cleaned_str = re.split('[^a-zA-Z]', cleaned_str)
        cleaned_str = [string for string in cleaned_str if string != ""]
        return cleaned_str

    def pre_process(self, corpus):
        """
        pre-process methos is used to clean the documents with the help of a util method clean_sentence()
        :param corpus: The document to be cleaned
        :return: returns cleaned document with every later converted to small letters, multiple spaces deleted and special characters are dropped.
        """
        cleaned_cor = [self.clean_sentence(sentence) for sentence in corpus]
        return cleaned_cor

    def join(self, t1, t2):
        """
        :param t1: document 1
        :param t2: document 2
        :return: returns a combined document of t1 and t2
        """
        train_data = []
        for sent in t1:
            train_data.append(sent)

        for sent in t2:
            train_data.append(sent)
        return train_data
