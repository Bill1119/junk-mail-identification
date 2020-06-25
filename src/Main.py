import numpy as np
from src.CountVectorizer import CountVectorizer  # user defined CountVectorizer class to generate sparse matrices
from src.PreProcess import PreProcess  # user define PreProcess class to help pre-process each document
from src.NaiveBayes import NaiveBayes  # user defined Naive Bayes classifier


def accuracy(y_true, y_pred, class_type):
    """
    Method to compute the accuracies of classes ham and spam
    :param y_true: expected class labels
    :param y_pred: predicted class labels by Naive Bayes classifier
    :param class_type: either ham or spam
    :return: return the computer accuracy = TP / (TP + TN + FP + FN)
    """
    true_positive, false_positive, true_negative, false_negative = parameters(y_true, y_pred, class_type)
    return (true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)


def _params(y_true, y_pred):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for idx in range(len(y_true)):
        if y_true[idx] == 1 and y_pred[idx] == 1:
            true_positive += 1
        elif y_true[idx] == 1 and y_pred[idx] == 0:
            false_negative += 1
        elif y_true[idx] == 0 and y_pred[idx] == 1:
            false_positive += 1
        elif y_true[idx] == 0 and y_pred[idx] == 0:
            true_negative += 1
    return true_positive, false_positive, true_negative, false_negative


def parameters(y_true, y_pred, class_type):
    """
    :param y_true: expected class label
    :param y_pred: predicted class label
    :param class_type: if class is spam: positive=0, negative=1, if class is ham: positive=1, negative=0
    :return: returns TP, TN, FP, and FN values
    """
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    if class_type == 'spam':
        for index in range(len(y_pred)):
            if y_true[index] == 0 and y_pred[index] == 0:
                true_positive += 1
            elif y_true[index] == 1 and y_pred[index] == 0:
                false_positive += 1
            elif y_true[index] == 0 and y_pred[index] == 1:
                false_negative += 1
            elif y_true[index] == 1 and y_pred[index] == 1:
                true_negative += 1
    elif class_type == 'ham' or class_type == 'joint':
        for index in range(len(y_true)):
            if y_true[index] == 1 and y_pred[index] == 1:
                true_positive += 1
            elif y_true[index] == 0 and y_pred[index] == 1:
                false_positive += 1
            elif y_true[index] == 1 and y_pred[index] == 0:
                false_negative += 1
            elif y_true[index] == 0 and y_pred[index] == 0:
                true_negative += 1
    return true_positive, false_positive, true_negative, false_negative


def confusion_matrix_per_class(y_true, y_pred, class_type):
    """
    computes a 2 by 2 matrix with actual and predicted class labels
    :param y_true: expected class label
    :param y_pred: predicted class label
    :param class_type: if class is spam: positive=0, negative=1, if class is ham: positive=1, negative=0
    :return: returns 2 by 2 confusion matrix
    """
    matrix = np.zeros((2, 2))
    true_positive, false_positive, true_negative, false_negative = parameters(y_true, y_pred, class_type)
    matrix[0, 0] = true_positive
    matrix[0, 1] = false_positive
    matrix[1, 0] = false_negative
    matrix[1, 1] = true_negative
    return matrix


def confusion_matrix(y_true, y_pred):
    """
    computes a 2 by 2 matrix with actual and predicted class labels
    :param y_true: expected class label
    :param y_pred: predicted class label
    :param class_type: if class is spam: positive=0, negative=1, if class is ham: positive=1, negative=0
    :return: returns 2 by 2 confusion matrix
    """
    matrix = np.zeros((2, 2))
    true_positive, false_positive, true_negative, false_negative = _params(y_true, y_pred)
    matrix[0, 0] = true_positive
    matrix[0, 1] = false_positive
    matrix[1, 0] = false_negative
    matrix[1, 1] = true_negative
    return matrix


def precision(y_true, y_pred, class_type):
    """
    computes the precision of a class
    :param y_true: expected class label
    :param y_pred: predicted class label
    :param class_type: spam or ham
    :return: returns precision value; TP / (TP + FP)
    """
    true_positive, false_positive, true_negative, false_negative = parameters(y_true, y_pred, class_type)
    return true_positive / (true_positive + false_positive)


def recall(y_true, y_pred, class_type):
    """
    computes the recall of a class
    :param y_true: expected class label
    :param y_pred: predicted class label
    :param class_type: spam or ham
    :return: returns precision value; TP / (TP + FN)
    """
    true_positive, false_positive, true_negative, false_negative = parameters(y_true, y_pred, class_type)
    return true_positive / (true_positive + false_negative)


def f_measure(y_true, y_pred, class_type):
    """
    computes the f1-measure of a class
    :param y_true: expected class label
    :param y_pred: predicted class label
    :param class_type: spam or ham
    :return: returns the f-measure value; (2 * P * R) / (P + R)
    """
    p = precision(y_true, y_pred, class_type)
    r = recall(y_true, y_pred, class_type)
    return (2 * p * r) / (p + r)


def read_files(file_type, class_type, files_count):
    """
    read all train/test files from dataset
    :param file_type: test or train
    :param class_type: spam or ham
    :param files_count: number of files to be read
    :return: returns read files
    """
    contents = []
    for i in range(1, files_count+1):
        file_number = class_type+'-'+str(i).zfill(5)
        file_name = '../datasets/'+file_type+'/'+file_type+'-'+file_number+'.txt'
        f = open(file_name, errors='ignore').read()
        contents.append(str(f))
    return contents


def labels(spam, ham):
    """
    compute the labels of each class from read files
    :param spam: spam class
    :param ham: ham class
    :return: returns computed labels for spam and ham classes
    """
    sample_labels = []
    for i in range(spam):
        sample_labels.append(0)
    for i in range(ham):
        sample_labels.append(1)
    return sample_labels


def save_to_file(model):
    """
    saves model with all necessary information to a model.txt file
    :param model: a dictionary that has its key as each word in the vocabulary and value as a list containing
                 the index of the word in the vocabulary, the word itself, the frequency of the word in class ham,
                 the conditional probability of the word in class ham, the frequency of the word in class spam, and
                 the conditional probability of the word in class spam,
    :return: no return type
    """
    file = open('../output_files/model.txt', 'w')

    for key in model.keys():
        file.writelines(model[key])

    file.close()


def save_results(test_labels, pred, posteriors):
    """

    :param test_labels: actual class labels
    :param pred: predicted class labels
    :param posteriors: posteriors used for predictions
    :return: no return type
    """
    file = open('../output_files/result.txt', 'w')
    # print('len(pred)', len(pred), 'len(test_labels)', len(test_labels))
    for i in range(len(pred)):
        class_type = ''
        pred_label = ''
        error = ''
        if int(test_labels[i]) == 0:
            class_type = 'spam'
        else:
            if int(test_labels[i]) == 1:
                class_type = 'ham'
        file_name = 'test-'+class_type+'-'+str(i+1).zfill(5)+'.txt'
        if int(pred[i]) == 0:
            pred_label = 'spam'
        else:
            if int(pred[i]) == 1:
                pred_label = 'ham'
        if class_type == pred_label:
            error = 'right'
        elif class_type != pred_label:
            error = 'wrong'
        pred_score_spam = posteriors[i][0]
        pred_score_ham = posteriors[i][1]
        file.writelines(str(file_name)+"  "+str(pred_label)+"  "+str(pred_score_ham)+"  "+str(pred_score_spam)+"  "+str(class_type)+"  "+error+"\n")
    file.close()


def get_class_data(test_labels, pred):
    """
    seperate spam from ham
    :param test_labels: actual class labels
    :param pred: predicted class labels
    :return: returns separated samples with their respective labels
    """
    spam_test_labels = []
    ham_test_labels = []
    spam_pred = []
    ham_pred = []

    for idx, c in enumerate(test_labels):
        if c == 0:
            spam_test_labels.append(c)
            spam_pred.append(pred[idx])
        elif c == 1:
            ham_test_labels.append(c)
            ham_pred.append(pred[idx])
    return spam_test_labels, spam_pred, ham_test_labels, ham_pred


def main():
    """
    main program
    :return: no return type
    """
    # read training data, labels
    print('---------------------------------------------------- Reading train/test data from documents -------------------------------------------------------')
    train_spam = read_files("train", "spam", 997)
    train_ham = read_files("train", "ham", 1000)
    train_labels = labels(len(train_spam), len(train_ham))
    #
    # # read test data, labels
    test_spam = read_files("test", "spam", 400)
    test_ham = read_files("test", "ham", 400)
    test_labels = labels(len(test_spam), len(test_ham))
    print('Data read successfully!!! \n')
    # # Pre-process strings
    print('---------------------------------------------------- Pre-processing the train/test data -----------------------------------------------------------')
    p = PreProcess()
    train_spam = p.pre_process(train_spam)
    train_ham = p.pre_process(train_ham)
    test_spam = p.pre_process(test_spam)
    test_ham = p.pre_process(test_ham)
    #
    # # join training samples
    train_data = p.join(train_spam, train_ham)
    #
    # # join test samples
    test_data = p.join(test_spam, test_ham)
    print('Data pre-processing successful!!! \n')
    # # transform data to sparse matrix
    print('-------------------------------------------------- Computing sparse matrices for train/test data --------------------------------------------------')
    print('------------------------------------- PLEASE NOTE: This will take approximately 2-3 minutes (Room for improvement!) -------------------------------')
    vectorizer = CountVectorizer()
    vectorizer.fit(train_data)
    train_data = vectorizer.fit_transform(train_data)
    test_data = vectorizer.fit_transform(test_data)
    print('Sparse matrices has been computed successfully!!! \n')
    # # training and prediction
    nb = NaiveBayes()
    model = nb.model(train_data, train_labels, 0.5, vectorizer.get_vocabulary_count(), vectorizer.get_feature_params())
    print('***************************************************************** Saving Model *****************************************************************')
    save_to_file(model)
    print('model.txt file has been saved successfully!!! Check generated output in the output_files folder. \n')
    print('-------------------------------------------------------------- Training started -----------------------------------------------------------------')
    nb.fit(train_data, train_labels, 0.5, vectorizer.get_vocabulary_count())
    print('Training completed successfully!!! \n')
    print('------------------------------------------------------------------- Predicting -----------------------------------------------------------------')
    pred, posteriors = nb.predict(test_data)
    print('Prediction completed successfully!!! \n')
    print('***************************************************************** Saving Result *****************************************************************')
    save_results(test_labels, pred, posteriors)
    print('result.txt file has been saved successfully!!! Check generated output in the output_files folder. \n')

    print('----------------------------------------------------------------- Prediction Started ------------------------------------------------------------')
    spam_test_labels, spam_pred, ham_test_labels, ham_pred = get_class_data(test_labels, pred)
    print('Prediction is successful!!! \n')
    print('------------------------------------------------------------- Performance for class spam --------------------------------------------------------')

    # performance for class spam
    print('accuracy for class spam : ', accuracy(spam_test_labels, spam_pred, class_type='spam'))
    print('precision for class spam : ', precision(spam_test_labels, spam_pred, class_type='spam'))
    print('recall for class spam : ', recall(spam_test_labels, spam_pred, class_type='spam'))
    print('f-measure for class spam : ', f_measure(spam_test_labels, spam_pred, class_type='spam'))
    print('confusion matrix for class spam : \n', confusion_matrix_per_class(spam_test_labels, spam_pred, class_type='spam'), '\n')
    print('------------------------------------------------------------- Performance for class ham ---------------------------------------------------------')

    # performance for class ham
    print('accuracy for class ham : ', accuracy(ham_test_labels, ham_pred, class_type='ham'))
    print('precision for class ham : ', precision(ham_test_labels, ham_pred, class_type='ham'))
    print('recall for class ham : ', recall(ham_test_labels, ham_pred, class_type='ham'))
    print('f-measure for class ham : ', f_measure(ham_test_labels, ham_pred, class_type='ham'))
    print('confusion matrix for class ham : \n', confusion_matrix_per_class(ham_test_labels, ham_pred, class_type='ham'), '\n')

    print('--------------------------------------------------------------- Overall system performance ----------------------------------------------------------------')

    print('accuracy : ', accuracy(test_labels, pred, class_type='joint'))
    print('precision : ', precision(test_labels, pred, class_type='joint'))
    print('recall : ', recall(test_labels, pred, class_type='joint'))
    print('f-measure : ', f_measure(test_labels, pred, class_type='joint'))
    print('confusion matrix : \n', confusion_matrix(test_labels, pred), '\n')


if __name__ == "__main__":
    main()
