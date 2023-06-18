import numpy as np

def true_positive(y_true, y_pred):
    """
    Function to compute the true positive
    Arguments should be array-like
    """
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1

    return tp
def true_negative(y_true, y_pred):
    """
    Function to compute the true negative
    Arguments should be `array-like`
    """
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1

    return tn

def false_positive(y_true, y_pred):
    """
    Function to compute the false positive
    Arguments should be `array-like`
    """
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1

    return fp

def false_negative(y_true, y_pred):
    """
    Function to compute the false negative
    Arguments should be `array-like`
    """
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1

    return fn

def micro_recall(y_true, y_pred):
    """
    Function to compute Micro-averaged Recall OvR (One-Over-Rest)
    """
    # number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fn = 0
    y_true_arr = np.array(y_true)
    # loop over all classes
    for class_ in np.unique(y_true):

        # Treat all classes except current as negative class
        temp_true = [1 if p == class_ else 0 for p in y_true_arr]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # compute false negative for current class
        # and update overall tp
        fn += false_negative(temp_true, temp_pred)

    # calculate and return overall recall
    recall = tp / (tp + fn)
    return recall

def micro_precision(y_true, y_pred):
    """
    Function to compute the Micro-averaged Precision OvR (One-Over-Rest)
    :param y_true:
    :param y_pred:
    :return:
    """
    # number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0
    y_true_arr = np.array(y_true)
    # loop over all classes
    for class_ in np.unique(y_true):

        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true_arr]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # compute true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)

        # compute false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)

    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision

def micro_f1(y_true, y_pred):
    """
    Function to compute the Micro-averaged F1 score OvR (One-Over-Rest)
    """
    # micro-averaged precision score

    p = micro_precision(y_true, y_pred)

    # micro-averaged recall score
    r = micro_recall(y_true, y_pred)

    # micro-averaged f1 score
    f1 = 2*p*r / (p + r)

    return f1

def macro_f1(y_true, y_pred):
    """
    Function to compute the Macro-averaged F1 score OvR (One-Over-Rest)
    """
    # number of classes
    num_classes = len(np.unique(y_true))

    # initialize f1 to 0
    f1 = 0
    y_true_arr = np.array(y_true)
    # loop over all classes
    for c in list(np.unique(y_true)):

        # Treat all classes except current as negative class
        temp_true = [1 if p == c else 0 for p in y_true_arr]
        temp_pred = [1 if p == c else 0 for p in y_pred]

        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # compute false negative for current class
        fn = false_negative(temp_true, temp_pred)

        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)


        # compute recall for current class
        temp_recall = tp / (tp + fn + 1e-6)

        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)

        # f1 score formula
        temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)

        # Accumulating f1 score for all classes
        f1 += temp_f1

    # averaging f1 score over all classes
    f1 /= num_classes

    return f1
