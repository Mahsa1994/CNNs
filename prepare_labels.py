import pandas as pd


# Extract class labels list from the train CSV
def extract_class_labels_list_from_csv(address_train_csv_file):
    # read the Train CSV file:
    df = pd.read_csv(address_train_csv_file)
    Y = df['tags']
    number_of_train_images = len(Y)
    results_list = []

    # Iterate over all Train elements:
    class_labels_set = set()  # empty set to store class labels
    for i in range(number_of_train_images):
        before_len = len(class_labels_set)
        class_labels_set.add(Y[i])
        after_len = len(class_labels_set)
        if before_len != after_len:
            results_list.append(Y[i])

    return results_list


# Extract class labels occurrences (label distributions) from the train CSV
def extract_train_labels_distribution(address_train_csv_file, listLabels):
    number_of_total_labels = len(listLabels)
    distributions_list = [0] * number_of_total_labels

    # read the Train CSV file:
    df = pd.read_csv(address_train_csv_file)
    Y = df['tags']

    number_of_train_images = len(Y)
    # iterate over all Train elements:
    for i in range(number_of_train_images):
        distributions_list[listLabels.index(Y[i])] += 1

    return distributions_list



