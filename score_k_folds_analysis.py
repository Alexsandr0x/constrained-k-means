import numpy as np
import matplotlib.pyplot as plt

import csv

from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
import __init__ as ck
import generate_constraints_link as generate_constraints_link

random_state = 170

K_FOLD = 5
ITERATIONS = 10
LINKS_VALUES = [2, 3, 5, 7, 11, 13, 15, 20, 23, 27, 31, 33, 35, 40, 43, 45, 50]

for dataset_label, dataset, k in generate_constraints_link.datasets:
    print 'RUNNING FOR {}'.format(dataset_label)
    X = dataset.data
    y = dataset.target
    link_size_list = [a for a in LINKS_VALUES if a < y.shape[0]/10 + 1]

    acc_avg_k_folds = 0
    serie_score_k_folds = []
    serie_score = []
    for link_size in link_size_list:
        generate_constraints_link.generate(link_array_size=link_size)
        links = np.load(dataset_label + '.npy').item()
        acc_avg_k_folds = 0
        acc_avg = 0
        for iter in range(ITERATIONS):
            rand_avg_sum_k_folds = 0
            skf = StratifiedKFold(n_splits=K_FOLD)
            count_test = 0
            for train, test in skf.split(X, y):
                count_test += 1
                print 'init k_fold for link size {} and k_fold n:{}...'.format(link_size, count_test)
                train_X = np.array([X[index] for index in train])
                train_y = np.array([y[index] for index in train])
                test_X = np.array([X[index] for index in test])
                test_y = np.array([y[index] for index in test])
                clf = ck.ConstrainedKMeans(n_clusters=k)
                clf.fit(train_X, train_y, **links)
                predicted = clf.predict(test_X)
                rand_avg_sum_k_folds += adjusted_rand_score(test_y, predicted)
                print 'finish k_fold for link size {} and k_fold n:{}'.format(link_size, count_test)
            print 'init CKM for link size {}...'.format(link_size)
            clf = ck.ConstrainedKMeans(n_clusters=k)
            clf.fit(X, y, **links)
            acc_avg += adjusted_rand_score(y, clf.labels_)
            rand_avg_sum_k_folds /= K_FOLD
            acc_avg_k_folds += rand_avg_sum_k_folds
            print 'finish CKM for link size {}'.format(link_size)
        acc_avg_k_folds /= ITERATIONS
        acc_avg /= ITERATIONS
        print "Overall: Link Size {} with {} clusters: {}".format(link_size * 2, k, acc_avg)
        print "test-set: Link Size {} with {} clusters: {}".format(link_size * 2, k, acc_avg_k_folds)
        serie_score_k_folds.append(acc_avg_k_folds)
        serie_score.append(acc_avg)

    with open('results_{}.csv'.format(dataset_label), 'wb') as csvfile:
        spam_writer = csv.writer(csvfile, delimiter=',')
        spam_writer.writerow(link_size_list)
        spam_writer.writerow(serie_score_k_folds)
        spam_writer.writerow(serie_score)

    plt.plot(link_size_list, serie_score_k_folds, label='Held Out test-set')
    plt.plot(link_size_list, serie_score, label='overall test')

    plt.xlabel("Link size")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.show()
