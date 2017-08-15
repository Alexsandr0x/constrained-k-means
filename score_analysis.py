# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
import __init__ as ck
import generate_constraints_link as generate_constraints_link

random_state = 170

for dataset_label, dataset, _ in generate_constraints_link.datasets:
    X = dataset.data
    y = dataset.target

    clusters = [2, 3, 5, 7, 11, 13]
    serie_score = []
    for n_cluster in clusters:
        y_pred = KMeans(n_clusters=n_cluster, random_state=random_state).fit_predict(X)
        rand_avg = adjusted_rand_score(y, y_pred)
        print "KMeans Classic with {} clusters: {}".format(n_cluster, rand_avg)
        serie_score.append(rand_avg)
    plt.plot(clusters, serie_score, label='KMeans', linewidth=2.0)

    for link_size in [5, 10, 15]:
        serie_score = []
        generate_constraints_link.generate(link_array_size=link_size)
        links = np.load(dataset_label + '.npy').item()
        for n_cluster in clusters:
            clf = ck.ConstrainedKMeans(n_clusters=n_cluster)

            clf.fit(X, y, **links)

            rand_avg = adjusted_rand_score(y, clf.labels_)
            serie_score.append(rand_avg)
            print "Link Size {} with {} clusters: {}".format(link_size, n_cluster, rand_avg)
        plt.plot(clusters, serie_score, label="LinkSize " + str(link_size))
    plt.xlabel("Clusters")
    plt.ylabel("Rand Ajustado")
    plt.legend(loc="upper right")
    plt.show()
