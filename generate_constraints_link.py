import numpy as np

from sklearn.datasets import load_digits, load_iris, load_diabetes

LINK_ARRAY_SIZE = 20
datasets = [
    ("iris", load_iris(), 3),
    ("digits", load_digits(), 10),
    ("diabetes", load_diabetes(), 7)
]


def generate(link_array_size=LINK_ARRAY_SIZE):
    for name, data_set, _ in datasets:
        samples = np.random.choice(len(data_set.data), link_array_size)
        must_links = []
        cannot_links = []
        for sample in samples:
            value = data_set.target[sample]
            for selected in range(len(data_set.data)):
                if value == data_set.target[selected]:
                    if sample == selected:
                        continue
                    must_link = [
                        np.asarray(data_set.data[sample]),
                        np.asarray(data_set.data[selected])
                    ]
                    must_links.append(must_link)
                    break
                else:
                    continue

        samples = np.random.choice(len(data_set.data), link_array_size)
        for sample in samples:
            value = data_set.target[sample]
            for selected in range(len(data_set.data)):
                if value != data_set.target[selected]:
                    cannot_link = [
                        np.asarray(data_set.data[sample]),
                        np.asarray(data_set.data[selected])
                    ]
                    cannot_links.append(cannot_link)
                    break
                else:
                    continue

        links = {'must_link': must_links, 'cannot_link': cannot_links}
        np.save(name, links)
