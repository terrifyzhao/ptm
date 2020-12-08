import numpy as np

data = np.random.random(10)


def data_loader():
    batch = []
    while 1:
        for d in data:
            batch.append(d)
            if len(batch) == 5:
                yield batch
                batch = []


if __name__ == '__main__':
    g = data_loader()
    print(next(g))
    print(next(g))
    print(next(g))
