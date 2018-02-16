import opfython

if __name__ == "__main__":
    d = opfython.core.dataset.Dataset(n_samples=10, n_classes=2, n_features=3)
    print(d.sample)