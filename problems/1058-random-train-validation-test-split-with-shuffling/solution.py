import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    data_list, n=[], data.shape[0]
    ordering = np.random.default_rng(seed).permutation(n)
    train_end = int(n*train_frac)
    validation_end = train_end + int(n * validation_frac)
    training_data = data[ordering[:train_end].tolist()]
    validation_data = data[ordering[train_end:validation_end].tolist()]
    testing_data = data[ordering[validation_end:]]
    data_list.append(training_data)
    data_list.append(validation_data)
    data_list.append(testing_data)
    return data_list