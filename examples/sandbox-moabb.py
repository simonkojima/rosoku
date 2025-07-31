import moabb.datasets

from moabb.datasets import Dreyer2023


dataset = moabb.datasets.Dreyer2023()
print(dataset.subject_list)

# dataset.subejct_list = [56]

sessions = dataset.get_data(subjects=[56])

# raw = sessions[56]["0"]["0R1acquisition"]

raws = sessions[56]["0"]
print(raws)
