from torch.utils.data import Dataset


class Batch(Dataset):
    def __init__(self, *, name, elements):
        count_elements = len(elements)

        self.count_elements = count_elements
        self.elements = elements

        self.iterator = 0
        self.iterations = self.count_elements
        self.name = name

    def __iter__(self): return self

    def __getitem__(self, index):
        return self.elements[index]

    def __next__(self):
        if self.iterator >= self.count_elements:
            raise StopIteration

        items = self.__getitem__(self.iterator)
        self.iterator = self.iterator + 1

        return items

    def __len__(self): return self.iterations