from timit import reader

dataset = reader.TIMIT()

dataset.load('data', 'train')
print(dataset.dump(5))
