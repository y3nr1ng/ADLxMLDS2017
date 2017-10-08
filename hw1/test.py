from timit import reader

dataset = reader.TIMIT()


temp = dataset.load('data', 'train')
print(temp.head(25))
