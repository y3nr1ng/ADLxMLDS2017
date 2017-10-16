from timit import reader
from timeit import default_timer as timer

dataset = reader.TIMIT('data')

start = timer()
dataset.load('train')
end = timer()
print('Data loaded in {0:.3f}s'.format(end-start))
print()

print(dataset.dump(5))
