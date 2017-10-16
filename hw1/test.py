from timit import reader
from timeit import default_timer as timer

# load the dataset
dataset = reader.TIMIT('data')

start = timer()
dataset.load('train')
end = timer()
print('Data loaded in {0:.3f}s'.format(end-start))
print()

# preview
print(dataset.dump(5))

# start training
