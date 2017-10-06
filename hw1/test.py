from timit import reader, preprocess

lut = reader.loadMap('data')
labels = reader.loadLabel('data', 'train')


for instance, frames in labels.items():
    # remove consecutive frames
    frames = preprocess.trim(frames)
    # convert to character notation
    frames = [lut[label] for label in frames]
    labels[instance] = frames

#labels[:] = [preprocess.trim(x, lut) for x in labels]
#print(labels)
