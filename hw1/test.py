from timit import reader, preprocess

samples, lut = reader.loadTIMIT('data', 'train', force=True)

for key, value in samples.items():
    print(key)
    print(value)

    break

"""
for instance, frames in labels.items():
    # remove consecutive frames
    frames = preprocess.trim(frames)
    # convert to character notation
    frames = [lut[label] for label in frames]
    labels[instance] = frames

    break
"""

#labels[:] = [preprocess.trim(x, lut) for x in labels]
#print(labels)
