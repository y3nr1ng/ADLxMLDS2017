"""
Modified list structure to store dynamic enlisted frames.
"""
class Frames(list):
    def __setitem__(self, index, value):
        if index >= len(self):
            for _ in range(index-len(self)+1):
                self.append(None)
        super().__setitem__(index, value)
