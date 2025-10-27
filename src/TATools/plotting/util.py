class CyclicList(list):
    def __init__(self, data):
        if isinstance(data, (str, bytes)):
            # treat strings and bytes as single items, not iterables
            data = [data]
        elif not hasattr(data, "__iter__"):
            # wrap non-iterables into a list
            data = [data]
        super().__init__(data)

    def __getitem__(self, i):
        if not self:
            raise IndexError("Cannot index into empty CyclicList")
        if isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            rng = range(start, stop, step)
            return [super().__getitem__(j % len(self)) for j in rng]
        return super().__getitem__(i % len(self))
