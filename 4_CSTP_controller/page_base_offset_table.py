from collections import deque

class Page_Base_Offset_Table:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.dict = {}

    def __setitem__(self, key, value):
        if key in self.dict:
            self.queue.remove(key)
        elif len(self.dict) == self.capacity:
            oldest_key = self.queue.popleft()
            del self.dict[oldest_key]
        self.queue.append(key)
        self.dict[key] = value

    def __getitem__(self, key):
        return self.dict[key]

    def __delitem__(self, key):
        self.queue.remove(key)
        del self.dict[key]

    def __len__(self):
        return len(self.dict)

    def __repr__(self):
        return str(self.dict)
