import os
import struct
import itertools


TYPEOF_STEP = 'i'
TYPEOF_VAL = 'd'
fmt = TYPEOF_VAL + TYPEOF_STEP
size = struct.calcsize(fmt)


class MetricsBase(object):
    def __init__(self, root):
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def _filename(self, key):
        return os.path.join(self.root, key + '.metric')


class MetricsWriter(MetricsBase):
    def write(self, key, val, step):
        with open(self._filename(key), 'ab') as f:
            f.write(struct.pack(fmt, val, step))


class MetricsReader(MetricsBase):
    def read(self, key, offset=None, limit=None):
        table = []
        with open(self._filename(key), 'r+b') as f:

            # Optionally skip.
            if offset:
                f.seek(offset * size)

            for ii, _ in enumerate(itertools.repeat(None)):
                # Terminate if limit.
                if limit and not ii < limit:
                    break

                inp = f.read(size)

                # Terminate if EOF.
                if not inp:
                    break

                row = struct.unpack(fmt, inp)
                table.append(row)
        return table
