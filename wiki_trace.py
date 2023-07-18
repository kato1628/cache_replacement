from collections import defaultdict, deque
import csv

import numpy as np

class WikiTrace(object):
    
    def __init__(self, filename, max_look_ahead=int(1e7)) -> None:
        self._filename = filename

        # window size the tracer look ahead
        self._max_look_ahead = max_look_ahead

        # maps obj_id to the list of access times in the look ahead buffer
        # e.x. (1: (1, 142, 275), 2: (2, 512, 710), ...)
        self._access_times = defaultdict(deque)

        # deque object to which each line in trace will be stored
        self._look_ahead_buffer = deque()
        self._reader_exhausted = False
        self._num_next_calls = 0

    def _read_next(self) -> None:
        if self._reader_exhausted:
            return

        try: 
            # read trace line and store it to the buffer
            time, obj_id, obj_size, obj_type = self._ssv_reader.next()
            self._look_ahead_buffer.append((time, obj_id, obj_size, obj_type))

            # store the access time of the object
            accessed_at = len(self._look_ahead_buffer) + self._num_next_calls
            self._access_times[obj_id].append(accessed_at)

        except StopIteration:
            self._reader_exhausted = True

    def next(self) -> tuple[int, int, int, int]:
        self._num_next_calls += 1

        time, obj_id, obj_size, obj_type = self._look_ahead_buffer.popleft() 

        self._access_times[obj_id].popleft()
        # memory optimisation: delete keys that don't have any access time
        if not self._access_times[obj_id]:
            del self._access_times[obj_id]

        self._read_next() # read ahead

        return time, obj_id, obj_size, obj_type

    def done(self) -> bool:
        return not self._look_ahead_buffer

    def next_access_time(self, obj_id):
        access_times = self._access_times[obj_id]

        # Returns number of access from current cursor to next access
        if not access_times:
            return np.inf
        return access_times[0] - self._num_next_calls
            
    def __enter__(self):
        self._file = open(self._filename, "r", encoding='latin-1')
        self._ssv_reader = SSVReader(self._file)

        for _ in range(self._max_look_ahead):
            self._read_next()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()


class SSVReader(object):

    def __init__(self, f) -> None:
        self._file = f
        self._tsv_reader = csv.reader(self._file, delimiter=" ")
    
    def next(self) -> tuple[int, int, int, int]:
        time, obj_id, obj_size, obj_type = next(self._tsv_reader)
        return (int(time), int(obj_id), int(obj_size), int(obj_type))