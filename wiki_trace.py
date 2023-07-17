from collections import deque
import csv

class WikiTrace(object):
    
    def __init__(self, filename, max_look_ahead=int(1e7)) -> None:
        self._filename = filename
        self._max_look_ahead = max_look_ahead
        self._look_ahead_buffer = deque()
        self._reader_exhausted = False

    def _read_next(self) -> None:
        if self._reader_exhausted:
            return

        try: 
            time, obj_id, obj_size, obj_type = self._ssv_reader.next()
            self._look_ahead_buffer.append((time, obj_id, obj_size, obj_type))
        except StopIteration:
            self._reader_exhausted = True

    def next(self) -> None:
        self._read_next() # read ahead
        return self._look_ahead_buffer.popleft() 

    def done(self) -> bool:
        return not self._look_ahead_buffer
            
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