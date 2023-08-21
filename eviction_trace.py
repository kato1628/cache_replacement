import json
from cache import EvictionEntry


class EvictionTrace(object):
    """Ordered set of accesses with information about which lines were evicted.
    
    Serialization details:
        Files are written as JSON format, where each line is a separate JSON object
        with representing a single entry. Everything is written in hex.
    """

    def __init__(self, filename: str, read_only:bool=True):
        """Reads / writes cache state + cache decision information from a file.
        
        Args:
            filename (str): path to eviction trace file to read / write.
            read_only (bool): the trace is either read only or write only.
            If read_only is True, it reads from the provided filename via the read
            method. If read_only is False, it writes to the provided filename via
            the write method.
        """
        self._filename = filename
        self._read_only = read_only

    def __enter__(self):
        if self._read_only:
            self._file = open(self._filename, "r")
        else:
            self._file = open(self._filename, "w")
        return self

    def read(self) -> EvictionEntry:
        # entry = json.loads(next(self._file))
        pass
    
    def write(self, entry: EvictionEntry):
        """Writes an eviction entry to the file.
        
        Args:
            entry (EvictionEntry): an eviction entry to write to the file.
        """
        json.dump({
            "time": entry.cache_state.cache_access.time,
            "obj_id": entry.cache_state.cache_access.obj_id,
            "obj_size": entry.cache_state.cache_access.obj_size,
            "obj_type": entry.cache_state.cache_access.obj_type,
            "cache_lines": [hex(obj_id) for obj_id in entry.cache_state.cache_lines],
            "cache_history": [(hex(access.time), hex(access.obj_id), hex(access.obj_size), hex(access.obj_type))
                                for access in entry.cache_state.cache_history],
            "evict": entry.cache_decision.evict,
            # Serialize items instead of dict to preven json from converting
            # int keys to strings.
            "cache_line_scores": list(entry.cache_decision.cache_line_scores.items()),
        }, self._file)
        self._file.write("\n")

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
