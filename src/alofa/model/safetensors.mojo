"""Single-file safetensors metadata reader with a read-only mapping."""

from alofa.core.dtype import DT_BF16, DT_FP16, DT_FP32
from alofa.core.error import ERR_INVALID_ARGUMENT, ERR_OUT_OF_RANGE, ERR_PARSE, ERR_UNSUPPORTED, AlofaError
from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr, Shape, TensorView, f32_data
from alofa.core.text import parse_int


def _text_slice(raw: String, start: Int, end: Int) -> String:
    var bytes = List[UInt8]()
    var source = raw.as_bytes()
    for i in range(start, end):
        bytes.append(source[i])
    return String(unsafe_from_utf8=bytes)


def _u64_le(file: MappedFile, offset: Int) raises AlofaError -> Int:
    """Read a little-endian unsigned 64-bit value bounded by the file."""
    if offset < 0 or offset + 8 > file.size:
        raise AlofaError(ERR_OUT_OF_RANGE, "safetensors header length is truncated", "offset=" + String(offset))
    var out = 0
    for i in range(8):
        out += Int(file.byte_at(offset + i)) << (8 * i)
    return out


def _field(text: String, key: String) raises AlofaError -> String:
    """Read one quoted metadata field from a safetensors tensor object."""
    var raw = text.as_bytes()
    var needle = key.as_bytes()
    var i = 0
    while i + len(needle) + 2 <= len(raw):
        if raw[i] == 34:
            var matches = True
            for j in range(len(needle)):
                if raw[i + 1 + j] != needle[j]:
                    matches = False
            if matches and raw[i + 1 + len(needle)] == 34:
                var p = i + len(needle) + 2
                while p < len(raw) and (raw[p] == 32 or raw[p] == 9 or raw[p] == 10 or raw[p] == 13 or raw[p] == 58):
                    p += 1
                var start = p
                if p < len(raw) and raw[p] == 34:
                    p += 1
                    start = p
                    while p < len(raw) and raw[p] != 34:
                        p += 1
                else:
                    while p < len(raw) and raw[p] != 44 and raw[p] != 125:
                        p += 1
                return _text_slice(text, start, p)
        i += 1
    raise AlofaError(ERR_PARSE, "safetensors metadata field not found", "key=" + key)


def _ascii_int(text: String) raises AlofaError -> Int:
    var raw = text.as_bytes()
    var out = 0
    var start = 0
    var sign = 1
    if len(raw) > 0 and raw[0] == 45:
        sign = -1
        start = 1
    for i in range(start, len(raw)):
        if raw[i] < 48 or raw[i] > 57:
            raise AlofaError(ERR_PARSE, "invalid safetensors integer", "text=" + text)
        out = out * 10 + Int(raw[i] - 48)
    return sign * out


def _array_for_key(text: String, key: String) raises AlofaError -> List[Int]:
    """Read the integer array belonging to a metadata key."""
    var raw = text.as_bytes()
    var needle = key.as_bytes()
    var p = 0
    while p + len(needle) + 2 <= len(raw):
        if raw[p] == 34:
            var matches = True
            for j in range(len(needle)):
                if raw[p + 1 + j] != needle[j]:
                    matches = False
            if matches and raw[p + 1 + len(needle)] == 34:
                p += len(needle) + 2
                while p < len(raw) and raw[p] != 91:
                    p += 1
                if p < len(raw):
                    var end = p
                    while end < len(raw) and raw[end] != 93:
                        end += 1
                    return _array_ints(_text_slice(text, p, end + 1))
        p += 1
    raise AlofaError(ERR_PARSE, "safetensors array field not found", "key=" + key)


def _array_ints(text: String) raises AlofaError -> List[Int]:
    """Read the first JSON integer array in `text`."""
    var raw = text.as_bytes()
    var out: List[Int] = []
    var i = 0
    while i < len(raw) and raw[i] != 91:
        i += 1
    if i == len(raw):
        out.append(_ascii_int(text))
        return out^
    i += 1
    while i < len(raw) and raw[i] != 93:
        while i < len(raw) and (raw[i] == 32 or raw[i] == 9 or raw[i] == 10 or raw[i] == 13 or raw[i] == 44):
            i += 1
        var start = i
        while i < len(raw) and raw[i] != 44 and raw[i] != 93:
            i += 1
        if i > start:
            out.append(_ascii_int(_text_slice(text, start, i)))
    if len(out) == 0:
        raise AlofaError(ERR_PARSE, "safetensors integer array is empty", "text=" + text)
    return out^


struct SafeTensorFile(Movable):
    """A single safetensors file indexed by tensor name.

    The first implementation accepts F32 tensors only. Other dtypes are
    rejected explicitly until a conversion-owning backend is added.
    """

    var mapped: MappedFile
    var names: List[String]
    var shapes: List[String]
    var begins: List[Int]
    var ends: List[Int]
    var dtypes: List[Int]
    var data_base: Int

    def __init__(out self, path: String) raises AlofaError:
        self.mapped = MappedFile(path)
        var header_len = _u64_le(self.mapped, 0)
        if header_len <= 0 or header_len > self.mapped.size - 8:
            raise AlofaError(ERR_PARSE, "safetensors header exceeds file", "header_length=" + String(header_len))
        var header_bytes: List[UInt8] = []
        for i in range(header_len):
            header_bytes.append(self.mapped.byte_at(8 + i))
        var header = String(unsafe_from_utf8=header_bytes)
        self.names = List[String]()
        self.shapes = List[String]()
        self.begins = List[Int]()
        self.ends = List[Int]()
        self.dtypes = List[Int]()
        self.data_base = 8 + header_len
        var cursor = 0
        while cursor < header.byte_length():
            var raw = header.as_bytes()
            if raw[cursor] != 34:
                cursor += 1
                continue
            var name_end = cursor + 1
            while name_end < len(raw) and raw[name_end] != 34:
                name_end += 1
            if name_end >= len(raw):
                raise AlofaError(ERR_PARSE, "unterminated safetensors tensor name", "path=" + path)
            var name = _text_slice(header, cursor + 1, name_end)
            if name == "__metadata__":
                cursor = name_end + 1
                continue
            var object_end = name_end
            var depth = 0
            while object_end < len(raw):
                if raw[object_end] == 123:
                    depth += 1
                elif raw[object_end] == 125:
                    depth -= 1
                    if depth == 0:
                        break
                object_end += 1
            if object_end >= len(raw):
                raise AlofaError(ERR_PARSE, "unterminated safetensors tensor metadata", "name=" + name)
            var object = _text_slice(header, name_end + 1, object_end)
            var dtype = _field(object, "dtype")
            var shape_values = _array_for_key(object, "shape")
            var shape = "["
            for j in range(len(shape_values)):
                if j > 0:
                    shape += ","
                shape += String(shape_values[j])
            shape += "]"
            var offsets = _array_for_key(object, "data_offsets")
            if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
                raise AlofaError(ERR_PARSE, "invalid safetensors data offsets", "name=" + name)
            var dt = DT_FP32
            if dtype == "F16":
                dt = DT_FP16
            elif dtype == "BF16":
                dt = DT_BF16
            elif dtype != "F32":
                raise AlofaError(ERR_UNSUPPORTED, "safetensors dtype is unsupported", "name=" + name + " dtype=" + dtype)
            var bytes_per = 4 if dt == DT_FP32 else 2
            var dims = _array_ints(shape)
            var count = 1
            for dim in dims:
                if dim <= 0:
                    raise AlofaError(ERR_PARSE, "safetensors shape has non-positive dimension", "name=" + name)
                count *= dim
            if offsets[1] - offsets[0] != count * bytes_per:
                raise AlofaError(ERR_PARSE, "safetensors shape and byte range disagree", "name=" + name)
            if self.data_base + offsets[1] > self.mapped.size:
                raise AlofaError(ERR_OUT_OF_RANGE, "safetensors tensor exceeds file", "name=" + name)
            self.names.append(name)
            self.shapes.append(shape)
            self.begins.append(offsets[0])
            self.ends.append(offsets[1])
            self.dtypes.append(dt)
            cursor = object_end + 1
        if len(self.names) == 0:
            raise AlofaError(ERR_PARSE, "safetensors header has no tensors", "path=" + path)

    def keep_alive(self):
        """Keep the mapped payload alive while returned views are used."""
        self.mapped.keep_alive()

    def index_of(self, name: String) raises AlofaError -> Int:
        """Find a tensor by its exact safetensors name."""
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise AlofaError(ERR_INVALID_ARGUMENT, "tensor is not in safetensors file", "name=" + name)

    def has(self, name: String) -> Bool:
        """Whether the file contains `name`."""
        for item in self.names:
            if item == name:
                return True
        return False

    def numel(self, name: String) raises AlofaError -> Int:
        """Return the number of elements in `name`."""
        var i = self.index_of(name)
        var dims = _array_ints(self.shapes[i])
        var n = 1
        for dim in dims:
            n *= dim
        return n

    def view(self, name: String) raises AlofaError -> TensorView:
        """Return an F32 view of `name`; non-F32 tensors are unsupported."""
        var i = self.index_of(name)
        if self.dtypes[i] != DT_FP32:
            raise AlofaError(ERR_UNSUPPORTED, "only F32 safetensors views are supported", "name=" + name)
        var dims = _array_ints(self.shapes[i])
        var ptr = self.mapped.ptr().unsafe_offset(self.data_base + self.begins[i])
        return TensorView(ptr, Shape(dims), DT_FP32)

    def ptr(self, name: String) raises AlofaError -> F32Ptr:
        """Return typed F32 data for `name`."""
        return f32_data(self.view(name))
