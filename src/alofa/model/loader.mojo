"""L2 model layer — loading parameters and configuration from disk.

The on-disk format is the one `scripts/dump_model_reference.py` writes: a TSV
index next to one fp32 payload. The index is plain text so a third party can
read the shapes and offsets without running anything, and the payload is
memory-mapped rather than read, so two gigabytes of parameters cost address
space and page cache instead of a second copy in the heap.

This layer knows L0 (views, dtypes, mappings) and L1 (the arithmetic it hands
views to). It has no notion of requests or serving, and it does not depend on
`verify/` — the verification pillar looks down at this code, never the other
way around.

Run:
    pixi run mojo run -I src tests/unit/test_model_parity.mojo
"""

from alofa.core.dtype import DT_FP32
from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_PARSE,
    AlofaError,
)
from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr, Shape, TensorView, f32_data
from alofa.core.text import parse_int, read_text


def config_value(path: String, key: String) raises AlofaError -> String:
    """One value from a `key <tab> value` configuration file.

    The exporter writes a fixed-point format for its floats (see
    `core/text.mojo`), so the values parse with `parse_int` /
    `parse_float64` and nothing locale-dependent.
    """
    var text = read_text(path)
    var lines = text.split("\n")
    for line_span in lines:
        var line = String(line_span)
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        if len(fields) < 2:
            continue
        var first = String(fields[0])
        if first != key:
            continue
        return String(fields[1])
    raise AlofaError(
        ERR_INVALID_ARGUMENT, "configuration key not found", "key=" + key
    )


struct TensorFile(Movable):
    """A mapped fp32 payload plus the index that gives its parts names.

    Shapes are kept as the raw text from the index and parsed on demand: a
    `List[List[Int]]` field would hand out references into itself on every
    lookup, which is exactly the aliasing Mojo makes explicit elsewhere.
    """

    var mapped: MappedFile
    var names: List[String]
    var shape_text: List[String]
    var offsets: List[Int]
    var counts: List[Int]

    def __init__(out self, dir_path: String) raises AlofaError:
        var index = read_text(dir_path + "/tensors.tsv")
        self.mapped = MappedFile(dir_path + "/tensors.f32")
        self.names = List[String]()
        self.shape_text = List[String]()
        self.offsets = List[Int]()
        self.counts = List[Int]()

        var lines = index.split("\n")
        for line_span in lines:
            var line = String(line_span)
            if line.byte_length() == 0:
                continue
            var fields = line.split("\t")
            if len(fields) != 4:
                raise AlofaError(
                    ERR_PARSE,
                    "index line does not have four fields",
                    "line=" + line,
                )
            var name = String(fields[0])
            var dims_text = String(fields[1])
            var offset = parse_int(String(fields[2]))
            var count = parse_int(String(fields[3]))

            # The element count is stored, but it is also derived: a mismatch
            # means the exporter and this reader disagree about what a shape
            # is, and that is worth failing on rather than papering over.
            var derived = 1
            if dims_text.byte_length() > 0:
                var parts = dims_text.split("x")
                for part_span in parts:
                    derived *= parse_int(String(part_span))
            if derived != count:
                raise AlofaError(
                    ERR_PARSE,
                    "element count does not match the shape",
                    "name=" + name + " shape=" + dims_text + " count=" + String(count),
                )
            if offset + count * 4 > self.mapped.size:
                raise AlofaError(
                    ERR_OUT_OF_RANGE,
                    "entry extends past the payload",
                    "name=" + name + " end=" + String(offset + count * 4),
                )

            self.names.append(name)
            self.shape_text.append(dims_text)
            self.offsets.append(offset)
            self.counts.append(count)

        if len(self.names) == 0:
            raise AlofaError(ERR_PARSE, "index is empty", "dir=" + dir_path)

    def keep_alive(self):
        """State that the mapping is still in use here.

        Views from this file point into the mapping, and Mojo ends a value's
        lifetime at its last use — see `MappedFile.keep_alive`.
        """
        self.mapped.keep_alive()

    def index_of(self, name: String) raises AlofaError -> Int:
        """Position of `name` in the index, or a named error.

        A missing name is an error rather than a default: a forward pass that
        silently read an empty tensor would produce a plausible-looking answer.
        """
        var i = 0
        while i < len(self.names):
            if self.names[i] == name:
                return i
            i += 1
        raise AlofaError(
            ERR_INVALID_ARGUMENT, "entry is not in the file", "name=" + name
        )

    def has(self, name: String) -> Bool:
        """Whether the file contains `name`."""
        var i = 0
        while i < len(self.names):
            if self.names[i] == name:
                return True
            i += 1
        return False

    def numel(self, name: String) raises AlofaError -> Int:
        """Element count of `name`."""
        return self.counts[self.index_of(name)]

    def dims(self, name: String) raises AlofaError -> List[Int]:
        """The shape of `name`, parsed from the index text."""
        var text = self.shape_text[self.index_of(name)]
        var dims = List[Int]()
        if text.byte_length() == 0:
            return dims^
        var parts = text.split("x")
        for part_span in parts:
            dims.append(parse_int(String(part_span)))
        return dims^

    def view(self, name: String) raises AlofaError -> TensorView:
        """A read-only fp32 view of `name`, over the mapped payload."""
        var i = self.index_of(name)
        var ptr = self.mapped.ptr().unsafe_offset(self.offsets[i])
        return TensorView(ptr, Shape(self.dims(name)), DT_FP32)

    def ptr(self, name: String) raises AlofaError -> F32Ptr:
        """Typed access to `name`'s elements; see `f32_data`."""
        return f32_data(self.view(name))
