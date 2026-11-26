"""L2 model layer — loading parameters and configuration from disk.

There are two on-disk formats, because there are two reasons to have one. The
first is the one `scripts/dump_model_reference.py` writes: a TSV index next to
one fp32 payload. The index is plain text so a third party can read the shapes
and offsets without running anything, and the payload is memory-mapped rather
than read, so two gigabytes of parameters cost address space and page cache
instead of a second copy in the heap.

The second is a Hugging Face `*.safetensors` file, which is what a checkpoint
actually ships: one file, metadata in front, and — for every modern checkpoint
— bfloat16 payloads. Widening those to fp32 cannot happen inside a read-only
mapping, so a file that needs it owns an arena to widen into; one that does not
never allocates. Both formats present the same thing to the caller: an fp32,
row-major view per name.

This layer knows L0 (views, dtypes, mappings) and L1 (the arithmetic it hands
views to). It has no notion of requests or serving, and it does not depend on
`verify/` — the verification pillar looks down at this code, never the other
way around.

Run:
    pixi run mojo run -I src tests/unit/test_model_parity.mojo
"""

from alofa.core.dtype import DT_BF16, DT_FP32, elem_name
from alofa.core.error import (
    ERR_INVALID_ARGUMENT,
    ERR_OUT_OF_RANGE,
    ERR_PARSE,
    ERR_UNSUPPORTED,
    AlofaError,
)
from alofa.core.ffi.mem import RawPtr
from alofa.core.memory import Arena, SharedRegion
from alofa.core.mmap import MappedFile
from alofa.core.tensor import F32Ptr, Shape, TensorView, f32_data
from alofa.core.text import parse_int, read_text
from alofa.model.safetensors import SafeTensorFile, bf16_to_f32, f32_copy


def _has_suffix(path: String, suffix: String) -> Bool:
    if path.byte_length() < suffix.byte_length():
        return False
    var p = path.as_bytes()
    var s = suffix.as_bytes()
    var start = len(p) - len(s)
    for i in range(len(s)):
        if p[start + i] != s[i]:
            return False
    return True


def _text_slice(raw: String, start: Int, end: Int) -> String:
    var bytes = List[UInt8]()
    var source = raw.as_bytes()
    for i in range(start, end):
        bytes.append(source[i])
    return String(unsafe_from_utf8=bytes)


def _encode_shape(encoded: String) -> String:
    """`"[a,b]"` → `"axb"`（TensorFile 的形状文本口径）。"""
    var clean = ""
    var raw_shape = encoded.as_bytes()
    for j in range(len(raw_shape)):
        if raw_shape[j] == 91 or raw_shape[j] == 93:
            continue
        if raw_shape[j] == 44:
            clean += "x"
        else:
            clean += _text_slice(encoded, j, j + 1)
    return clean


struct WeightMap(Movable):
    """safetensors index.json 的 `weight_map` 解析结果：张量名 → 分片文件名。"""

    var names: List[String]
    var files: List[String]

    def __init__(out self):
        self.names = List[String]()
        self.files = List[String]()


def _parse_weight_map(text: String) raises AlofaError -> WeightMap:
    """safetensors 分片 index.json 的 `weight_map` 扫描器。

    只收录"值为带 `.safetensors` 后缀的字符串"的键值对：张量名不含引号，
    `metadata` / `weight_map` 这类容器键的值是对象或数字，天然不会成对。
    扫描顺序即 JSON 顺序（HF 导出按名排序），所以结果确定。
    """
    var out = WeightMap()
    var raw = text.as_bytes()
    var i = 0
    while i < len(raw):
        if raw[i] != 34:
            i += 1
            continue
        # 一个引号字符串
        var start = i + 1
        var end = start
        while end < len(raw) and raw[end] != 34:
            end += 1
        if end >= len(raw):
            break
        var key = _text_slice(text, start, end)
        i = end + 1
        # 跳过空白，看下一个有效字符是不是 `:`
        while i < len(raw) and (raw[i] == 32 or raw[i] == 10 or raw[i] == 13 or raw[i] == 9):
            i += 1
        if i >= len(raw) or raw[i] != 58:
            continue
        i += 1
        while i < len(raw) and (raw[i] == 32 or raw[i] == 10 or raw[i] == 13 or raw[i] == 9):
            i += 1
        if i < len(raw) and raw[i] == 34:
            var vstart = i + 1
            var vend = vstart
            while vend < len(raw) and raw[vend] != 34:
                vend += 1
            if vend >= len(raw):
                break
            var value = _text_slice(text, vstart, vend)
            i = vend + 1
            if _has_suffix(value, ".safetensors"):
                out.names.append(key)
                out.files.append(value)
        # 值是对象/数组/数字：不碰，下一轮扫描自然跳过
    if len(out.names) == 0:
        raise AlofaError(
            ERR_PARSE, "safetensors index has no weight_map entries", ""
        )
    return out^


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
    # Storage for payloads that are not already fp32. A checkpoint ships in
    # bf16, and widening it cannot be done in the mapping — a mapping is
    # read-only — so the widened copy is owned here and outlives every view
    # handed out of it.
    #
    # 类型是 `SharedRegion` 而不是 `Arena`：工作线程池里的孪生体**借用原型
    # 这一块**（借用 = 析构不还），见 `SharedRegion` 与下面的 `borrowing`。
    var arena: SharedRegion
    var converted: Bool
    # 用于孪生体重新映射同一份文件（映射本身是 MAP_PRIVATE 只读，同一份文件的
    # 页在进程内共享）。
    var path: String
    # 分片 safetensors（一百八十）：`model.safetensors.index.json` + N 个 shard。
    # `safes` 持有每片的元数据与映射；`shard_of[i]` 是张量 i 所在分片。
    # 非分片路这两个字段为空，一切走原有字段 —— 判定只看 `len(safes) > 0`。
    var safes: List[SafeTensorFile]
    var shard_paths: List[String]
    var shard_of: List[Int]

    def __init__(out self, dir_path: String) raises AlofaError:
        self.path = dir_path
        self.converted = False
        self.safes = List[SafeTensorFile]()
        self.shard_paths = List[String]()
        self.shard_of = List[Int]()
        if _has_suffix(dir_path, ".safetensors"):
            var safe = SafeTensorFile(dir_path)
            var n = len(safe.names)
            self.mapped = MappedFile(dir_path)
            self.names = safe.names.copy()
            self.shape_text = List[String]()
            for i in range(n):
                self.shape_text.append(_encode_shape(safe.shapes[i]))
            self.offsets = List[Int]()
            self.counts = List[Int]()
            var total = 0
            var widens = False
            for i in range(n):
                var count = safe.numel(safe.names[i])
                self.counts.append(count)
                total += count
                if safe.dtypes[i] != DT_FP32:
                    widens = True
            if not widens:
                # One page is held even when nothing is widened: every field is
                # initialised on every path, and the alternative — an optional
                # arena — would make each reader answer a question the file
                # answers once.
                self.arena = SharedRegion(1)
                for i in range(n):
                    self.offsets.append(safe.data_base + safe.begins[i])
                return
            self.converted = True
            self.arena = SharedRegion(total * 4 + 16 * (n + 1))
            var base = self.arena.alloc(total * 4)
            var cursor = 0
            for i in range(n):
                # 16-byte alignment per tensor: the widened buffer is what the
                # vector kernels read, and an unaligned row start costs more
                # than the padding it takes to avoid.
                cursor = (cursor + 15) // 16 * 16
                self.offsets.append(cursor)
                var source = self.mapped.ptr().unsafe_offset(
                    safe.data_base + safe.begins[i]
                )
                var target = base.unsafe_offset(cursor).unsafe_bitcast[Float32]()
                var kind = safe.dtypes[i]
                if kind == DT_FP32:
                    f32_copy(source, target, self.counts[i])
                elif kind == DT_BF16:
                    bf16_to_f32(source, target, self.counts[i])
                else:
                    raise AlofaError(
                        ERR_UNSUPPORTED,
                        "safetensors dtype is unsupported",
                        "name=" + safe.names[i] + " dtype=" + elem_name(kind),
                    )
                cursor += self.counts[i] * 4
            return
        # 分片路（一百八十）：目录下有 `model.safetensors.index.json`。
        # 现代大 checkpoint 都长这样（Qwen3-8B = 5 片 + index）；HF 的 index 只
        # 有一张映射表 `weight_map`（张量名 → 分片文件名），本分支把它展开成与
        # 单文件完全一致的 TensorFile API —— 上面的架构层一行都不用改。
        var index_text = ""
        try:
            index_text = read_text(dir_path + "/model.safetensors.index.json")
        except:
            pass
        if index_text.byte_length() > 0:
            var wm = _parse_weight_map(index_text)
            # 分片文件按首次出现顺序开（HF 导出即排序序）；每片独立解析元数据。
            var shard_index = List[Int]()
            for j in range(len(wm.files)):
                var file = wm.files[j]
                var which = -1
                for s in range(len(self.shard_paths)):
                    if self.shard_paths[s] == file:
                        which = s
                if which == -1:
                    self.shard_paths.append(dir_path + "/" + file)
                    self.safes.append(
                        SafeTensorFile(dir_path + "/" + file)
                    )
                    which = len(self.safes) - 1
                shard_index.append(which)
            # `mapped` 在分片路上无消费方（view 走 safes），但每个字段都必须
            # 在每条路上被初始化 —— 用第一片占位。
            self.mapped = MappedFile(self.shard_paths[0])
            # 校验：每片张量名都必须在 index 里 —— 多出来的意味着 index 与
            # 分片内容脱钩（导出错、传输缺），宁可拒绝也不静默读半套。
            var n = len(wm.names)
            for s in range(len(self.safes)):
                for t in range(len(self.safes[s].names)):
                    var found = False
                    for j in range(n):
                        if wm.names[j] == self.safes[s].names[t]:
                            found = True
                    if not found:
                        raise AlofaError(
                            ERR_PARSE,
                            "shard contains a tensor the index does not know",
                            "shard=" + self.shard_paths[s]
                            + " tensor=" + self.safes[s].names[t],
                        )
            self.names = List[String]()
            self.shape_text = List[String]()
            self.offsets = List[Int]()
            self.counts = List[Int]()
            # 每张量在**其分片映射内**的字节偏移与 dtype（扁平化，避免从
            # `self.safes` 里拷结构体 —— SafeTensorFile 不可隐式拷贝）。
            var src_off = List[Int]()
            var src_kind = List[Int]()
            var total = 0
            var widens = False
            for j in range(n):
                var which = shard_index[j]
                var t = self.safes[which].index_of(wm.names[j])
                if t == -1:
                    raise AlofaError(
                        ERR_PARSE,
                        "index names a tensor its shard does not have",
                        "name=" + wm.names[j] + " shard=" + self.shard_paths[which],
                    )
                self.names.append(wm.names[j])
                self.shard_of.append(which)
                self.counts.append(self.safes[which].numel(wm.names[j]))
                self.shape_text.append(_encode_shape(self.safes[which].shapes[t]))
                src_off.append(
                    self.safes[which].data_base + self.safes[which].begins[t]
                )
                src_kind.append(self.safes[which].dtypes[t])
                total += self.counts[j]
                if src_kind[j] != DT_FP32:
                    widens = True
            if not widens:
                self.arena = SharedRegion(1)
                for j in range(n):
                    self.offsets.append(src_off[j])
                return
            self.converted = True
            self.arena = SharedRegion(total * 4 + 16 * (n + 1))
            var base = self.arena.alloc(total * 4)
            var cursor = 0
            for j in range(n):
                cursor = (cursor + 15) // 16 * 16
                self.offsets.append(cursor)
                var source = self.safes[shard_index[j]].mapped.ptr().unsafe_offset(
                    src_off[j]
                )
                var target = base.unsafe_offset(cursor).unsafe_bitcast[Float32]()
                var kind = src_kind[j]
                if kind == DT_FP32:
                    f32_copy(source, target, self.counts[j])
                elif kind == DT_BF16:
                    bf16_to_f32(source, target, self.counts[j])
                else:
                    raise AlofaError(
                        ERR_UNSUPPORTED,
                        "safetensors dtype is unsupported",
                        "name=" + wm.names[j] + " dtype=" + elem_name(kind),
                    )
                cursor += self.counts[j] * 4
            return
        self.arena = SharedRegion(1)
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

    def __init__(out self, proto: TensorFile, borrowed: Bool) raises AlofaError:
        """搭在 `proto` 的只读权重上（服务线程孪生体用；`borrowed` 只做区分）。

        这里**不再加宽**：加宽后的 fp32 副本是那份 bf16 checkpoint 的 2 倍大
        （988 MB → 1.98 GB），每个工作线程各来一份就是账本四十三量到的那
        +1.96 GB/条。孪生体要的只是"能读到同一批字节"，所以映射重新开一份
        （同一文件的页在进程内共享），加宽区按 `SharedRegion.borrowed` 借过来。

        `offsets` 直接抄原型的：加宽分支里它们相对加宽区基址、不加宽的分支里
        相对映射基址，两种口径在孪生体上都成立（基址来源一一对应）。
        """
        self.path = proto.path
        self.names = proto.names.copy()
        self.shape_text = proto.shape_text.copy()
        self.offsets = proto.offsets.copy()
        self.counts = proto.counts.copy()
        self.arena = proto.arena.borrowed()
        self.converted = proto.converted
        self.safes = List[SafeTensorFile]()
        self.shard_paths = proto.shard_paths.copy()
        self.shard_of = proto.shard_of.copy()
        # 分片路：每片重新解析映射（同一文件，页在进程内共享）。`mapped` 与
        # 原型侧一致取第一片占位 —— **不能重开 `proto.path`**：分片路上它是
        # 目录，mmap 目录必然失败（一百八十一排障实录：serve 的引擎线程里
        # 这个 raise 无人接，进程裸崩无任何输出）。
        if len(self.shard_paths) > 0:
            self.mapped = MappedFile(self.shard_paths[0])
            for s in range(len(self.shard_paths)):
                self.safes.append(SafeTensorFile(self.shard_paths[s]))
        else:
            self.mapped = MappedFile(proto.path)

    def keep_alive(self):
        """State that the mapping is still in use here.

        Views from this file point into the mapping, and Mojo ends a value's
        lifetime at its last use — see `MappedFile.keep_alive`.
        """
        self.mapped.keep_alive()
        self.arena.keep_alive()
        for s in range(len(self.safes)):
            self.safes[s].keep_alive()

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
        """A read-only fp32 view of `name`.

        Over the widened buffer when the payload was not fp32, over the mapping
        when it was; the two differ in where they point, not in what they are
        — fp32, contiguous, row-major.
        """
        var i = self.index_of(name)
        var base: RawPtr
        if self.converted:
            base = self.arena.data
        elif len(self.safes) > 0:
            # 分片路：非加宽视图的偏移是"分片映射内"字节偏移（data_base+begin），
            # 基址取张量所在分片的映射。
            base = self.safes[self.shard_of[i]].mapped.ptr()
        else:
            base = self.mapped.ptr()
        var ptr = base.unsafe_offset(self.offsets[i])
        return TensorView(ptr, Shape(self.dims(name)), DT_FP32)

    def ptr(self, name: String) raises AlofaError -> F32Ptr:
        """Typed access to `name`'s elements; see `f32_data`."""
        return f32_data(self.view(name))
