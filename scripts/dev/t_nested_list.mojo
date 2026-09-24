struct Item(Movable):
    """`Job` 的最小替身：结构体里嵌一块 `List[UInt8]`。

    要验的只有一件事：**这样一块内层 List 被"塞进 List → pop → 挪走"搬一圈之后，
    字节还是不是原来那些**。邮箱（`Mailbox.jobs`）对 `Job.raw` 做的正是这三步。
    """

    var slot: Int
    var raw: List[UInt8]

    def __init__(out self, slot: Int, n: Int):
        self.slot = slot
        self.raw = List[UInt8](capacity=n)
        var i = 0
        while i < n:
            self.raw.append(UInt8((i * 7 + 3) % 251))
            i += 1


def main():
    var q = List[Item]()
    var bad = 0
    var k = 0
    while k < 300:
        q.append(Item(k, 64 + (k % 17))^)
        k += 1
    var checked = 0
    while len(q) > 0:
        var got = q.pop(0)
        var ok = got.slot == checked and len(got.raw) == 64 + (checked % 17)
        var i = 0
        while i < len(got.raw):
            if Int(got.raw[i]) != (i * 7 + 3) % 251:
                ok = False
            i += 1
        if not ok:
            bad += 1
            print(
                "BAD at "
                + String(checked)
                + " slot="
                + String(got.slot)
                + " len="
                + String(len(got.raw))
            )
        checked += 1
    print("checked=" + String(checked) + " bad=" + String(bad))
