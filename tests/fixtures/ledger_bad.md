# 坏账本 fixture —— 自检门专用

> **本文件故意包含违规条目。它不是文档，是一份"给校验器做的抗原"。**
>
> 作用：证明 `tests/capability/ledger.mojo` 这道门**真的会拒绝坏记录**。
> 如果哪天校验器被改坏（比如把文件存在性检查删掉），这个 fixture 就会让它暴露 ——
> 那时本文件的错误数会低于预期，`test_ledger.mojo` 失败。
>
> **门若不能拒绝下面的东西，它就不是门，只是装饰。**
> 对应 `docs/plan/03-roadmap.md` 的 P0 门：「故意提交错误账本条目，CI 必须失败」。

## 违规用例

| 能力 | 状态 | 证据 |
|---|---|---|
| 编造的能力 A | `verified` | `evidence:tests/this_file_does_not_exist.mojo`（**文件不存在**） |
| 编造的能力 B | `verified` | 只有一句描述，没有 `evidence:` 键 |
| 编造的环境事实 C | `verified-env` | 只有一句描述，没有 `probe:` 键 |
| 编造的能力 D | `verifed` | 拼错的标签 —— 不会被任何规则命中，静默绕过 |
