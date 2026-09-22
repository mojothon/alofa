"""L6 接口适配：`srv` 包 —— 把引擎接到 HTTP 上。

分层见 `docs/plan/02-architecture.md`：L5 是服务运行时（连接循环），L6 是接口适配
（线上格式与对外契约）。这个包目前只有非流式一块：

- `srv/http.mojo`   HTTP/1.1 的请求解析与响应组帧（纯字节，无 I/O）
- `srv/openai.mojo` `/v1/chat/completions` 的**非流式**请求/响应与路由
- `srv/server.mojo` 单进程、一次一个连接、阻塞的服务循环

入口在 `src/alofa/serve.mojo`（与 `cli.mojo` 同层）。
"""
