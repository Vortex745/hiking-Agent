"""稳定评估包：被跟踪、无 scratch 后备，供 Promptfoo 与 DeepEval 共用。

模块（spec T09 迁移的五个必要模块 + 状态分类扩展）：

- ``models``  —— 用例/动作/记录/检查的版本化数据模型；
- ``target``  —— ASGI/HTTP 目标适配器与 SSE 记录解析（含 Chat 终态提取）；
- ``fixtures`` —— 受控评测环境：临时 SQLite、真实 FastAPI 路由、替身注入；
- ``kernels`` —— 可控内核替身（scripted / agent_policy），记录调用入参；
- ``status``  —— 执行契约与语义质量的分离分类、报告构建（T10）。

历史 scratch CLI、报告与 runs 目录不迁入本包。
"""

from __future__ import annotations
