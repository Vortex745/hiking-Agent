# UniHarness fork provenance（Ticket 25 / Ticket 01 / harness-p0-p1 Ticket 03）

更新时间：2026-10-28（invalid tool arguments delta）；离线 bundle 章节更新于 2026-10-28
决策：本地 vendor 固定 commit（用户批准，未创建 GitHub 远端 fork）

## 来源与 pin

| 项 | 值 |
|---|---|
| 上游仓库 | `https://github.com/UnicomAI/UniHarness` |
| 上游基点 | `757bf3d60f763706aae60b040d775e8563b9c577`（上游 HEAD，已用 `git ls-remote` 核对一致） |
| 项目 delta | HEAD `aeceee7ea5bdb8c4dcaa7d1c54400afacabe5556`（上游基点 `757bf3d` 之上的五个受控 delta commit） |
| 运营 pin | `AGENT_V2_UNIHARNESS_PIN` = delta HEAD（fork 全量 commit）；身份校验额外要求上游基点是其祖先 |
| vendor 位置 | `third_party/uniharness/`（完整克隆，detached HEAD 于上述 delta commit） |
| 安装方式 | `pip install -e third_party/uniharness/libs/uniharness`（hatchling + hatch-vcs，fallback version 0.0.0） |

### 受控 delta 清单

| commit | 内容 | 原因 |
|---|---|---|
| `2cffe80` | `create_agent` 新增 `builtin_tool_groups` 参数：按组过滤内建工具面（cli/web/todo/task/skills）；默认 None 保持上游逐字节行为；web providers 要求 web 组；subagent spawner 挂在 task 组 | Ticket 26 无工具 chat 需要纯 LLM 循环；该参数与旧 stub 的五个 ToolGroup 语义对齐，属此前批准的窄扩展点 |
| `67a9497` | `create_agent` 新增 `resolve_environment: bool = True` 参数；False 时跳过 Computer 环境探测、装配合成 EnvironmentContext（仅适用于自定义 system_prompt 的部署，compaction 重构不读 env） | 无 shell chat 运行不得触碰沙箱；保持工装生命周期不被 no-tool 路径隐式创建 |
| `aaf0f8d` | `create_agent` 新增 `permission_gate: PermissionGate \| None = None` 参数：调用方可注入预配置 gate（挂产品 SafetyRule），透传至 runner 与 middleware；None 保持上游自建行为 | Ticket 28 要求产品侧唯一 PermissionGate 裁决每个工具调用，而上游内部自建 gate 无注册口——属既定窄扩展点缺口的最小闭合 |
| `b026a26` | `create_agent` 新增 `approval_callback: Any \| None = None` 参数：透传至 AgentMiddleware 的 NEEDS_APPROVAL 分支（middleware 本已支持该参数但工厂未接）；None 保持上游默认拒绝 | Ticket 31 durable approval 需要 kernel tool-call 边界回调；配合 effect-ledger 守卫 adapter 实现"批准后才放行" |
| `aeceee7` | `AgentMiddleware.aafter_model` 在工具执行与压缩前检查 `AIMessage.invalid_tool_calls`：全非法且调用 ID 完整时按 ID 追加一次格式错误 ToolMessage 并 `jump_to=model`；混合调用、预算耗尽、缺 ID、纠错回复无工具调用均抛类型化 `InvalidToolArgumentsError`；`UniHarnessState` 新增两个可选检查点字段 | 2026-10-28 诊断确认非法生图参数只进入 `invalid_tool_calls`，LangChain 模型→工具边只看有效调用，循环直接结束并把前言当成功；本 delta 只做一次有界纠错和明确失败，不猜补参数、不重发生图 |

新增 delta 的流程：先在本表登记意图 → 在 vendor 树提交 → 更新运营 pin/config 默认值/`.env`
→ 重跑 factory contract 套件。禁止无登记改动。

## 与 ticket 措辞的偏差说明

Ticket 25 要求"从上游 revision 派生的项目控制 fork"。本仓库选择**本地 vendor**而非在
GitHub 账号下创建远端 fork：pin 的不可变性由固定 commit + `agent_v2.uniharness.kernel_identity`
的 `git rev-parse HEAD` 运行时证明保证。若后续需要远端 fork ownership（CI 只读访问、
对外发布），可在不动 pin 的前提下把本目录推送到项目控制的远端并补登 origin URL。

## 身份验证链

1. 运行时：`agent_v2/uniharness/kernel_identity.py::resolve_kernel_identity`
   - 读 `importlib.metadata.version("uniharness")`
   - 解析加载包的真实源路径，执行 `git rev-parse HEAD`
   - 与 `AGENT_V2_UNIHARNESS_PIN` / `VENDORED_UPSTREAM_REVISION` 比对；不一致 → 启动 fail closed
2. 健康：`/chat/health`（及 agent_v2.health）暴露 `kernel_identity.distribution_version/fork_commit`
3. 升级检查点：换 pin = 换 `VENDORED_UPSTREAM_REVISION`、config 默认值、`.env`，
   重跑 factory contract 测试套件

## license

上游 license 以 `third_party/uniharness/LICENSE` 为准；本仓库分发保持其原始声明，
不修改版权与署名。

## 离线 bundle（harness-p0-p1 Ticket 03）

普通 checkout（`git clone`/`git worktree`/CI 干净检出）不包含 `third_party/uniharness/.git`
嵌套历史，`git rev-parse HEAD` 会落到外层仓库，kernel 身份校验因此 fail closed
（2026-10-15 在集成 worktree 全量套件中记录到该失败的实证：3 个
`test_uniharness_fork_factory_contract.py` 身份用例失败，主仓库内通过）。
离线 bundle 让干净环境无需联网即可恢复同一份受控历史。

| 项 | 值 |
|---|---|
| bundle 文件 | `third_party/uniharness-fork.bundle`（随仓库分发，~2.7 MB） |
| bundle 内容 | 仅固定 pin 的可达历史（97 个 commit，上游基点及其五个受控 delta）；不含无关引用、远端分支或本地配置 |
| 固定 pin | `aeceee7ea5bdb8c4dcaa7d1c54400afacabe5556`（bundle 内记为 `HEAD`） |
| bundle SHA-256 | `3b47a260202d5abb9580df7526a2d177d93d118e774f833ead3f72a2a78d0480` |
| 来源 | 本机 `third_party/uniharness`（detached HEAD 于 pin，`git bundle create` 生成，未经网络） |

恢复与校验（干净 checkout 的临时目录中执行，不触碰原有 vendor 树）：

```bash
git clone third_party/uniharness-fork.bundle /tmp/uniharness-restore
cd /tmp/uniharness-restore
git rev-parse HEAD                        # 必须等于固定 pin
git merge-base --is-ancestor 757bf3d60f763706aae60b040d775e8563b9c577 HEAD  # 上游基点必须是祖先
pip install --no-deps --no-build-isolation ./libs/uniharness   # 离线安装同一版本
```

错误 pin、被篡改的 bundle（SHA-256 不符）或不完整历史（缺少祖先）必须真实拒绝，
不允许伪造身份或跳过校验；契约见
`ai-service/test_uniharness_fork_restore_contract.py`。升级 pin 时必须同步重建
bundle、更新本表 SHA-256，并重跑恢复契约与 factory 契约套件。

## 已知的窄 fork delta 需求（观察项，T26/T28 处理）

- `create_agent()` 未透传 `approval_callback`（`AgentMiddleware` 支持该参数但工厂未接）
- 内核内部自建 `PermissionGate`，产品侧无法通过公共 API 注册 `SafetyRule`

两者在 T25 中以类型级契约测试证明 surface 可用；真正接线时优先探索不改 vendor
源码的产品侧方案（wrapper / middleware 注入），确有必要再引入受控 patch 分支。
