# Responses 受控生图

为 AstrBot 提供独立绘图流程：按需收集参考图、生成或编辑图片，再将实际图片交回主 Agent 查看并发送。支持 Responses 和 Images API，复用通用模型配置，无需修改 AstrBot 源码，也无需额外安装 skills。

## 功能

- 自然语言绘图，或说“修改刚才那张图”继续编辑最近生成的图片。
- 从当前消息及明确选中的近期文字中解析角色，独立绘图上下文避免混入其他历史图片。
- 按需搜索和核对参考图，支持原生 `web_search`、Tavily 和管理员维护的角色参考库。
- 分别指定设定收集模型、绘图连接和实际图片模型；新图搜索失败时继续无参考绘制。
- 可选后台队列，主聊天不用等待；自然语言查询队列、查询额度和取消排队任务。
- 按窗口限制一小时和 24 小时的请求数，管理员免额。
- 实际图片进入主 Agent 工具结果和会话历史。成功发送后结束这一轮，不再额外调用模型确认。
- QQ 官方机器人与 OneBot v11 群聊通知适配，允许简短说明，不附带参考来源链接。

## 安装与前提

安装版本范围为 AstrBot **`>=4.29.4,<5`**。当前回归覆盖正式 v4.29.4，以及包含原生 `wake_main_agent` 的固定 core 修订；后续 4.x 若调整内部接口，仍需验证与适配。运行依赖由 `requirements.txt` 声明，在插件安装时由 AstrBot 管理。

1. 在插件管理中使用仓库地址 `https://github.com/twfx1207/astrbot_plugin_responses_image` 安装，重载插件或重启 AstrBot。也可上传构建的 `astrbot_plugin_responses_image-v1.0.0.zip`，或将仓库克隆到 `data/plugins/astrbot_plugin_responses_image/`。
2. 主聊天模型须支持图片输入、工具调用，启用 `send_message_to_user`，并能接收工具返回的图片。
3. 在插件面板选择绘图接口、worker 模型连接及可选搜索。
4. 后台模式要求连接器支持主动消息。没有原生 `wake_main_agent` 的 v4.29.4 使用内置兼容流程，要求使用 `local` Agent runner。

Docker 部署请使用持久化数据卷。角色库和额度记录位于 AstrBot 数据目录的 `plugin_data/astrbot_plugin_responses_image/`，应一并备份。插件名、配置键和数据目录在 v1.0 中保持兼容。

## 配置

### 面板选项

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `worker_provider_id` | 留空 | 绘图连接的模型配置 ID，留空使用当前聊天模型连接。Responses 要求 `openai_responses`，Images 也支持 `openai_chat_completion`。 |
| `generation_api` | `responses` | `responses` 通过通用模型的图片工具绘图；`images` 直接调用纯图片模型。 |
| `image_model` | 留空 | 实际图片模型 API 名称，优先于 worker 的 `image_generation.model`。均为空时，Responses 由上游选择，Images 使用 worker 自身模型名。 |
| `sub_agents_enabled` | `false` | 启用设定收集与绘图分工，核对参考图时额外整理外观摘要。已有明确参考或建档角色跳过收集。 |
| `research_provider_id` | 留空 | 开启分工后的收集模型配置 ID，留空沿用 worker；需要支持识图的 `openai_responses` 连接。 |
| `reference_search` | `disabled` | `disabled` 关闭搜索；`native` 使用原生 `web_search`；`tavily` 复用当前会话的 Tavily 密钥。 |
| `background_generation` | `false` | 先返回队列回执，完成后由主 Agent 看图并发送。 |
| `max_concurrent_jobs` | `4` | 后台并发数，1–16；参考收集和绘图均占并发。 |
| `max_queue_size` | `20` | 额外队列容量，0–200；总容量为并发数加此值，包含等待主 Agent 领取的结果。 |
| `hourly_limit` | `10` | 每窗口最近一小时请求限额，`0` 不限。 |
| `daily_limit` | `30` | 每窗口最近 24 小时请求限额，`0` 不限，与小时限额同时生效。 |

### 接口与模型

| 场景 | 设置 |
| --- | --- |
| 通用模型支持 Responses 图片工具 | `generation_api=responses`，worker 选择该 `openai_responses` 模型配置，`image_model` 可指定图片工具模型。 |
| 直接调用纯图片模型 | `generation_api=images`，worker 提供 Images API 地址和密钥，`image_model` 填实际图片模型名称。 |
| 纯图片模型需要搜参考图 | 开启分工，为 `research_provider_id` 指定支持识图的 Responses 模型，再选择搜索方式。 |
| 模型不支持原生搜索 | 使用 `tavily`，收集模型仍须支持识图；也可关闭搜索或使用角色参考库。 |

Responses 外层 `model` 来自 worker 配置，`image_model` 写入 `image_generation.model`。Images 的 `image_model` 直接作为请求的 `model`；无参考时调用 `/images/generations`，有参考或编辑时调用 `/images/edits`，一次请求生成一张图片。上游须支持相应接口和图片输入。模型名称和参数以实际服务商支持情况为准，不会自动切换接口或重试生图。

图片质量、尺寸等可在**独立 worker** 的 `custom_extra_body` 配置。例如 Responses：

```json
{"tools": [{"type": "image_generation", "quality": "high", "output_format": "png"}]}
```

Images 模式可设置顶层 `quality`、`size`、`background`、`output_format` 等图片参数。不要在主聊天模型额外参数中启用原生 `image_generation`，以免绕过插件队列、额度和上下文控制。修改插件面板的图片模型不会修改共享模型配置。

### 参考图搜索

主 Agent 在调用前判断是否需要搜索。原创内容、不搜索要求、明确参考和编辑目标通常不需搜索；未建档的既有角色可触发检索。一次最多选择四个角色或四张参考图。

- `native` 无需搜索服务密钥，要求收集模型和上游支持 Responses `web_search`。没有图片结果时，尝试从实际引用网页提取静态图片，再看图核对。
- `tavily` 读取当前会话配置的 `provider_settings.websearch_tavily_key`，无需复制密钥到插件；搜索后仍由收集模型核对图片。
- `disabled` 不调用搜索服务，仍可使用明确上传、引用或建档的图片。

新图自动参考获取失败时，清空本次自动参考，按角色名和当前要求继续绘制，不能保证原作外观准确度。明确提供的参考或编辑目标无效时会停止。搜索和核对可能产生额外模型费用。

### 后台队列与额度

流程为“提交 → 排队 → 收集参考及绘图 → 主 Agent 领取并查看 → 发送”。同一窗口串行，不同窗口可并发。等待领取的结果释放绘图并发槽位，但仍占总容量。

普通用户在同一平台连接中跨窗口最多有一个未完成任务。管理员不受此个人限制，仍遵守总容量和窗口串行规则。切换或清空原会话后，结果不会注入另一段对话。重启不恢复队列。

额度按**实际发起的图片接口请求**计数。排队时预留，搜索不计绘图次数；开始绘图前取消释放预留。已发请求即使失败或超时也计数，管理员不计数且不占窗口额度。已用记录和窗口限额覆盖值持久化；重置已用额度不会释放排队预留。

同步模式需将 Agent 工具调用超时设为足以覆盖搜索和绘图总耗时，同时检查提供商 HTTP 超时。后台提交立即返回；完成后的 Agent 仍需足够步数领取结果、看图并调用发送工具。

## 使用

```text
画一只蓝色的小鸟，不用搜索参考图。
画一张《作品名》中的角色名，先查原作参考。
把我们刚才讨论的两位角色画在同一张图里。
修改刚才那张图，把背景改成夜晚。
我的绘图任务排到哪里了？
这个群今天还能画几次？
取消我还在排队的画图任务。
```

编辑只从当前会话最近一批已记录的生成结果选择目标，不自动混入其他历史图片。结果已被清理出上下文时需重新上传或引用；当前消息明确上传或引用的图片也可作为参考。

普通用户可取消自己尚未开始的排队任务；管理员可指定任务 ID 取消任意人的排队任务。已开始绘图或等待领取的任务不能通过取消工具撤回。

### 命令与工具

| 命令 | 用途 |
| --- | --- |
| `/生图队列` | 查询当前窗口，管理员可追加 `all` 或指定窗口 UMO。 |
| `/生图额度` | 查询当前窗口额度和 UMO，管理员可追加其他窗口 UMO。 |
| `/生图限额 5 15 [UMO]` | 管理员设置窗口一小时 5 次、24 小时 15 次；省略 UMO 为当前窗口。 |
| `/生图限额 reset [UMO]` | 管理员重置指定窗口已用额度；省略 UMO 为当前窗口。 |
| `/生图建档 角色名\|别名一\|别名二` | 管理员在同条消息上传或引用恰好一张参考图建档，同名可更新。 |
| `/生图角色列表` | 查看已建档角色。 |

主 Agent 工具为 `responses_draw_image`、`responses_image_queue`、`responses_image_quota`、`responses_image_cancel`。自然语言查询及取消无需命令；跨窗口查询和取消他人任务仍按实际发言者的管理员权限检查。

### 队列查询 API

`GET /api/plug/responses_image/queue` 使用 AstrBot Dashboard 登录 Cookie 或 Bearer JWT 鉴权。支持新版扩展 API 的 core 也可使用 `GET /api/v1/plugins/extensions/responses_image/queue`，按 core 登录或 API Key 权限鉴权。

返回任务 ID、窗口 UMO、状态、全局排队序号、时间戳、操作类型和容量统计，不返回绘图原文、图片、参考 URL 或密钥。UMO 含窗口标识，仅供已授权管理端使用。序号不等于完成顺序，同窗口串行可能使其他窗口任务先执行。

### 群聊通知

| 连接器 | 通知方式 |
| --- | --- |
| QQ 官方群聊：`qq_official` / `qq_official_webhook` | 关闭 Markdown 时，简短说明随图片合并发送；启用且支持 Markdown 时另外 @ 原请求者，明确拒绝 Markdown 时降级为普通简短提示。 |
| QQ 官方文字子频道 | 普通文本 @ 原请求者，图片单独发送。 |
| OneBot v11：`aiocqhttp` 群聊 | 标准 `at` 消息段，和图片及简短说明一同发送，无需 Markdown。 |
| 私聊 | 不添加 @。 |

图片通过 AstrBot 内置发送工具构造 `MessageChain`，由 `context.send_message` 路由到原会话，保留 core 的文件检查。QQ 官方群聊的 Markdown @ 使用既有连接单独发送，以避免适配器降级时输出原始 `<qqbot-at-user ... />` 标签。主 Agent 可附一句简短说明，插件不发送参考来源链接。通知失败不会重发已成功送达的图片。

## 排查

| 现象 | 检查项 |
| --- | --- |
| 不调用绘图工具 | 插件已加载、模型支持工具调用，用户明确请求绘图。 |
| 无法看图或发送 | 主聊天模型支持图片输入，启用 `send_message_to_user`。 |
| 原生搜索缺少结果 | 查看 AstrBot Debug 日志；搜索完成不代表有可用图片，可改用 Tavily 或参考库。 |
| 角色外观不准确 | 指定作品和版本，提供参考图或建档；无参考生成不能保证原作准确度。 |
| 同步绘图超时 | 检查 Agent 工具超时和提供商 HTTP 超时，或开启后台模式。 |
| 后台无法回传 | 连接器支持主动消息、原会话仍有效；v4.29.4 兼容流程使用 `local` runner。 |
| 额度不足 | 查询已用和预留；取消未开始任务释放预留，管理员可调整或重置窗口额度。 |

同一条消息不自动重复生成。超时后上游可能仍在处理，应确认状态再发新请求。诊断信息写入 AstrBot 日志，不直接发到聊天窗口。

## 开发与许可

模块划分、测试和可复现打包见 [CONTRIBUTING.md](CONTRIBUTING.md)。

采用 [AGPL-3.0-or-later](LICENSE)。仓库基于 [AstrBot 插件模板](https://github.com/Soulter/helloworld) 创建，图片客户端部分逻辑源自 [AstrBot](https://github.com/AstrBotDevs/AstrBot)。v4.29.4 后台完成流程参考 AstrBot Cron 入口及 [astrbot_plugin_active_message](https://github.com/singularity2000/astrbot_plugin_active_message) 的主动运行方式。开发及发送接口见 [插件开发指南](https://docs.astrbot.app/dev/star/plugin-new.html) 和 [消息发送说明](https://docs.astrbot.app/dev/star/guides/send-message.html)。
