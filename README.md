# LingBot-VLA：后训练、推理优化与具身 Agent

基于 LingBot-VLA 2.0 与 RoboTwin 2.0，完成双臂操作 VLA 的多任务后训练、推理时动作执行策略比较，以及 RPent 风格高层 Agent 接入。项目围绕一个核心工程问题展开：**如何判断一次失败应该通过训练、推理调度还是任务规划来解决。**

## 项目概览

| 研究问题 | 对应实验 | 主要结论 |
| --- | --- | --- |
| 后训练能否改善多任务泛化？ | SFT、增强训练、ActionEMA、B/C 路线 checkpoint 对照 | B 路线 B29000 是当前候选中表现最好的权重，但 randomized 仍明显低于 clean |
| 动作块执行长度如何影响闭环性能？ | Fixed50/30/20、AutoHorizon、Async25、RTC25 | 固定 20 步成功率最高，但推理请求约增加 2.36 倍 |
| 长程任务是否需要高层规划？ | 纯 VLA、官方 RPent Agent | Agent 可以组织任务和调用工具，但不能替代低层动作能力 |
| 官方经验记忆是否有效？ | 纯 VLA 与官方 RPent Agent 对照 | 官方 Agent 结果与纯 VLA 接近，现有条件不足以证明严格净增益 |

## 评测设置

| 项目 | 设置 |
| --- | --- |
| 基础模型 | LingBot-VLA 2.0 6B |
| 训练数据 | RoboTwin clean 数据，50 个任务、2,500 条示教、549,787 帧 |
| 后训练 | 4×H100、FSDP2、Muon + AdamW、global batch 256 |
| 评测任务 | 25 个任务、clean/randomized 两种 setting、每种 setting 5 个公共场景 |
| 单次推理 | BF16、10 次去噪、batch size 1、预测动作窗 50 |
| 成功判据 | RoboTwin 原生 `eval_success`，逐回合保存结果 |

所有主结果都保留成功数和有效回合数。环境初始化失败、专家规划失败、接口异常和模型正常失败单独记账；只有预先定义的客观环境故障才允许按协议顺延 seed。

## 后训练与 checkpoint 选择

### 训练路线

第一阶段先建立一个可比较的 SFT 基线，再根据 clean 与 randomized 的差距设计后续路线。后续实验同时改变了训练时的增强、teacher 约束、动作 EMA 和任务采样，因此最终结果评价的是**整条训练配方**，不能把收益归因到单一技巧。

| 路线 | 训练阶段 | 训练变化 | 评测权重 |
| --- | --- | --- | --- |
| SFT 基线 | 0 → 16,000 | 官方 clean 示教上的监督微调 | SFT16000 raw |
| 增强训练 | 16,000 → 23,000 | 颜色、噪声、模糊、JPEG 等视觉增强 | Enhanced23000 raw |
| ActionEMA | 23,000 → 26,000 | 维护动作模块的 shadow EMA | ActionEMA26000 |
| B 路线 | 26,000 → 29,000 | 固定 clean teacher，对增强视图施加一致性约束 | **B29000 student** |
| C 路线 | 26,000 → 29,000 | 使用 dynamic EMA teacher | C29000 student / teacher |

### 结果

![Checkpoint success rate](assets/figures/checkpoint-success.svg)

四份基础 checkpoint 和两份 C 路线权重均按 250 个回合统计，每份包含 125 个 clean 和 125 个 randomized 回合：

| 权重 | clean | randomized | overall |
| --- | ---: | ---: | ---: |
| SFT16000 raw | 95/125 · 76.0% | 43/125 · 34.4% | 138/250 · 55.2% |
| Enhanced23000 raw | 98/125 · 78.4% | 48/125 · 38.4% | 146/250 · 58.4% |
| ActionEMA26000 | 94/125 · 75.2% | 45/125 · 36.0% | 139/250 · 55.6% |
| **B29000 student** | **107/125 · 85.6%** | **50/125 · 40.0%** | **157/250 · 62.8%** |
| C29000 student | 102/125 · 81.6% | 49/125 · 39.2% | 151/250 · 60.4% |
| C29000 dynamic-EMA teacher | 106/125 · 84.8% | 45/125 · 36.0% | 151/250 · 60.4% |

B29000 比 SFT 基线多成功 19 个回合，整体提高 7.6 个百分点。它是这组候选中用于后续推理和 Agent 实验的权重。THU 上重新冻结场景后的 B29000 复核为 159/250（63.6%）；两批场景不同，因此分别报告。

C 路线的 student 和 dynamic-EMA teacher 总分相同，但成功回合并不完全一致；两者都没有超过 B 路线。这说明 dynamic teacher 在当前协议下没有带来可见的总体收益，也说明“加入 EMA”本身不能直接等同于性能提升。

训练曲线用于检查训练是否稳定，模型选择仍然以留出任务成功率为准。粗线是 101 步滑动均值；9,376→15,000 步没有保留原始日志，因此图中保留空档，没有插值：

![Full archived training loss from SFT to B29000](assets/figures/training-loss-full.svg)

## 推理时动作执行策略

VLA 一次可以预测一段动作，但实际执行多长再重新观察，会改变反馈频率、动作连续性和模型调用成本。因此固定 B29000 checkpoint，比较六种执行策略：

![Inference strategy comparison](assets/figures/inference-strategies.svg)

| 策略 | 执行方式 | 成功率 | clean | randomized | 请求次数 |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed50 | 每次执行 50 步 | 63.6% | 107/125 | 52/125 | 1,725 |
| Fixed30 | 每次执行 30 步 | 59.6% | 103/125 | 46/125 | 3,024 |
| **Fixed20** | 每次执行 20 步 | **66.8%** | 110/125 | 57/125 | 4,073 |
| AutoHorizon | 根据动作注意力选择执行前缀 | 62.0% | 106/125 | 49/125 | 3,483 |
| Async25 | 执行旧动作时异步生成下一块 | 65.2% | 110/125 | 53/125 | 3,336 |
| RTC25 | 异步生成并用未执行前缀引导新动作块 | 60.0% | 105/125 | 45/125 | 3,561 |

Fixed20 是本批实验最高分，但请求次数约为 Fixed50 的 2.36 倍。AutoHorizon 和 RTC 的结果评价的是当前 LingBot 适配、延迟和仿真接口，不能直接当作原论文在其他模型和平台上的复现结果。

这组结果支持的判断是：执行窗会影响闭环成功率，短窗可能通过更频繁反馈改善表现；它不支持“越短越好”或“自适应策略一定更优”。Fixed30 低于 Fixed50 和 Fixed20，执行长度与策略轨迹之间存在非单调关系。

## 具身 Agent

### Agent 与 VLA 的职责

Agent 和 VLA 处在不同层级：

```text
Agent：理解长程目标，决定下一步做什么，选择工具，根据反馈重新规划
  │
  ▼
VLA：根据当前 RGB、机器人状态和局部目标生成动作序列
  │
  ▼
RoboTwin：执行动作并返回新观测、状态和原生成功判据
```

Agent 不替换 VLA 权重，也不直接生成关节动作。它决定什么时候调用 VLA、什么时候调用运动工具，以及失败后是否继续或重新规划。

### 记忆从哪里来

**官方 RPent Agent** 使用官方发布的 RPent memory 资源，保持官方规划器、工具和提示词，只对 B29000 的 qpos14 VLA 接口做适配。记忆是任务经验和工具使用参考，不是写入 VLA 参数的训练结果；换一个 VLA checkpoint 后，仍需重新验证其中的物体、坐标、工具语义和动作经验是否适用。

### Agent 结果

以下结果均使用 B29000 作为底层 VLA：

| 条件 | clean | randomized | overall |
| --- | ---: | ---: | ---: |
| 纯 VLA | 107/125 · 85.6% | 52/125 · 41.6% | 159/250 · 63.6% |
| 官方 RPent Agent | 107/125 · 85.6% | 55/125 · 44.0% | 162/250 · 64.8% |

官方 Agent 的 162/250 与纯 VLA 的 159/250 不能直接作为严格因果对照，因为环境版本、初态 RGB 和 seed 处理并不完全同源。

当前证据更适合支持这样的结论：官方 Agent 已经形成了规划—工具—VLA—反馈的完整工程闭环；它是否在某类长程任务上稳定优于纯 VLA，还需要同一环境、同一初态和同一 seed 的严格对照。

## 任务级分析

总体成功率不能说明哪些任务真正受益。下面的热力图保留每个任务、保留 checkpoint 和官方 Agent 的结果，用于定位路线选择或高层规划带来的局部改善与退化：

![Per-task checkpoint heatmap](assets/figures/per-task-checkpoint-heatmap.svg)

公开结果表：

- [`results/checkpoint_summary.csv`](results/checkpoint_summary.csv)：checkpoint 的 clean/randomized 汇总。
- [`results/per_task_checkpoint.csv`](results/per_task_checkpoint.csv)：25 个任务 × 4 个保留 checkpoint 与官方 Agent 的逐任务结果。
- [`results/inference_comparison.csv`](results/inference_comparison.csv)：推理策略、请求次数和模型前向时间。
- [`results/training_full.csv`](results/training_full.csv)：从 SFT 到 B29000 的完整已归档训练曲线，保留日志缺口。
- [`scripts/plot_showcase.py`](scripts/plot_showcase.py)：从公开结果表重绘所有图表。

Agent 任务级结果暂不按“抓取/放置/堆叠”主观归类，避免人为分类先于数据。当前 README 只报告可直接核对的 setting 和总结果。

## 工程实现

- 训练侧：后训练路线、checkpoint 保存/恢复和动作 teacher 约束。
- 推理侧：固定动作块、AutoHorizon、异步动作队列和 RTC 前缀引导。
- Agent 侧：结构化规划、动作工具、反馈分析、官方记忆和 qpos14 VLA 适配。
- 评测侧：公共场景冻结、逐回合账本、seed 顺延规则、异常分类和断点续跑。
- 系统侧：模型服务与仿真环境隔离，避免每个回合重复加载权重；实际前向保持 batch1，双环境并不等同于 batch2 推理。

## 结论与限制

当前项目得到的结论是：

1. B29000 是当前后训练候选中最好的底层权重，但 clean 与 randomized 之间仍有明显差距。
2. Fixed20 是已测推理设置中的最高分，但用更多推理请求换取更高反馈频率。
3. AutoHorizon 和 RTC 在当前 LingBot 适配上没有超过 Fixed20，不能据此否定它们在其他模型或系统上的方法价值。
4. 官方 Agent 和记忆的完整工程链路已经跑通，但尚未完成同环境、同初态和同 seed 的严格因果对照。

因此，项目展示的是一套从训练、推理到高层规划的系统分析，而不是把不同实验表中的最高数字相加成一个“最终提升”。

## 仓库内容

- `src/`：VLA 客户端、动作块执行、推理策略、Agent 和结果记录。
- `patches/`：与上游实现之间的最小差异。
- `tests/`：接口、动作执行、结果账本和推理策略测试。
- `results/`、`assets/figures/`：结果表和展示图。
- `docs/`：实验协议、结果说明和证据边界。

## 第三方与许可

LingBot-VLA、RoboTwin、RPent 及相关依赖保留各自版权和许可证，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

图表的数据完整性、缺失值和可读性检查参考 Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065 (2026)：<https://arxiv.org/abs/2609.00065>。
