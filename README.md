# LingBot-VLA：从后训练到具身 Agent

这个项目从一个很具体的问题开始：**一个在熟悉场景里能完成动作的 VLA，到了随机化场景和更长的任务里，为什么会失败？应该先改训练、推理，还是加高层 Agent？**

我把这三个层次放在同一套 RoboTwin 评测里比较：先看后训练能不能改善泛化，再看动作块长度如何影响反馈和推理成本，最后看规划、工具和经验记忆能不能帮助低层策略完成长程任务。

这个仓库是项目展示材料，不是从零复现的安装手册。模型权重、数据、仿真资产、集群环境和 API 凭据没有放进来。

## 先看三个结论

**后训练有效，但随机化场景仍然是主要瓶颈。** 在同一批 250 个回合上，M4 是当前候选中最好的 checkpoint：clean 85.6%，randomized 40.0%，整体 62.8%。

**更短的动作块带来更及时的反馈，也带来更多请求。** 固定 20 步得到 66.8%，高于固定 50 步的 63.6%，但模型请求次数从 1,725 增加到 4,073。它是控制反馈和推理成本之间的折中，不是单纯的加速。

**Agent 能组织任务，但不能替代低层控制能力。** 官方 RPent 方案成功完成了 162/250 个回合（64.8%）。它说明规划、工具调用、VLA 执行和反馈可以接通；现有实验条件还不足以把这个数字解释成严格的 Agent 净增益。

## 1. 后训练：模型到底泛化到了哪里

我没有只看训练 loss，而是把 clean 和 randomized 分开统计。每个 checkpoint 都跑 125 个 clean 和 125 个 randomized 回合：

![Checkpoint success rate](assets/figures/checkpoint-success.svg)

| checkpoint | clean | randomized | overall |
| --- | ---: | ---: | ---: |
| M0 | 76.0% | 34.4% | 55.2% |
| M1 | 78.4% | 38.4% | 58.4% |
| M2 | 75.2% | 36.0% | 55.6% |
| M3 | 81.6% | 36.0% | 58.8% |
| **M4** | **85.6%** | **40.0%** | **62.8%** |

M4 作为后续实验的候选权重。另一次 THU 复核得到 B29000 159/250（63.6%）；由于运行环境和协议版本不同，这两个数字分别保留，不能拼成一个更高的成绩。C 路线的 student 和 dynamic-EMA teacher 都是 151/250（60.4%）。这组结果能说明 checkpoint 选择，不能说明某一个训练技巧单独带来了全部提升。

## 2. 推理：动作块应该执行多长

一次预测 50 步动作，并不意味着必须连续执行完 50 步。执行得短一些，可以更早看到环境变化；执行得长一些，可以减少模型调用和等待。我在同一个 B29000 checkpoint 上比较了六种策略：

![Inference strategy comparison](assets/figures/inference-strategies.svg)

| 策略 | 成功率 | clean | randomized | 请求次数 | 相对固定 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 50 步 | 63.6% | 107/125 | 52/125 | 1,725 | — |
| 固定 30 步 | 59.6% | 103/125 | 46/125 | 3,024 | -4.0 pp |
| **固定 20 步** | **66.8%** | 110/125 | 57/125 | 4,073 | +3.2 pp |
| AutoHorizon | 62.0% | 106/125 | 49/125 | 3,483 | -1.6 pp |
| Async 25 | 65.2% | 110/125 | 53/125 | 3,336 | +1.6 pp |
| RTC 25 | 60.0% | 105/125 | 45/125 | 3,561 | -3.6 pp |

这个实验没有得到“自适应策略一定更好”的结论。它更直接地说明：在当前模型和仿真协议下，反馈频率本身就会影响结果；如果要把固定 20 步用于更大规模评测，还要同时考虑请求成本和整体吞吐。

## 3. Agent：什么时候值得加一层规划

低层 VLA 负责把当前观察变成动作，高层 Agent 负责决定下一步做什么、是否调用工具、是否根据反馈重新规划。它解决的是任务组织问题，不是重新训练一个低层控制器。

官方 RPent 评测结果是 162/250：

- clean：107/125
- randomized：55/125

这组实验重点验证了规划器、工具、VLA、仿真器和反馈之间的接口是否能连续工作。旧自定义 Agent 的纯 VLA、无记忆和冻结记忆结果分别为 63.6%、57.2% 和 60.4%。但官方 Agent 与旧纯 VLA 的初态、环境版本和 seed 处理并非完全同源，所以这里不把 64.8% 写成严格的因果提升。

这也是我对 Agent 的判断：如果 VLA 不会抓取或动作已经发生漂移，Agent 不能凭空补出低层能力；它更适合处理长程任务的阶段组织、工具选择和失败恢复。

## 4. 训练曲线和任务级结果

训练 loss 主要用来检查训练是否正常，最终模型选择仍然看留出任务成功率：

![B29000 training loss](assets/figures/training-loss-b29000.svg)

总体平均值还不够。下面的热力图用来查看某个 checkpoint 是否只改善了少数任务，或者是否在某类任务上退化：

![Per-task checkpoint heatmap](assets/figures/per-task-checkpoint-heatmap.svg)

对应的公开数据表：

- [`results/checkpoint_summary.csv`](results/checkpoint_summary.csv)：checkpoint 的 clean/randomized 汇总。
- [`results/per_task_checkpoint.csv`](results/per_task_checkpoint.csv)：25 个任务 × 5 个 checkpoint 的结果。
- [`results/inference_comparison.csv`](results/inference_comparison.csv)：推理策略、请求次数和模型前向时间。
- [`results/training_b29000.csv`](results/training_b29000.csv)：B29000 训练曲线采样。

## 评测为什么可信

评测固定了任务集合、公共场景、checkpoint、seed、精度、去噪步数、动作块长度和成功判据。每个回合都有自己的身份和结果记录。

环境初始化失败、专家规划失败、接口错误和模型正常失败分开记录。只有事先定义的客观环境故障才允许按规则顺延 seed；正常失败不会被删除，也不会从分母里消失。不同服务器或不同环境的结果，如果初态和 seed 不是同源，就只做工程参考，不做严格因果比较。

## 系统结构

```text
任务目标 + RGB/状态
        │
        ▼
规划器 / 经验记忆 ──► 动作工具与恢复决策
        │                       │
        └──────► LingBot-VLA ───┘
                         │
                         ▼
                动作块执行 / AutoHorizon / RTC
                         │
                         ▼
              RoboTwin 状态、成功判据、结果账本
```

模块职责见 [`ARCHITECTURE.md`](ARCHITECTURE.md)，实验口径见 [`docs/experiment-protocol.md`](docs/experiment-protocol.md)，结果说明见 [`docs/results-summary.md`](docs/results-summary.md)。

## 仓库里有什么

- `src/`：VLA 客户端、动作块执行、推理策略、Agent 和结果记录。
- `patches/`：与上游实现之间的最小差异。
- `tests/`：接口、动作执行、结果账本和推理策略测试。
- `results/`、`assets/figures/`：用于查看项目结果的脱敏表格和图表。

仓库不包含权重、数据、视频、仿真资产、集群路径、运行日志或 API 密钥。

## 第三方与许可

LingBot-VLA、RoboTwin、RPent 及相关依赖保留各自版权和许可证。本项目只发布个人实现、最小补丁、脱敏结果和第三方声明；详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`PUBLIC_SCOPE.md`](PUBLIC_SCOPE.md)。
