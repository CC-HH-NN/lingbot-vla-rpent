# LingBot-VLA 后训练、推理优化与 RPent Agent

这个项目把一个视觉语言动作模型（VLA）的后训练、推理策略和高层具身 Agent 放到同一套可追溯的评测框架中。代码关注**可复现实验协议、失败记录和接口边界**，不包含模型权重、数据集或集群运行环境。

## 项目做了什么

- **后训练对比**：比较 SFT、增强训练、ActionEMA、A 路线和 B/C 路线 checkpoint。
- **推理侧对比**：在相同 checkpoint 和任务协议下比较固定动作块、不同固定步长、AutoHorizon、异步动作块和 RTC。
- **分层 Agent**：把 RPent 风格的规划、工具调用、VLA 动作执行、反馈分析和冻结经验记忆串成闭环。
- **工程可靠性**：固定任务/场景/seed/checkpoint，保存逐回合结果，区分模型失败、专家初始化失败和环境/接口错误。

## 系统结构

```text
任务目标 + RGB/状态
        |
        v
规划器 / 经验检索  --->  动作原语与恢复决策
        |                         |
        +------> LingBot-VLA ----> 动作块执行
                                      |
                         RoboTwin 状态、成功判据、细粒度日志
```

主要代码位于 `src/`；测试位于 `tests/`；`patches/` 保存与上游实现之间的最小、可追溯差异。集群提交脚本、模型服务配置、日志、权重和运行快照没有放入公开副本。

## 实验结果（已完成的内部评测）

| 方向 | 条件 | 成功率 |
| --- | --- | ---: |
| 后训练 | B29000（最佳已测 checkpoint） | 62.8%（旧 5090） / 63.6%（THU） |
| 推理 | 固定 50 步 | 63.6% |
| 推理 | 固定 20 步 | **66.8%** |
| 推理 | AutoHorizon | 62.0% |
| 推理 | Async25 | 65.2% |
| 推理 | RTC25 | 60.0% |
| Agent | 官方 RPent（250 回合） | 64.8% |

固定 20 步的请求次数约为固定 50 步的 2.36 倍，因此结果反映的是反馈频率与成功率的折中，不能直接称为纯推理加速。官方 Agent 与纯 VLA 的任务环境并非完全同一批初态，不能据此宣称已经完成严格的 Agent 因果增益证明。详见 [`docs/results-summary.md`](docs/results-summary.md)。

## 本地运行

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-public.txt
PYTHONPATH=src pytest -q
```

完整 RoboTwin/LingBot 运行需要用户自行准备对应版本的仿真器、checkpoint 和 CUDA 环境。规划器从环境变量读取 API 配置：

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://your-compatible-endpoint/v1'
export OPENAI_MODEL='your-model'
```

密钥只应存在于运行终端或密钥管理器中，不能写进配置文件、日志或 Git 历史。

## 公开范围与限制

公开副本刻意不包含：模型权重、数据集、视频、NPZ、`.venv`、集群路径、代理/中继配置、Slurm 作业现场、运行日志和任何 `site.env`。这些内容既体积较大，也包含机器内部信息。结果表是项目内部评测的摘要，不等同于官方排行榜成绩。

## License

本仓库目前只发布项目代码与实验说明；上游 LingBot-VLA、RoboTwin、RPent 及其依赖分别遵循各自许可证。正式发布前请为本项目补充合适的许可证和第三方声明。
