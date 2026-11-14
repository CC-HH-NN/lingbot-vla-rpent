# Public scope

这个仓库是用于面试和项目展示的公开副本，重点呈现方法、工程贡献和结果证据，不承担从零复现实验的职责。

## Included

- 可阅读的 VLA 推理、动作块、AutoHorizon、RTC、Agent 和评测账本代码。
- 与上游实现之间的最小补丁和架构说明。
- 不含服务器路径、凭据和运行时信息的结果摘要：`results/`。
- 由 Matplotlib 绘制的 SVG/PNG 结果图：`assets/figures/`。
- 8 个未经处理的 Demo 原始 MP4、已有首帧缩略图及场景/SHA 清单：`assets/demos/`。
- 示例配置、测试、实验协议和第三方声明。

## Excluded

- 模型权重、训练数据、仿真器和机器人资产。
- 集群环境、Slurm 作业、运行日志、NPZ、缓存和虚拟环境。
- API key、代理/中继地址、内部服务器路径和 `site.env`。
- 可以直接还原内部运行目录的脚本或凭据。

## How to read the project

建议按以下顺序阅读：

1. `README.md`：项目贡献、核心结果和结论边界。
2. `assets/figures/` 与 `results/`：训练曲线、策略折中和任务级证据。
3. `ARCHITECTURE.md`：模块职责和数据流。
4. `docs/experiment-protocol.md`：评测协议与失败记账规则。
5. `src/`、`patches/`、`tests/`：实现细节和回归证据。
