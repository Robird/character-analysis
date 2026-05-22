# Native-Agentic Model Design Notes

## 核心区别
- Instruction Model: 外触发、单回合、任务闭环。驱动力 = user prompt
- Native-Agentic Model: 内触发、连续状态、无外部 EOS。驱动力 = 内部状态匮乏

## 实现思路（讨论中）
1. SFT 冷启动：角色数据 → 合成 (上下文, 工具, CoT, 内部动作, 外部动作) 数据点
2. DPO：反派角色做"同一情境不同选择"的偏好配对
3. 激活工程（快速路径）：找"被动等待方向"向量 → 抵消，或找角色决策时的 hidden state → 叠加"主动性方向"
4. 需同时改模型内部驱动力 + 外部调度循环（scheduling: 等 user → 模型自主选择时机调用工具）

## 相关项目
- `https://github.com/p-e-w/heretic` — 激活空间向量算术，可用于模式抑制/增强
