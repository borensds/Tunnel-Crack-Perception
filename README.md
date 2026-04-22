MDC-Net: Tunnel Fracture Intelligent Perception & Quantification System

Disclaimer: Due to commercial confidentiality agreements with industry partners, the proprietary tunnel face dataset and pre-trained weights (.pth) cannot be open-sourced. This repository contains the core algorithmic framework, model architecture, and a dummy dataset for logic verification.

中文版本 (Chinese)

<span id="中文说明"></span>

项目概述 (Project Overview)

本项目是一个面向真实工业落地场景的隧道掌子面裂隙识别与工程参数量化系统 (PoC Demonstration)。

针对边缘端设备算力受限（4GB VRAM）、正负样本极度不平衡、以及长裂隙易断裂等工业痛点，本项目提出了一套涵盖**“极轻量化模型设计、高并发大图推理融合、专家在环 (HITL) 纠错闭环”**的端到端解决方案。

核心性能指标 (Benchmarks)

参数量极低：1.82 M (可轻松部署于无人机载或便携式边缘算力盒子)

边缘端极速：203.45 FPS (本地单卡 RTX 3050 Laptop, 4GB VRAM 极限压榨)

亚像素高精度：物理长度误差 ~4%，宽度提取误差控制在 0.53 Pixel。

核心特性 & 工程亮点 (Key Features)

1. 极致轻量化与全局感知 (MDC-Net Architecture)

D-LKA (空洞大核注意力)：在极小分辨率的瓶颈层 ($16 \times 16$) 引入感受野高达 23 的大核卷积，实现特征图的全局上下文感知，以极低参数代价有效解决细长裂隙的断裂问题。

CoordAtt (坐标注意力)：引入 X-Y 十字方向池化，精确定位裂隙走向，同时强力抑制掌子面粉尘、水渍、探照灯光斑等高频背景噪声。

2. 工业级全 GPU 并发推理 (Industrial Inference)

针对 4K 高清隧道原图，设计了 50% 重叠滑窗裁剪策略。

自研纯 GPU In-place 高斯加权软投票 (Gaussian Soft-Voting) 融合机制。彻底消除大图拼接产生的“十字刀疤”伪影，避免 CPU-GPU 频繁显存通信阻塞。

3. “宁错杀不漏检”的损失函数 (Recall-Focused Loss)

融合 Tversky Loss (控制 $\alpha$ 与 $\beta$ 惩罚权重) 与带 pos_weight 的 BCE Loss。在训练阶段强迫网络对“漏检 (False Negative)”极度敏感，完美契合工程安全领域“零漏报”的业务需求。

4. HITL 专家在环与量化闭环 (Human-In-The-Loop)

C-HTP 拓扑解析算法：实现裂隙骨架的交叉点拓扑解耦、方向共线缝合，输出具备物理意义的长度、均宽、倾角等参数。

交互式修正闭环：基于 Streamlit 开发 PoC 前端。允许领域专家对 AI 的误检进行手动涂抹修正，后台算法实时重算工程参数并更新报表，实现高价值“难例 (Hard Negatives)”的数据回流。

工业级代码导航 (Directory Structure)

为保证代码的模块化与高内聚，本项目核心结构如下：

Tunnel-Fracture-Detection/
├── core/                       # 核心算法与数据结构
│   ├── model.py                # 包含 D-LKA, CoordAtt 及 MDC-Net 组装
│   ├── dataset.py              # 工业级防截断数据增强 (BORDER_REFLECT_101)
│   ├── loss.py                 # Recall-Focused 组合损失函数
│   └── postprocess.py          # C-HTP 图论骨干解析与物理参数量化核心
├── tools/                      # 训练与推理执行流水线
│   ├── train.py                # 支持梯度累加、混合精度的训练脚本
│   └── inference.py            # 防 OOM 大图高斯融合推理引擎
├── demo/                       # 产学研交付层
│   └── app.py                  # Streamlit 可视化闭环分析原型系统
└── dummy_data/                 # 供逻辑验证的示例数据


快速开始 (Quick Start)

1. 环境依赖 (Environment Setup)

git clone [https://github.com/YourUsername/Tunnel-Fracture-Detection.git](https://github.com/YourUsername/Tunnel-Fracture-Detection.git)
cd Tunnel-Fracture-Detection

# 建议使用 conda 或 venv 创建独立环境
pip install -r requirements.txt


2. 模型训练 (Training)

针对 4GB 显存设备优化了梯度累加与 AMP 混合精度训练：

# 执行训练 (配置已在 argparse 中针对 Edge 端做默认优化)
python tools/train.py --model ours --batch_size 2 --target_batch 16 --epochs 100


3. 启动交互式 Web 系统 (Launch Demo)

强烈建议通过可视化界面体验完整的量化闭环与专家纠错流程：

streamlit run demo/app.py


致谢与合作 (Contact)

本项目源于真实的隧道产学研合作 PoC 项目。感谢合作企业提供的一线场景数据与业务输入。如有算法交流、论文探讨或边缘端部署合作意向，欢迎联系：

Email: 17673840652@163.com