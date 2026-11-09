<p align="center">
  <img src="docs/assets/cover.svg" alt="cutliner — Agent auto-editing. Manual precision." width="100%" />
</p>

<p align="center">
  <strong>一句需求，Agent 自动交付成片。</strong><br/>
  深度学习定位候选，Agent 按需取证、裁剪与复核，保留手动精调。
</p>

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-2C7E99?style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Temporal_Models-F28C7C?style=flat-square" />
  <img alt="React and TypeScript" src="https://img.shields.io/badge/React-TypeScript-56BDBB?style=flat-square" />
  <img alt="Local first" src="https://img.shields.io/badge/Local_First-Your_Media_Stays_Local-7DAE9C?style=flat-square" />
</p>

<p align="center">
  <a href="#体验">体验</a> ·
  <a href="#工作原理">工作原理</a> ·
  <a href="#实验结果">实验结果</a> ·
  <a href="#开始使用">开始使用</a> ·
  <a href="#能力边界">能力边界</a>
</p>

<p align="center">
  <img src="docs/assets/workspace-agent.jpg" alt="cutliner Agent 自动剪辑界面：大尺寸任务输入区、素材预览、候选池与真实行动记录" width="100%" />
</p>

## 体验

**描述目标 → 自动定位 → 按需取证 → 裁剪与复核 → 交付成片。**

| 提交任务 | Agent 自动执行 | 手动精调 |
| :--- | :--- | :--- |
| 描述画面、声音事件或组合条件，授权后一次启动。 | 自动选择模型、定位候选；证据不足时补看，不符则换候选，通过后裁剪并复核实际成片。 | 一键切换时间线，调整入点和出点，结合音轨波形精剪并导出。 |

- **Agent 行动台**：展示实际工具调用、候选处理与交付状态；未通过不标记为成功，每次交付首个通过复核的片段。
- **双模式工作台**：默认自动剪辑，手动精调独立保留；大尺寸任务输入区、彩色动态背景与液态玻璃控件。
- **本地优先**：媒体和任务记录保存在本机。云端需求解析与视觉复核须明确授权，API 密钥只由后端读取。

## 工作原理

```mermaid
flowchart LR
    Q[描述剪辑目标] --> P[授权后解析任务]
    P --> R{需要什么证据?}
    R -->|画面| V[CLIP + QD-DETR]
    R -->|声音事件| A[CLIP + PANN + 时序融合头]
    R -->|组合条件| H[视觉与声画联合检索]
    V --> C[候选时间区间]
    A --> C
    H --> C
    C --> G[Agent 取证与复核]
    G -->|证据不足| T[补看画面 / 声画工具]
    T --> G
    G -->|不符或无法确认| N[下一个候选]
    N --> G
    G -->|通过| E[自动裁剪]
    E --> F[实际成片复核]
    F -->|通过| D[交付一段成片并停止]
    F -->|未通过| N
    C -.-> M[手动时间线精调]
    D -.-> M
    M --> X[手动导出]
```

### 检索先行，Agent 后置

视觉分支使用 **QD-DETR**：对 CLIP 视觉特征与英文查询进行条件交互，预测相关时间区间。声画分支冻结 **CLIP ViT-B/32** 与 **PANN Cnn6** 编码器，训练轻量时序融合头，定位支持类别中“有声且可见”的事件。复合需求可以使用两个分支的候选证据。

Agent 只在候选产生后进行内容核验，通过 `inspect_frames`、`inspect_av_events` 与 `finish` 形成受预算约束的复核循环。`POST /api/auto-edit` 串联任务解析、模型检索、候选复核、裁剪与输出复核，无需用户逐段选择或另行导出。行动台展示真实执行摘要，不展示内部推理或虚构进度。当前自动交付策略为**首个源片段与成片均通过复核后停止**；候选耗尽或审核受阻时不标记交付成功。

手动精调使用原视频时间坐标，可从 Agent 成片跳回对应源区间。手动导出尊重当前裁剪范围，修改后的范围不会沿用原候选的审核标记；自动任务的记录与已交付成片可从历史恢复。

## 实验结果

声画模块在 **215 个独立原始视频源、28 类事件、3 个预声明随机种子**上的平均结果：

| 方法 | 帧级 Macro F1 ↑ | 事件查询 R@1，tIoU ≥ 0.5 ↑ |
| :--- | ---: | ---: |
| 纯视觉时序模型 | 73.3% | 66.8% |
| **声画时序融合** | **81.0%** | **76.8%** |
| **绝对提升** | **+7.7 个百分点** | **+10.0 个百分点** |

同源数据按视频分组清理，训练、开发和测试来源互不重叠；模型模式在开发集选定，部署固定使用预声明的 seed 42。相对纯视觉对照，两项指标的按源配对 bootstrap 95% 置信区间下界均大于零。

这组结果来自 **AVE 衍生的闭集声画任务**，不是任意长视频检索或整个 Agent 的端到端成功率；音频编码器预训练来源为 AudioSet。原始评测输出与实验配置见 [评测结果](docs/evaluation/heldout-results.json) 和 [训练配置](docs/evaluation/training-config.json)。

## 开始使用

当前本地工作台以 **Windows + Python 3.11 + Node.js 22** 为验证环境。界面不依赖云端部署；先在本机跑起来，再按需配置模型。

### 1 / 获取源码与依赖

克隆本仓库时启用子模块，或在仓库根目录执行：

```powershell
git submodule update --init --recursive
conda create -n cutliner python=3.11 -y
conda activate cutliner
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python studio_server.py
```

打开 **http://127.0.0.1:7861/**。首次启动没有内置素材，点击「导入素材」选择自己的视频。此时可以查看工作台和媒体；执行检索还需要下面的本地模型环境。

### 2 / 配置视觉检索

在同一个仓库根目录，为模型创建独立环境，并下载上游公开权重：

```powershell
python -m venv experiments/video_grounding_local/.venv
experiments/video_grounding_local/.venv/Scripts/python.exe -m pip install -r requirements-models.txt
experiments/video_grounding_local/.venv/Scripts/python.exe experiments/video_grounding_local/prepare_models.py
```

有 NVIDIA GPU 时，请先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择对应的 CUDA wheel。默认也可以使用 CPU，长视频处理会更慢。

在界面中切换「手动精调」，选择「视觉」，展开「检索设置」填写英文视觉查询，即可不调用云端解析完成本地检索。也可以直接使用命令行：

```powershell
python video_main.py --video "D:/Videos/input.mp4" --request "找到人物拿起杯子的片段" --visual-query-en "A person picking up a cup" --retrieve-only
```

### 3 / 可选云端解析与复核

复制 `.env.example` 为 `.env`，在本机填写 DeepSeek 和千问 API 配置。默认「Agent 自动剪辑」模式下，输入需求并点击「开始自动剪辑」，明确授权本次云端解析与画面复核后才会执行。未授权不会发起模型调用；本地模型环境仍须事先准备。

### 声画模型与研究资产

仓库包含声画训练、评估与推理源码，以及经过核对的实验结果。**原视频、特征缓存、预训练权重和训练检查点不随源码分发。** 声画与联合检索需要额外准备经过校验的部署资产，不能只配置 API Key 就运行；`approved_event_model.json` 是资产清单，不是模型本身。

已有完整本地部署时，可通过 `.env.example` 中的 `LOCAL_AV_EVENT_RUN` 与 `LOCAL_AV_EVENT_PYTHON` 指向对应资产和 Python 环境。研究入口为 `prepare_ave.py`、`extract_ave.py`、`train_events.py` 与 `evaluate_events.py`，位于 `experiments/av_grounding_local/`。

## 项目结构

```text
cutliner/
├── frontend/                    React / TypeScript 剪辑工作台
├── studio_server.py             本地媒体与任务 API
├── video_main.py                视频检索命令行入口
├── mini_agent/video/            检索路由、复核 Agent、裁剪与导出
├── experiments/
│   ├── neural_retrieval/        视觉与声画检索执行器
│   ├── video_grounding_local/  QD-DETR 推理与固定版本上游子模块
│   ├── av_grounding_local/     声画时序模型、训练及独立评测
│   └── action_verifier_local/  早期动作核验实验
└── docs/                       项目展示素材与评测依据
```

## 能力边界

- 声画网络支持 **28 类有声且可见的事件**，不识别任意动作、具体台词、画外声或否定声音条件。
- QD-DETR 给出的是候选区间，不保证任意人物关系和复杂动作都判断正确。未检出也不等于不存在。
- 当前界面不是多轨非线性编辑器；不包含转场、字幕生成、多人协作或云端项目同步。
- 完整液态玻璃折射主要在 Chromium 浏览器生效；其他浏览器保留模糊与高光，WebGL 不可用时使用 CSS 背景。

## 致谢

检索与视觉特征基于 [Lighthouse](https://github.com/line/lighthouse)、[QD-DETR](https://github.com/wjun0830/QD-DETR)、[UniVTG](https://github.com/showlab/UniVTG) 和 [CLIP](https://github.com/openai/CLIP)；音频编码参考 [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)，实验数据来源为 [AVE](https://github.com/YapengTian/AVE-ECCV18)。

界面使用 [Vanta](https://github.com/tengbao/vanta)、[Three.js](https://github.com/mrdoob/three.js)、[Ashi Liquid Glass](https://github.com/the-Ashkanian/ashi-liquid-glass) 与 [Lucide](https://lucide.dev/)，视觉方向参考 [Taste Skill](https://github.com/Leonxlnx/taste-skill)、[Interface Design](https://github.com/Dammyjay93/interface-design)、[Formwork](https://github.com/shellcat-com/formwork) 和 [DecorativeBackgrounds](https://github.com/Mamboleoo/DecorativeBackgrounds)。第三方代码与资源保留各自的许可证。
