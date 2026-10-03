# OverLoCK + YOLO11/YOLO26 V2：多版本、640、原生指标/loss与独立Git仓库

日期：2026-10-03。执行者：DeepSeek Flash。本文是 V1 的增量设计与验收合同；与 V1 冲突处以本文为准。

## 1. 用户要求、决策和本轮范围

用户拟租 **RTX 4090 24GB**，镜像为 Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4，目前尚未租赁且不依赖主代理SSH。新增要求：640×640 等比例 resize 后补边；OverLoCK-T；主干/YOLO系列/YOLO规格独立可选；COCO数据接入；原生Ultralytics评价与YOLO loss；独立本地Git仓库，未来由用户在服务器clone。

决策：

- 支持 `backbone.variant = xt|t|s|b`，`yolo.family = yolo11|yolo26`，`yolo.scale = n|s|m|l|x`，形成 **40个**合法结构组合。YOLO neck/head 成套遵循同一个family/scale，不将同名字母自动绑定到OverLoCK。
- 默认实验改为 **OverLoCK-T + YOLO11s neck/head + 640×640**；提供 T+n 和 B+s 配置。仍保留 V1 的最小三个 1×1 Conv-BN-SiLU adapters，不新增 P2、蒸馏、并行 CNN、额外融合或新损失。
- backbone工厂、YOLO parser、adapter、权重审计、训练/验证CLI、原生criterion、报告和成本统计全部接收这三个独立选项，不能只改YAML名称。默认仍为T+YOLO11s。
- 当前无 GPU；本轮只在现有 conda `3dete` 下做 CPU 验证，并产出目标服务器环境安装说明、预检和 GPU 冒烟脚本。**不租服务器、不连接远端、不安装/升级本地依赖、不启动正式训练或全量评估。**
- GPU 目标已确定为 RTX 4090 24GB；软件 wheel 存在与该代 GPU 的 CUDA 支持不等于实际驱动或完整训练已验证。不要据此写“服务器已验收”。

V1 文档：`/Users/lw/Desktop/ieeeaccessyolo/OverLoCK_Base_YOLO11s_Design_DeepSeekFlash_20261003.md`。既有实现：`/Users/lw/Documents/CNN-Mamba/overlock_yolo11`。

## 2. 工作范围和既有结果的使用

唯一允许持久修改的实现目录为 `/Users/lw/Documents/CNN-Mamba/overlock_yolo11`。目录名为历史兼容名称，不能据此限制只支持YOLO11。你不是唯一使用代码库的人；保留用户和其它任务修改，不执行reset/clean/stash，不修改官方源码、原数据或checkpoint。用户最新指令明确要求本地Git仓库，**允许且要求**在该目录初始化独立仓库并提交本轮工程文件（第11节），不得提交到父项目或操作远程。临时clone验证可使用自行创建的临时目录。

用户要求节省 token：禁止递归遍历项目、图片、权重或旧实验目录。定向阅读现有 `model.py`、`backbone.py`、`attention_backend.py`、`checkpoint.py`、`trainer.py`、相关配置/测试，以及确有需要的官方函数。不得展开 Dino-Mamba 代码或旧结果分析。

V1 任务 `hub-15-murw34bt` 已完成并提交报告。其“CPU 全通过”只是旧测试范围内的执行结果；本次定向阅读已发现第 7 节的问题，因此不能把该结果当作可直接上 GPU 正式训练的认证。

保留 V1 原报告作历史记录；V2 报告放 `reports/v2/`，日志放 `artifacts/v2/`。将本文复制为 `DESIGN_V2.md`；README 明确当前推荐入口。不要重复从零生成整个工程或再次反复全量审计不受改动影响的文件。

只使用这些已知输入：

| 项目 | 绝对路径 |
|---|---|
| 官方检测主干 | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/detection/models/overlock.py` |
| 官方分类实现/README | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/models/overlock.py`、`/Users/lw/Documents/CNN-Mamba/OverLoCK-main/README.md` |
| T 权重 | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/checkpoints/overlock_t_in1k_224.pth` |
| B 权重 | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/checkpoints/overlock_b_in1k_224.pth` |
| YOLO 源码根目录 | `/Users/lw/Documents/CNN-Mamba/ultralytics-main` |
| YOLO YAML | 上述根目录下 `ultralytics/cfg/models/11/yolo11.yaml` 与 `ultralytics/cfg/models/26/yolo26.yaml` |
| SODA YAML | `/Users/lw/Documents/CNN-Mamba/data/SODA10M/soda10m.yaml` |

## 3. 服务器软件兼容性与可迁移路径

### 3.1 已核实的兼容路线

| 项目 | 目标 |
|---|---|
| 系统/架构 | Ubuntu 22.04，Linux x86_64；实际机器预检确认 |
| GPU | RTX 4090，24GB；实际型号、可用显存与驱动仍由预检记录 |
| Python | 3.12 |
| torch | 2.5.1，CUDA 12.4 wheel |
| torchvision | 0.20.1，与 torch 配套 |
| NATTEN | `0.17.4+torch250cu124`，CPython 3.12 Linux x86_64 wheel |
| timm | 优先保留工程已适配的现代接口；服务器模板可采用 0.6.13 的最小修复候选，明确需预检；禁止照抄 0.6.12 到 Python 3.12 |
| 大核卷积 | `use_gemm=False`，使用真实 nn.Conv2d；本轮不编译额外 iGEMM 扩展 |
| MMDetection/MMCV | 此隔离实现不需要安装 |

依据（已核查官方来源）：

- [PyTorch 安装矩阵](https://pytorch.org/get-started/previous-versions/)：2.5.1 + torchvision 0.20.1 有 cu124 安装组合。
- [NATTEN v0.17.4 发布说明](https://github.com/SHI-Labs/NATTEN/releases/tag/v0.17.4)：`torch250cu124` 对应 Torch **2.5.X**，并非仅 2.5.0。
- [官方旧 wheel 索引](https://whl.natten.org/old/)存在 `natten-0.17.4+torch250cu124-cp312-cp312-linux_x86_64.whl`。
- [精确 wheel](https://github.com/SHI-Labs/NATTEN/releases/download/v0.17.4/natten-0.17.4%2Btorch250cu124-cp312-cp312-linux_x86_64.whl)；已核查链接可访问，不要求本机下载约 475 MB 的文件。
- [NATTEN 0.17.4 API](https://github.com/SHI-Labs/NATTEN/blob/v0.17.4/src/natten/functional.py#L1499)仍有 `na2d_av`。
- [timm 旧版 Python 3.11+ dataclass 问题](https://github.com/huggingface/pytorch-image-models/issues/1723)、[0.6.13 修正源码](https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/timm/models/maxxvit.py#L213)。0.6.13 是依赖候选，不是目标 GPU 完整验证结果。

生成服务器专用依赖约束和命令说明，不修改本地 3dete。优先检查镜像中已有 torch/torchvision，已匹配则保留；不要无条件 `pip install -U` 将 torch 换成最新版。安装其他包也应用 constraints 保护 torch/torchvision/NATTEN 组合。不要给用户仅一条无版本约束的 `pip install natten`。

服务器文档可包含以下命令，但本轮不执行：

```bash
# 仅在服务器中对应包缺失/版本不匹配时安装；使用当前目标环境的 python。
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'natten==0.17.4+torch250cu124' --only-binary=:all: --no-deps -f https://whl.natten.org/old/
```

提供 `server_preflight.py`：只读检查 Python、CPU 架构、torch/torchvision、`torch.version.cuda`、NATTEN/timm/einops、torchvision NMS、指定本地 Ultralytics 的来源、GPU 名称/显存/compute capability、实际 CUDA 算子运行及驱动信息。不把 `nvidia-smi` 的“CUDA Version”上限当作 torch 实际编译版本。错误返回非零退出码并解释最小修复路径；不自动重装包。

最小备用路线：若其它依赖仍阻碍 Python 3.12，在该服务器另建 Python 3.10 环境。不要因单次导入失败就擅自大规模换栈或把失败测试标成 skipped/pass。

### 3.2 消除 macOS 路径硬编码

新增一致的路径解析入口：显式 CLI/config 路径优先，其次相对工程根目录的默认值。可提供 `--project-root`/`--ultralytics-root`；具体命名自洽即可。配置相对路径以配置文件位置或明确 project_root 为基准，写清约定，不依赖当前 cwd。

所有 CLI（train/val/smoke/profile/data/preflight）在导入本地 Ultralytics 前先解析其路径。不能在 `model.py` import 时先硬编码 `/Users/lw/...`，事后才读 CLI；也不能删除已导入的 Ultralytics 模块来掩盖多来源混用。来源冲突应明确报错。

服务器迁移保留源码目录相对布局，重新生成数据视图；本机绝对符号链接不能直接搬到 Linux 使用。提供路径参数和示例占位路径，不猜测用户服务器目录。测试从另一个 cwd 调用 CLI，解析结果仍正确。

## 4. 权重类型、变体注册与 adapter 合同

### 4.1 checkpoint 的准确含义

用户下载的 `overlock_{t,b}_in1k_224.pth` 是 **ImageNet-1K 分类模型的预训练 state_dict**，包括主干和分类头；不是带检测 neck/head 的完整目标检测器。用于本项目时只加载可匹配的主干权重，忽略已列明的分类头。

T 文件已经通过 `torch.load(weights_only=True,map_location='cpu')` 定向读取确认：141,560,021 字节、OrderedDict、2737 个 key，含 `head.4.weight=(1000,1024,1,1)` 与 `aux_head.2.weight=(1000,512,1,1)`。这只确认权重类型，完整 T 检测主干匹配审计由执行者完成。

为每个 variant 绑定其 factory、feature_info 与预期 checkpoint 文件名。T/B 文件已存在；XT/S 默认不得下载，不存在时仅允许显式 `pretrained=false` 的结构测试，并标为 random。正式配置要求显式可用权重。选 T 配 B 权重必须报结构不匹配，不得“大多数 shape 不同就跳过”。

继续 safe load，记录 SHA256、容器、前缀规范化、逐 key/numel 匹配、shape mismatch 和白名单。V1 的 B 审计表明分类权重缺 `extra_norm.*` 与检测根模块的 `h_proj.*`；T 须实际确认。不能把所有 missing 自动纳入白名单。报告只写“该文件未含这些检测新增参数”；旧报告“没被 forward 使用所以不会进 state_dict”的因果解释不成立，应在 V2 更正。

默认 `deploy=False`，不加载重参数化 checkpoint 来冒充训练结构。分类权重里的 224 表示其预训练/评估设置，不是主干输入形状锁定。

### 4.2 OverLoCK 原生检测输出

官方返回 x0/x1/x2/x3，YOLO 使用后三个。下表由官方源码核对，运行时仍需断言：

| variant | depth | sub_depth | embed_dim | sub_num_heads | P3/P4/P5 通道 |
|---|---|---|---|---|---|
| xt | 2,2,3,2 | 6,2 | 56,112,256,336 | 4,6 | 112 / 340 / 420 |
| t | 4,4,6,2 | 12,2 | 64,128,256,512 | 4,8 | 128 / 384 / 640 |
| s | 6,6,8,3 | 16,3 | 64,128,320,512 | 8,16 | 128 / 448 / 640 |
| b | 8,8,10,4 | 20,4 | 80,160,384,576 | 6,9 | 160 / 528 / 720 |

其它 factory 参数照官方逐版本核对，不从 Base 任意推算。P3/P4/P5 通道公式为 `(e1,e2+e3//4,e3+e3//4)`。kernel、context pool、RPB、overview/focus 与官方层次保持一致。

### 4.3 YOLO11/YOLO26 neck/head 规格

| scale | depth / width / max_channels | neck 入口（原 4/6/10） | Detect 入口（原 16/19/22） |
|---|---|---|---|
| n | 0.50 / 0.25 / 1024 | 128 / 128 / 256 | 64 / 128 / 256 |
| s | 0.50 / 0.50 / 1024 | 256 / 256 / 512 | 128 / 256 / 512 |
| m | 0.50 / 1.00 / 512 | 512 / 512 / 512 | 256 / 512 / 512 |
| l | 1.00 / 1.00 / 512 | 512 / 512 / 512 | 256 / 512 / 512 |
| x | 1.00 / 1.50 / 512 | 768 / 768 / 768 | 384 / 768 / 768 |

两个系列当前checkout的scale和上述通道表相同，结构仍不能混用。沿用各自YAML和本地parser，其规则为 `make_divisible(min(c,max_channels)*width,8)`。m/l/x会强制C3k2的`c3k=True`；YOLO26的部分neck层本来已为True，最后一层重复数及附加参数也与11不同。不能只改宽度/Detect就称为YOLO26。把family对应YAML的dict设置scale后，交给 `DetectionModel(dict,...)`，从正确初始化的原生模型取得该系列tail。

由 backbone.feature_info 和解析后的 YOLO neck 入口建立 adapters；上述表是回归断言，不应成为分散在 forward/CLI 的硬编码。例：T+s 为 128→256、384→256、640→512；T+n 为 128→128、384→128、640→256。

### 4.4 配置命名与覆盖规则

必须区分 `yolo.scale='s'`（结构规格）和 `train.scale=0.5`（原生几何增强）；拒绝 YAML 重复 key，不允许后值静默覆盖前值。

推荐结构：

```yaml
backbone:
  family: overlock
  variant: t
  weights: ../OverLoCK-main/checkpoints/overlock_t_in1k_224.pth # 路径基准须在实际配置中校正
  pretrained: true
  deploy: false
yolo:
  family: yolo11
  scale: s
  nc: 6
runtime:
  attention_backend: auto
  device: cpu
train:
  imgsz: 640
  rect: false
  multi_scale: false
  scale: 0.5
  amp: false
```

实现 `--config`，CLI 显式参数覆盖配置，默认值不要意外覆盖文件设置；未知 key/variant/scale fail fast。每次启动写出完全解析的配置。既有 `build_overlock_b` 可保留为通用 factory 的兼容包装。默认随机 YOLO neck/head 状态继续如实报告，不从分类权重推断出检测预训练。

## 5. 640×640 的训练、验证与推理协议

输入变换是等比例 resize + letterbox 到**确切 640×640**，不拉伸。设置 `imgsz=640` 之外，还必须检查实际 batch shape；原生验证/预测的 rect/auto padding 可能产生 640×384，不能仅凭配置字段声称固定 640。

- 基准验证/推理使用 `new_shape=(640,640)`、`auto=False`、`scale_fill=False`（具体参数名按本地 API），padding 值 114，RGB→float/255→一次 ImageNet mean/std。
- 明确固定 `scaleup` 策略并与 baseline 对齐；V2 模板允许等比例放大（scaleup=True）。记录 resize 比例、取整和左右/上下 padding，框反变换用实际 ratio_pad，不能重新近似推算。
- 正式训练 `rect=False`、`multi_scale=False`，每批最终为 640×640。可保留与 baseline 对齐的 mosaic/几何增强；输出张量仍固定 640。另提供关闭增强的几何单测，不能把训练增强当作 letterbox 测试。
- 验证器构建、预测入口和测速入口都必须使用相同固定方形策略。确认实际传给 native dataloader 的 rect 值；有必要在自定义 trainer/validator 的对应函数中覆写，而非只在 CLI 写 rect=False。
- T+B 乃至其它版本 640 下的原生 P3/P4/P5 空间应为 80×80、40×40、20×20；Detect 总位置数 A=8400。nc=6 的默认 eval y 应为 `[B,10,8400]`。
- 224→640 不改 OverLoCK 的局部核、固定 7×7 context pooling 或 RPB 参数尺寸。固定 pool 与其 49 通道的权重投影配套，不能把 7 改成 20/40。
- 单元/快速回归可继续使用小尺寸，但结果必须标清；至少有一次带真实 T 权重的 **完整 640 模型前向**，不能用小尺寸成功代替 640 验收。

## 6. 参数、计算量与性能的统计协议

对固定 variant/scale/nc/结构，224→640 **参数量不变**。输入面积比约 8.16；它只是许多空间算子的粗略增长参考，不能直接乘官方分类 FLOPs 得到完整检测 FLOPs。结构变化、分类头移除、检测新增模块、deploy/reparam 状态都会影响统计。

新增 `profile_model.py`，配置与训练同源，输出 JSON/Markdown。YOLO26需区分含O2M/O2O两分支的训练结构参数量、实际eval执行路径和可能的fuse部署结构，不能删头之后的参数与YOLO11未融合参数混比：

1. 参数：按 backbone/adapters/neck/head 和总计分组；区分 total/trainable/buffers；共享参数按唯一对象只计一次。比较同模型 224 与 640 的参数数目相同。
2. 计算量：batch=1、eval、FP32、实际640×640，说明 MACs 与 FLOPs 口径（若采用 1 MAC=2 FLOPs，明确写出）。forward 成本不包含 resize/letterbox/NMS 时必须标注。
3. 自定义 attention/动态权重生成不能漏计为零。对库不支持的 `na2d_av` 单独计数，其主乘加项为 `B*heads*H*W*D*K*K` MACs；其它 einsum/matmul/动态算子逐项核实，避免 native custom handler 与 reference matmul 双计。分辨率导致的内部上采样按实际运行形状计。
4. 输出 profiler 名称/版本、支持/不支持的算子清单、结构状态（deploy=False、是否 fuse）。只覆盖 conv/linear 时命名 `partial_macs`，列遗漏；不得打印看似完整的 GFLOPs。没有可用完整 profiler 时可完成参数和部分 MACs 报告，但将完整 FLOPs 标为未验证，不安装大包凑结果。
5. GPU 延迟/显存留给服务器：实际后端=NATTEN，记录 GPU、torch/CUDA、batch/dtype、warmup、同步和测量次数，区分模型 forward 与含预后处理延迟。CPU reference 不用于宣称部署速度。

可以利用4个主干、2×5个native tail及adapter的独立参数计数生成40项矩阵，避免40次大主干加载。T+11s、T+11n、B+11s、T+26n、T+26s至少与实际组装模型去重总计一致；不下载XT/S权重来统计参数。

参考官方分类表：T 为约 33M、B 为约 95M；这些是 224 分类模型的数量，不是本检测器实测参数。T 比 B 小仍不能据此称为与 YOLO11s 同成本。报告以本工程实测为准。

## 7. 必须同时修正的 V1 接口问题

以下由当前源文件定向阅读发现；修复并补充针对性的回归证据，不能因旧测试通过跳过。

1. **dict forward 错误**：`model.py:forward(dict)` 当前返回预测 dict；本地原生 `engine/trainer.py` 非 compile 分支调用 `loss,loss_items=self.model(batch)`。改为原生 BaseModel 合同：dict→`self.loss(batch,...)`，tensor→预测；检查不递归、loss 是正确向量且可 sum/backward。测试必须调用原生训练器使用的入口，不能只手调 criterion。
2. **构造与设备绑定**：当前构造时传 CPU 并缓存 resolved backend，`.to(cuda)` 后可能继续 reference；显式 natten 又可能在 CPU 构造时提前失败。按真实 forward 输入设备惰性解析/按设备失效缓存，CPU 构造再迁移 CUDA 必须有效。CUDA auto 缺 NATTEN 必须报错，不静默回退。显式 reference 仅供测试对照并标注。
3. **head 初始化**：当前通过 `parse_model` 直接建模块，模型中未见 `bias_init`；另一个 native reference helper 先建默认模型再替换 modules，会丢掉新 Detect 初始化。改为明确 scale 的正确 `DetectionModel(dict)` 路线，初始化 stride/bias/BN 后取 tail，再组装 OverLoCK；head 权重载入后不重置，backbone 装载后不全局初始化。
4. **重复注册**：当前 stem 既作为 `self.stem` 又放进 `self.model[10]`，adapters 又单独注册一次。使用一个规范注册路径，其它访问用 property/普通解析；避免 state_dict 多路径重复、保存膨胀或 reload 歧义，保持 criterion 所需 `model[-1]`。
5. **CPU reference 实际内存与注释冲突**：当前先构造完整 `patches[B,heads,D,H*W,K²]`，之后才按 row_chunk 做 matmul，故没有限制主要中间张量。将 gather/patch 构造也移入行块循环，或逐邻域累加；清理 circular padding 等不成立的数学解释。保留独立 loop oracle/梯度检查，控制640推理内存。reference 数学规则不因优化而变化。
6. **配置冲突及硬编码**：旧 YAML 同时有结构 `scale:s` 和增强 `scale:0.5`，实际 CLI 也未消费整个实验 YAML。实现第 4.4 节的独立命名、严格读取和统一 config→model/trainer/profile。移除 Base/s 和 macOS 路径的硬编码限制。
7. **恢复和保存描述**：旧 CLI 把 backbone weights 设为 required，却声称 resume 可不传，并对所有 resume 直接拒绝。不要留下误导接口。若本轮未实现可靠 resume，明确报“暂不支持”并从 runnable 例子移除；如果实现，须恢复结构配置/优化器/epoch 等，且不得再覆盖主干初始化。自己生成的纯 tensor state_dict+配置至少完成一次重建/reload 一致性检查。

## 8. 验证安排：本轮本地与未来 GPU 分开

### 8.1 本地 CPU 必须执行

环境仍是现有 `3dete`，记录实际版本；它不是 Python3.12/torch2.5.1 目标服务器环境。threads=2、workers=0、FP32，单项昂贵测试预算180秒。修复失败后只重跑受影响测试，不进行无关的大规模测试生成。

| ID | 检查 | 关键通过条件 |
|---|---|---|
| M01 | 配置/路径 | variant/family/scale三者独立；无重复key；CLI覆盖正确；另一个cwd可解析；非法选项拒绝 |
| M02 | 四个主干 | 官方配置/feature_info 一致；逐个小尺寸 forward，XT/S 标 random；不同时驻留多个大模型 |
| M03 | 两系列×五种YOLO tail | 正确family/scale、C3k/repeats和head；同权重同原生特征的tail输出等价；stride/bias及criterion正确 |
| M04 | 40组合接口 | 通道/路由、组装元信息/参数计数一致；不要求40次640forward |
| M05 | T/B 真权重 | 分别 safe load 和审计，互换错误权重必须失败；14个左右白名单以实际参数核对为准 |
| M06 | 固定640 | T+11s与T+26s带真实权重、batch1完整640forward；80/40/20与8400成立，按系列核对最终输出；letterbox正逆变换正确；两模型顺序释放或安全复用同一主干 |
| M07 | 两系列完整训练一步 | T+11s与T+26s小尺寸（例如96/128）、合法batch，通过 `model(batch)` 得到各自native loss，单一计算图反向；主干/adapter/neck/head代表参数梯度有限，执行受控optimizer step；26验证原生epoch更新合同 |
| M08 | attention回归 | 真实行块gather与独立oracle边界/矩形/梯度一致；CUDA不存在时GPU测试明确skipped，缺native的CUDA请求不静默回退（可做调度单元测试） |
| M09 | 成本/状态保存 | 参数分组去重/输入不变性；两系列代表组合报告；安全state_dict+variant/family/scale重建；FLOPs覆盖如实标注 |
| M10 | 工具与原生评价 | train/val/profile/preflight/GPU smoke入口可用；原生Ultralytics validator在1–2图产生预期指标键；不以该随机head指标评估精度；两系列后处理及固定640符合合同 |
| M11 | COCO桥接与Git | 数据映射/image_id/类别/框正确，无cache写回源数据；本地repo提交仅含工程，临时git clone后无原Mac路径依赖的help/随机小模型构造成功 |

M06 若因资源超时，只能记 partial，保留命令供服务器续验；不能缩成224后宣称640通过。完整640反向不要求在本机执行。M07 只是一批小尺寸测试，不启动正式epoch训练、不计算AP、不生成全量数据视图。

按依赖顺序先修复接口和reference内存，再跑T640。T+n/B+s的小尺寸组装回归可复用已验证组件；不得用mock主干替代M05/M06/M07。已有2+2样本的隔离数据视图可复用，不重新遍历数据集。

### 8.2 生成但本轮不执行的服务器验证

新增 GPU smoke 命令：预检 → native NATTEN 与 reference 小tensor数值/梯度对照 → T+s带权重640前向 → 同一图完整loss/backward/optimizer一步 → 峰值显存/耗时 → checkpoint重建。先 FP32；AMP 作为独立后续检查，不能自动启用后把FP32成功视为AMP已验收。

GPU reference 对照使用小tensor与真实 K=5/7/13、边界、H=K/矩形、实际相关 head_dim，分别报告前向/反向误差和合理 FP32 容差。目标为 RTX 4090 24GB，不预填未经实测的 batch16；脚本从batch1开始，用户以后根据实测显存选择batch。提供显式的1/2/4等批量容量探测选项，遇到OOM如实记录而不无限重试；不在本机运行。单卡GPU路线优先，本轮不声称DDP可用。

train/val提供同一实验配置入口和完全可复制命令；未来训练使用完整数据视图，不能默认指向2张图smoke数据。正式train读到smoke标记须明确拒绝或要求显式debug模式，以免租机后误跑小数据集。

## 9. 交付与状态报告

交付现有package增量修改、`DESIGN_V2.md`、README、两系列代表配置、服务器依赖/constraints、preflight/GPU smoke/profile/原生固定640评价入口、针对性测试和独立本地Git仓库。两个系列共享主干/adapter逻辑，但loss、head模式和后处理依据原生合同分流。

在 `reports/v2/` 保存：`environment.json`、`compatibility.json`、`checkpoint_t.json`、`checkpoint_b.json`、`variant_matrix.json`、`validation.json`、`profile*.json` 和 `HANDOFF.md`。以JSON为事实来源，标清计划值/推导值/实际执行值；不能把静态shape推导写成完整forward测试。

报告列出：修改文件、旧问题修复证据、40组合支持边界、T/B权重、两个系列640/训练/原生评价验证、profile完整性、本地Git根目录/分支/commit、未验证的目标GPU/AMP/完整AP/恢复/DDP/部署项。不要覆盖V1旧结果或把旧11/11带入V2结论。

## 10. 最新用户补充：COCO数据、原生YOLO loss与原生Ultralytics指标

本节与其它节冲突时以本节为准，尤其此前沿用V1的YOLO11专用输出断言和COCOeval为主指标的假设。

### 10.1 COCO读取：复用的边界

已经定向核查官方OverLoCK `detection/README.md`、`detection/configs/_base_/datasets/coco_instance.py`和检测训练入口：它能读取COCO，但采用 `mmdet==2.28.2/mmcv-full==1.7.2`，`CocoDataset`、`DefaultFormatBundle/Collect`输出MMCV DataContainer和 `gt_bboxes/gt_labels/gt_masks`。这不等同于可以直接喂给Ultralytics的 `img/cls/bboxes/batch_idx`；还包含默认mask和归一化流程。

用户优先复用适配的dataloader，目标是不重复实现已有功能。这里应复用现有工程 `data.py` 的COCO解析/映射与**原生Ultralytics loader、增强、collate**，提供隔离COCO→YOLO数据视图。不要为了读同一份JSON引入整套旧MMCV/MMDetection或两个不一致的增强/归一化流程。README明确解释“支持COCO文件”和“兼容YOLO训练batch”的区别及采用映射的理由。

保留源COCO作为标注真值，不修改JSON或图像。按JSON的image id/file_name读取，验证categories与用户6类映射；label转换与原图ID可逆追踪。主验证指标由Ultralytics计算，不要因源文件是COCO就换用MMDetection runner/COCOeval作为主评价器。

修正当前 `data.py` 整个images目录symlink分支的风险：视图中的 `images/train`、`images/val` 必须是**真实目录**，内部只给入选图逐文件链接，labels在对应真实目录。避免Ultralytics解析根目录symlink后转回原数据路径，导致labels/cache错位；`--limit`须同时限制可见图片和标签，而不是只生成少量标签却暴露全图目录。全量视图脚本可实现，当前只复用/修正1–2图smoke视图。

统一crowd/ignore/退化框处理并计数。当前报告称该数据crowd/ignore为0，未来实际输入仍须检查；若非0，不能无声当正样本。未知类别、缺图、重复ID、越界路径明确失败。检查native loader实际读到的样本数、标签计数、batch类ID和框归一化，且源数据目录没有新cache。

### 10.2 两个系列的head与loss合同

下面以当前本地checkout为准，不能用网上旧版YOLO接口替代。

| 项目 | YOLO11 | YOLO26 |
|---|---|---|
| YAML | `11/yolo11.yaml` | `26/yolo26.yaml` |
| reg_max | 16 | 1 |
| 分支 | one-to-many | one-to-many + one-to-one |
| criterion | 原生 `v8DetectionLoss` | 原生 `E2ELoss`，不是旧 `E2EDetectLoss` |
| 第三个回归项 | 原生DFL | 当前实现的归一化L1，不能假定为零DFL |
| 原生主要推理 | 密集预测→NMS | end-to-end O2O/top-k，默认无需IoU NMS |

复用相同family原生DetectionModel的criterion选择或等价原生 `init_criterion` 调用，不手写新loss、assigner或训练权重曲线。YOLO26当前E2ELoss内部使用O2M topk10、O2O topk7/topk2=1，带分支权重schedule；保留原实现，不能强行改成“topk1等权求和”。

原生trainer在每个完成epoch末、验证之前调用 `criterion.update()`；保存EMA模型时移除criterion，resume时恢复其updates进度。自定义wrapper需让这个生命周期继续工作；仅正确构造criterion并不足够。日志直接保留native返回字典及其语义（当前E2ELoss日志items反映O2O，而总loss是加权两分支）。空标签batch和loss.sum().backward都要验证。YOLO26的O2O输入特征在原生head中detach，保留该梯度语义，主干由原生O2M路径获得梯度；不要为“所有分支都回传主干”而移除detach。当前L1仍乘hyp.dfl，沿用原实现，不更名后错接超参。

保留 `model.end2end` property/setter、`set_head_attr`、max_det等接口。Detect构造时虽有O2O分支，实际end2end推理模式仍由原生validator/args.nms选择；不要误判为原生bug，也不要统一强制 `_end2end=False`。直接smoke推理时显式指定该family预期模式并记录。

640下两系列P3/P4/P5均对应8400个位置，但最终输出不同：YOLO11常规eval为 `[B,4+nc,8400]`（另含raw返回）；YOLO26 end-to-end通常为 `[B,K,6]`，列为xyxy/conf/class，K受max_det等控制。按原生实际接口断言，不对YOLO26套 `[B,10,8400]`。训练raw分支也按各自格式处理。

### 10.3 主评价体系锁定Ultralytics

通过当前本地/随仓库固定的Ultralytics原生DetectionValidator与DetMetrics计算，输出它的标准precision、recall、mAP50、mAP50-95、每类指标与原生results_dict；主键名以该checkout实际返回为准。不自行重写AP积分、类别平均或IoU匹配。

默认SODA nc=6，严格同一val拆分、类别名/映射、640 letterbox、max_det、评估conf等。原生detection评价默认conf为0.001，应显式记录，不用混淆矩阵/展示图片常用0.25阈值删掉低分预测。YOLO11按原生NMS，YOLO26按原生end-to-end路径；YOLO26不再叠加一次自写NMS。报告nms/end2end/max_det/conf/iou及哪些参数对当前模式实际生效。主评估默认save_json=False，避免某些原生COCO分支触发额外COCOeval并替换主指标；如果另行导出COCO预测/评估，标为独立诊断并保留原始DetMetrics结果。

V1历史52.21%/53.92%来自标准COCO AP50，不得直接放进本轮原生Ultralytics表格当同协议结果。未来baseline须在本轮同一Ultralytics版本与评估设置重新验证。COCOeval、面积分桶、候选覆盖可作另行注明的补充诊断，本轮不新增另一套主评估。

本轮用1–2图执行validator只检验接口和指标键，结果不代表模型精度。特别核实native验证路径是否触发auto-backend/fuse、模型info/stride/end2end、EMA调用；不能仅写一个返回假metrics的演示脚本。

## 11. 独立本地Git仓库与用户自行clone的迁移方式

用户已明确回复：**尚未建立远程仓库，先整理本地仓库。** 不创建GitHub/Gitee仓库、不设置猜测的origin、不推送或公开任何代码，不尝试SSH服务器。

在当前实现目录 `/Users/lw/Documents/CNN-Mamba/overlock_yolo11` 建立独立Git仓库。初始化前精确检查该目录自己的`.git`，以及父项目对此目录是否已有tracked文件；不把父目录的git root误认为本任务repo。若已有用户建立的独立repo则沿用；若目标已有跟踪冲突，保留并报告，不改父项目index。

源码依赖必须可重现：本项目当前用的是本地Ultralytics快照（先前记录8.4.148），不能发布一个只在作者Mac上依赖兄弟目录或任意pip最新版的仓库。优先将**指定的**本地 `ultralytics-main/ultralytics` 纯源码作为 `vendor/ultralytics/ultralytics` 收入repo，配置默认导入该vendor根；OverLoCK检测适配源码已在package中，保留来源与许可证说明。文件复制属于必要打包，可对白名单源码树做自动复制，不逐文件读取/输出、不复制整个CNN-Mamba树；忽略`.git/__pycache__/*.pyc/权重/数据/缓存/运行产物`。不要从网络另拉一个不同版本替代已经核对的源码。

记录vendor来源URL、原版本、源文件/树manifest哈希及必要本地差异；保留源码中的版权、许可证文件和第三方声明，不擅自给第三方代码改许可证。README如实写“固定源码快照”，不伪造一个未验证的upstream commit。

建议仓库至少包含package、vendor必需源码、configs、scripts、tests、依赖constraints、README、DESIGN_V2、第三方来源说明和小型报告摘要。`.gitignore`排除：全部`.pth/.pt/.ckpt/.onnx/.engine`，数据/图片视图、cache、logs、runs、artifacts、完整大JSON、环境目录、密钥和本地机器专属配置。checkpoint manifest只记录文件名/哈希/来源/预期放置位置，不提交checkpoint。

明确列入追踪的白名单文件路径后再stage，禁止对父项目`git add .`。初始分支可用 `codex/overlock-yolo-portable`；在使用者已有Git身份可用时提交一次本工程变更。不得杜撰作者身份；若未配置则保留已初始化/暂存仓库并准确说明阻塞。这条局部Git提交是用户本轮授权，覆盖V1“不commit”的旧限制。

通过本地临时 `git clone` 验证仓库可独立取出；在克隆目录运行CLI help、配置读取、无预训练的小模型构造和指定vendor来源断言，不能偷偷回退作者Mac上的兄弟源码。需要数据/权重的命令明确报所缺资源，不自动下载。记录commit、tracked文件数、总仓库体积、异常大文件检查及clone验证结果；不把所有文件路径打印进报告。

README给未来流程：用户创建远程→添加其实际URL→push→服务器clone→按锁定依赖安装→单独放置COCO数据/下载权重→配置路径→preflight→GPU smoke→正式训练/原生val。远程URL用占位符且注明尚未提供；不能声称服务器现在就能从远程clone。数据和checkpoint不随Git传输。租赁/登录方式不影响本轮代码交付。

主代理派发本次任务后不轮询，等待用户通知。执行者独立完成授权范围内工作后结束；不再委派其它agent、不升级Pro、不等待主代理确认、不创建自动化。失败时完成独立可做事项，留下明确阻塞与最小下一步。
