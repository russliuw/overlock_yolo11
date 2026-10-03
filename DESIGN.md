# OverLoCK-Base + YOLO11s：设计与实施交接

日期：2026-10-03。执行者：DeepSeek Flash。当前阶段：实现、CPU 冒烟验证、留下可复核证据；不启动正式训练。

## 1. 任务目标和结论边界

构建一个独立的道路目标检测实验：**官方 OverLoCK-Base 检测主干 → 三个通道适配层 → 本地原生 YOLO11s neck/head → 原生检测损失**。使用用户已经下载的 Base 预训练权重，在 SODA10M 六类任务上打通数据、预测与损失接口。

这次检验的假设是：保留主干原生的分层空间信息、官方 overview/focus 计算及检测输出，比从单一最终特征重建金字塔更适合当前任务。它是待验证假设；现有 CNN-Mamba 的小目标几何候选不足，也不能直接当作 OverLoCK 或 YOLO11s 已经存在的同一缺陷。

- 本轮交付：代码、配置、明确的训练入口、轻量 CPU 验证、机器可读报告及操作说明。
- 不在本轮执行：完整训练、完整验证集评估、远端连接、GPU 测速、导出部署、Dino-Mamba 验证、额外模块创新。
- Base 是较大主干。官方分类模型规模不能直接当作新检测模型规模；必须实际统计新模型各部分参数，不能称为与 YOLO11s 同参数量比较。
- 不承诺 AP50 达到 60%。本轮通过只证明指定范围内的实现正确性，不能证明检测精度、训练稳定性或部署成本。
- 历史 COCO AP50：模型 A 为 52.21%，YOLO11s 参照为 53.92%。仅作背景；两个模型参数不同，且必须核实未来实验的评估协议，不能混用 native 指标和 COCO 指标。

## 2. 工作边界、路径与最小读取原则

| 用途 | 已确认的绝对路径 |
|---|---|
| 项目根目录 | `/Users/lw/Documents/CNN-Mamba` |
| 官方 OverLoCK 检测实现 | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/detection/models/overlock.py` |
| 官方 OverLoCK 分类实现，仅供预训练/归一化核对 | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/models/overlock.py` |
| 已下载的权重 | `/Users/lw/Documents/CNN-Mamba/OverLoCK-main/checkpoints/overlock_b_in1k_224.pth` |
| 本地 Ultralytics 源码根目录 | `/Users/lw/Documents/CNN-Mamba/ultralytics-main` |
| YOLO11 配置 | `/Users/lw/Documents/CNN-Mamba/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml` |
| SODA10M 数据配置 | `/Users/lw/Documents/CNN-Mamba/data/SODA10M/soda10m.yaml` |
| 数据目录 | `/Users/lw/Documents/CNN-Mamba/data/SODA10M` |
| 原有实验，保留原样 | `/Users/lw/Documents/CNN-Mamba/architecture_v1` |
| **本任务唯一新增代码/产物目录** | `/Users/lw/Documents/CNN-Mamba/overlock_yolo11` |
| 本设计文档 | `/Users/lw/Desktop/ieeeaccessyolo/OverLoCK_Base_YOLO11s_Design_DeepSeekFlash_20261003.md` |

已知 checkpoint 大小为 395,257,645 字节；加载前自行记录 SHA256，不以大小替代身份核对。

用户明确要求节省 token，不要遍历这些文件。遵守：

1. 只精读本文件列出的代码入口和实际遇到问题时必须阅读的被调用函数。禁止递归 `find`、`ls -R`、全项目 `rg --files`、无边界全文搜索、枚举图片目录。
2. 定位代码使用针对已知文件的 `rg -n` 和短 `sed` 片段。COCO JSON 可按需求读取一次并汇总，只打印计数、字段和少数样本，不打印全文或大量框。
3. 不读取 Dino-Mamba 代码，不分析既有实验结果，不扫描所有 checkpoints，不查找远程服务器。
4. 你不是唯一使用本代码库的人。项目存在用户未提交的修改；不得撤销、覆盖、整理这些修改，不做 reset/clean/stash/commit。
5. 不修改官方源码、`architecture_v1`、数据与 checkpoint；所有代码、转换视图、缓存、测试输出放在新目录。若目标目录已存在，只检查其中与本任务有关的入口，保留已有工作并报告冲突。
6. 不安装或升级依赖，不下载其他权重，不使用联网训练跟踪服务。遇到权限或必要依赖阻塞，保存已完成内容与准确错误；不得绕过权限或伪造成功。

## 3. 已确认的本地环境

Python 环境：`conda run -n 3dete python`，解释器为 `/opt/miniconda3/envs/3dete/bin/python`。本机没有可用于本任务的 GPU，明确使用 `device=cpu`、FP32、`amp=False`、`workers=0`。

模块发现检查：torch、torchvision、timm、einops、ultralytics、yaml、PIL 存在；natten、mmcv、mmengine、mmdet 不存在。这只是模块可发现性，不是导入、ABI 或运行验证。执行者需记录 Python/torch/torchvision/timm/einops 版本及实际导入路径，并验证必要导入。

导入 Ultralytics 前显式把上述本地源码根目录放在导入搜索路径首位；断言 `ultralytics.__file__` 位于该目录。不能悄悄使用 conda 中另一个版本。将本任务产生的 Ultralytics 设置、缓存等指向新目录；不要为了导入而安装整套 MMDetection。

## 4. 版本 V1 的固定架构

```text
RGB 图像 → 原生检测预处理/letterbox → float [0,1]
                                      ↓ 仅一次 ImageNet mean/std 归一化
                           官方 OverLoCK-Base 检测主干
                                      ↓
                    C3 / stride 8 / 160 channels
                    C4 / stride 16 / 528 channels
                    C5 / stride 32 / 720 channels
                                      ↓
                    1×1 Conv-BN-SiLU × 3，保持空间尺寸
                          256 / 256 / 512 channels
                                      ↓
                    原 YOLO11s 第 11–22 行 neck
                                      ↓
                    128 / 256 / 512，stride 8/16/32
                                      ↓
                    原第 23 行 Detect，nc=6
```

完全替换 YOLO 原第 0–10 行 backbone，包括原 SPPF/C2PSA；不要又在 OverLoCK 后面追加它们。暂不加入 P2 检测头、CNN 并行支路、额外注意力、BiFPN、门控、特征蒸馏、新损失或新的标签分配。

### 4.1 OverLoCK-Base 的精确配置和特征来源

以**检测版** `overlock_b` 为依据，而不是调用分类版最后的 logits 或自己从中间 block 随意挂钩：

```yaml
depth: [8, 8, 10, 4]
sub_depth: [20, 4]
embed_dim: [80, 160, 384, 576]
kernel_size: [17, 15, 13, 7]
mlp_ratio: [4, 4, 4, 4]
sub_num_heads: [6, 9]
sub_mlp_ratio: [3, 3]
smk_size: 5
deploy: false
use_gemm: false
drop_rate: 0.0
drop_path_rate: 0.0  # V1 显式记录，不把不同训练配方的值暗中带入
use_checkpoint: [0, 0, 0, 0]
```

保留 overview/focus 分支、`high_level_proj`、`patch_embedx`、`h_proj`、相对位置偏置、官方归一化与小空间尺寸处理。`embed_dim` 不是检测最终输出通道：后两个输出还包含上下文通道 `576/4=144`。

| 官方检测输出 | stride | 实际通道 | 640×640 输入的空间尺寸 | 用途 |
|---|---:|---:|---:|---|
| x0 | 4 | 80 | 160×160 | 主干正常计算，不进入本版 neck |
| x1 | 8 | 160 | 80×80 | C3/P3 来源 |
| x2 | 16 | 528 | 40×40 | C4/P4 来源 |
| x3 | 32 | 720 | 20×20 | C5/P5 来源 |

640 只是接口推导示例，不要求本机实际跑 640。执行时对输出通道、步长、空间尺寸和有限值做断言。V1 公共输入约定为已 padding 到 32 倍数的 BCHW；非法尺寸给清晰错误，不能插值修补 neck 的尺寸冲突。

官方动态块在特征尺寸小于所需 kernel 时会内部插值，并在后面恢复原尺寸；这是原实现的一部分，必须保留。不能用“我们禁止重建金字塔”为理由删掉这些内部操作。

官方检测构造器先创建再删除 `head/aux_head`，其中 `use_ds=False` 可能触发删除不存在属性的问题。保留可工作的默认构造语义，或在隔离副本中明确消除无用分类头构造并记录差异，不改变剩余参数名及前向计算。

### 4.2 归一化和适配层

- 检测图像为 RGB，loader/native preprocess 只负责转 FP32 和除以 255。
- 在 backbone 边界做一次 `(x - mean) / std`：mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]`；以 buffer 保存，支持设备迁移。输入接口和数据脚本需避免重复归一化。
- 不做分类训练的中心裁剪，不将训练分辨率固定为 checkpoint 文件名中的 224；保留检测几何变换。
- 三个 adapter 分别为 `Conv2d(160,256,1,bias=False)+BN+SiLU`、`Conv2d(528,256,1,bias=False)+BN+SiLU`、`Conv2d(720,512,1,bias=False)+BN+SiLU`。
- 优先使用本地 Ultralytics 的 `Conv(c1,c2,k=1,s=1)` 以匹配 YOLO 分支初始化和 BN 约定。只改变通道，不做 resize、不跨层融合。
- 不能在装入 OverLoCK 后对整个混合模型调用 YOLO 的全局初始化：它可能改动已加载权重或 backbone 的 BN 属性。原生 YOLO 部分先独立构造，再装配/加载各部分。

### 4.3 YOLO11s 精确路由与当前接口

必须显式指定 `scale='s'`、`nc=6`。直接读取 `yolo11.yaml` 而省略 scale 可能落到 n。s 的 depth=0.5、width=0.5、max_channels=1024。

从本地 `DetectionModel` 构造/取得原生模块，再复用第 11–23 行，避免手写一个“类似 YOLO”的 neck。原 backbone 三个入口是第 4/6/10 行，通道为 **256/256/512**；**128/256/512 是 neck 到 Detect 的通道**。

| 原行号 | 运算/来源 | 实际输出通道 | stride |
|---|---|---:|---:|
| 11 | nearest 上采样第 10 行 ×2 | 512 | 16 |
| 12 | concat(11,6) | 768 | 16 |
| 13 | 原生 C3k2(False) | 256 | 16 |
| 14 | nearest 上采样第 13 行 ×2 | 256 | 8 |
| 15 | concat(14,4) | 512 | 8 |
| 16 | 原生 C3k2(False) | 128 | 8 |
| 17 | Conv 3×3 stride 2 | 128 | 16 |
| 18 | concat(17,13) | 384 | 16 |
| 19 | 原生 C3k2(False) | 256 | 16 |
| 20 | Conv 3×3 stride 2 | 256 | 32 |
| 21 | concat(20,10) | 768 | 32 |
| 22 | 原生 C3k2(True) | 512 | 32 |
| 23 | Detect(16,19,22), nc=6 | 见下文 | 8/16/32 |

不能把 `model[11:]` 当作普通 Sequential。建议自定义 `_predict_once`，建立以原行号索引的局部缓存，把 adapters 注入 4/6/10，再按原生模块的 `m.f` 执行 11–23。模块保留原 i/f 元数据；`model.model[-1]` 必须仍是实际 Detect，不重复注册同一组模块造成 state_dict 键重复。

当前本地代码已核对到的契约：

- `Detect(nc=80, reg_max=16, end2end=False, ch=())`，`ch` 必须作为关键字使用，不能套旧签名。保留本地 parser 设置的非 legacy 分类分支。
- 本 YAML 无 one-to-one 分支；使用 `v8DetectionLoss`，不套 YOLO26 的 E2ELoss。
- train forward 返回 dict：`boxes: [B,64,A]`、`scores: [B,6,A]`、`feats: 三层特征`。scores 是 logits。不是三个 70 通道 tensor 的旧式列表。
- eval 非 export 返回 `(y,preds)`；y 为 `[B,10,A]`，前四维为默认像素 xywh，后六维为 sigmoid 分数。该 forward 不自动做 NMS。
- 本地 criterion 返回 `(三分量 loss 向量×batch_size, detached loss 字典)`；字典有 box/cls/dfl loss。独立测试做 `loss_vector.sum().backward()`；接入 native trainer 则保持其原生返回协议。
- criterion 从 `model.args` 读取超参，从 `model.model[-1]` 读取 Detect；在设备和 args 确定后懒初始化。不要持有过期 CPU criterion。
- Detect stride 必须为 `[8,16,32]`。原生模型初始化用 dummy forward 求 stride，随后才做 bias_init。复用已初始化原生 head 并校验实际尺度即可；不要再用一次昂贵的完整 Base forward 仅初始化 head。
- 本地 `bias_init()` 的 box bias 是 2.0，不是旧示例的 1.0。载入已有 head 权重后不得重新 bias_init。
- 处理 `.to()`/`.half()` 时的 Detect `stride/anchors/strides`；它们并非全部是普通 buffer。优先复用 `BaseModel._apply` 并做迁移后设备检查，本轮不做 FP16 验收。

参考文件仅限 `nn/tasks.py`、`nn/modules/head.py`、`utils/loss.py`、`models/yolo/detect/train.py`；需要核实 trainer/validator 调用时再定向读对应函数。

## 5. 官方实现隔离与 CPU 算子

### 5.1 主干的移植原则

检测版 import 依赖 mmdet/mmcv，当前环境没有。允许把该单一源码作为带来源说明的隔离副本放进新 package，只做必要适配：移除 registry/logger/checkpoint IO 依赖，改为显式 factory 和显式权重加载；将 NATTEN 导入改为后端接口。保留许可证/作者/来源说明，记录原文件 SHA256 和实际修改点。

不在 `sys.modules` 伪造 mmdet/mmcv/natten，不改变系统包或官方文件。timm/einops 使用已安装且实际可导入的版本。普通大核 depthwise conv 使用官方已经支持的 `use_gemm=False` 路径，即真实 nn.Conv2d；不要缩小 kernel、减少深度或跳过 overview/focus。

### 5.2 NATTEN 后端是本轮重点验收项

提供显式 `attention_backend: auto|natten|torch_reference`，报告解析后的实际后端。

- CPU `auto` 可选择真实可微的 PyTorch 参考实现；本机不得伪装为运行了 NATTEN。
- 请求 `natten` 而依赖不存在时明确失败。GPU 的 `auto` 在原生后端缺失时明确报错，不能悄悄回退后再报告 GPU 性能。
- 不强行安装 CUDA/NATTEN/MMCV，不把 `na2d_av` 替换成平均池化、零函数、普通卷积或 identity。

实现官方所需的 `na2d_av(attn,value,kernel_size)` 子集：dilation=1、非 causal，输入 attention `[B,heads,H,W,K²]` 和 value `[B,heads,H,W,D]`，输出 `[B,heads,H,W,D]`。其它参数/不合法尺寸 fail fast。

需要遵循旧版 NATTEN 的完整邻域平移边界：

```text
start_y = clamp(y - K//2, 0, H-K)
start_x = clamp(x - K//2, 0, W-K)
out[b,h,y,x,d] = Σ_(i,j) attn[b,h,y,x,i*K+j]
                            * value[b,h,start_y+i,start_x+j,d]
```

要求 H,W≥K，满足条件前的上采样由官方动态块处理。不能用 zero padding unfold 或逐邻点 clamp 代替完整窗口平移；两者边界值不同。实现时核对所采用 NATTEN API/窗口顺序；如果缺乏与 native backend 的数值对照，报告只可写“按窗口语义的 CPU 参考实现通过”，不能声称 GPU kernel parity 已验证。

用独立的朴素循环 oracle 检查角点/边缘/内部、H=K、矩形输入及 K=3/5；加有限值、对 attention/value 的梯度测试。保留官方 apply_rpb 的索引及两套 attention softmax，不将参考算子再次 softmax。避免构造全分辨率巨型 K² 张量，按窗口元素或行块累加，保留 autograd。

## 6. 预训练权重加载必须可审计

1. 使用已确认的本地 `.pth`，默认 `torch.load(..., map_location='cpu', weights_only=True)`。不要调用官方 `pretrained=True`：本地 factory 会将真值改为下载 URL，传路径也未必按本地加载。
2. 识别实际容器，只接受明确的 tensor state_dict 或已核实的 `state_dict/model/ema` tensor 映射；若多个候选含义不清，报告问题，不猜测。只按必要规则去 `module.` 等前缀，并检测重名冲突。
3. 将权重加载到 backbone 对象本身。共有 tensor 必须 key/shape 匹配；shape mismatch 直接失败。不要以宽泛 `strict=False` 掩盖漏载。
4. 分类权重中的 `head.*`、`aux_head.*` 可以显式排除；检测新增的 `extra_norm.*` 可以保留初始化。所有例外须逐 key 列表，不能再用泛化前缀扩大白名单。
5. backbone 其它共有可训练参数必须全部覆盖，尤其 `sub_blocks3/4`、`high_level_proj`、`patch_embedx`、`h_proj`；输出按模块统计的 numel 覆盖率，而不只 tensor 个数。若实际 checkpoint 与该假设不符，定位并如实修订报告，不改模型以强行匹配。
6. 记录 checkpoint SHA256、容器选择、前缀变换、matched/missing/unexpected/shape mismatch、允许例外以及总覆盖率。安全受限加载失败时保留错误，不自动切换 `weights_only=False` 或添加不明 globals。
7. 新 adapter 和新检测层的随机初始化使用固定 seed=0，记录 init 策略；载入后选若干代表参数验证确实与 checkpoint 相等。

V1 默认只要求现有 OverLoCK checkpoint。实现可选的 YOLO neck/head **纯 state_dict** 初始化入口，参数为空时明确标为 random；不要扫描其他目录找 YOLO 权重，不下载、不臆测类别映射。COCO 80 类到 SODA 6 类的分类输出层不做猜测性映射。

若之后要声称是严格的主干替换对照，须对齐可共享的 neck/head 初始化、数据和训练预算。当前随机 neck/head 与已有全模型检测预训练 baseline 不构成严格初始化对照。Resume 与初始化必须是互斥路径，不能恢复后又重载 backbone 覆盖训练状态。

## 7. SODA10M 数据接口与训练入口

### 7.1 数据事实与映射

已有 YAML 是自定义 COCO 格式，而不是可直接喂给原生 Ultralytics 的常规 labels 配置：

```yaml
format: coco
train: train
val: val
annotations:
  train: annotations/instance_train.json
  val: annotations/instance_val.json
nc: 6
names: [Pedestrian, Cyclist, Car, Truck, Tram, Tricycle]
category_id_map: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
```

原文件 names 实际为 0–5 的字典；两种写法语义一致。校验 annotations/categories 与该映射，不按出现顺序重新编号；`Tram` 不改名为 train，`Tricycle` 不并入 Cyclist。

### 7.2 建议采用隔离 YOLO 数据视图

为了复用原生增强、loss 和训练器，提供 COCO→YOLO 视图生成脚本：JSON 提供 image id/file_name，生成新目录下的 `images/{train,val}`、`labels/{train,val}`、常规 `data.yaml` 和 image-id 映射。图片用符号链接，不复制整个数据集；label/cache 均在视图目录中。路径以原 YAML 所在目录为基准，不依赖启动 cwd。

本轮只允许建立 `--limit-per-split 1` 或 2 的 smoke 视图，按稳定 image id 选样；不枚举图片目录、不生成全量数千文件视图。完整转换能力可实现并写明未来命令，但不在本轮执行。

- 使用 JSON 给出的尺寸核对实际读取样本尺寸；COCO 像素 xywh → YOLO 归一化 xywh，clip 后剔除退化框并统计。
- 校验 category、负面积、文件缺失、重复 ID、路径越界；未知类别或缺失图片不能静默跳过。
- 明确处理 `iscrowd/ignore`：常规 YOLO txt 无完整 ignore-region 语义。报告这两类实例数量和处理策略；若非零，训练语义差异须列为待对齐事项，不能声称与 COCO 完全等价。
- 空标签图保留空 label；建立样本路径与原 image id 的稳定映射，避免未来 COCO 评估图像 ID 错配。
- 不修改图片/JSON/YAML 原件，不向源目录生成 `.cache`；提供正向/逆向框变换的单元检查。
- smoke 使用原生数据变换或与之严格一致的 letterbox，检查正确的 bbox、RGB、归一化、类别与 batch 字段。

### 7.3 模型与原生 trainer 的衔接

独立 subclass 本地 `DetectionTrainer`，覆写 `get_model()` 返回混合 detector。不要仅把新 YAML 交给原生 parser，假设未注册主干会自动生效；也不要以只有 forward 的裸 nn.Module 冒充可训练模型。

模型至少支持 native 所需的 `forward(tensor)`、`forward(dict)`、`loss(batch,preds=None)`、`predict`、`stride`、`nc/names/args`、`model[-1]`、`set_head_attr` 及设备迁移。保留训练器设置的 names/args，criterion 的 device/args 生命周期正确。检查本地 trainer/validator 对 yaml/save/EMA 的实际使用，能复用 BaseModel 的地方优先复用。

batch 至少为 `img,batch_idx,cls,bboxes`；cls=0–5，bboxes 为相对于当前输入图的归一化 xywh，不能将原图像素框直接交给 loss。空标签 batch 也必须得到有限 loss。

配置显式列出 backbone、weights、backend、数据、seed、imgsz、优化器/学习率/weight decay、各参数组、BN 策略、冻结策略和 AMP；不要把实验 recipe 混进网络代码。所有 trainable 参数必须且只能进入一个优化器参数组，归一化/bias 的 decay 策略与所选 native 配方一致。

本轮默认不冻结整个 backbone；测试中为了节省资源冻结/切断特征梯度是独立 smoke 模式，不能泄漏为正式训练默认值。若实现 `backbone_lr_mult`，清晰分组且默认 1.0；0.1 等值是后续实验选项，不宣称已调优。未确认 baseline 的训练 recipe 时只提供显式标为模板的配置，不凭空称其为公平对照配方。

训练 CLI 的 `--help` 和配置/模型构造要实际检查；formal train 命令只写进 README，不在本机运行一个完整 epoch。代码支持训练入口不等于完成 DDP、resume、EMA final-eval 或 checkpoint reload 验收，逐项报告实际证据。不默认开启 compile、fusion、export 或 auto-batch。

## 8. 建议文件组织

在 `/Users/lw/Documents/CNN-Mamba/overlock_yolo11` 下建立以下结构；可合并小文件，但职责必须清晰：

```text
README.md
DESIGN.md                         # 本设计文档副本
overlock_yolo/
  __init__.py
  backbone.py                     # 官方检测实现的隔离适配，保留来源
  attention_backend.py            # lazy native + 可微 CPU reference
  checkpoint.py                   # 安全加载/匹配审计
  model.py                        # normalize + adapters + native tail route
  trainer.py                      # 原生 trainer 接口
  data.py                         # COCO 视图/映射
configs/
  overlock_b_yolo11s_soda.yaml
scripts/
  prepare_data.py
  smoke.py
  train.py
tests/
  test_attention_backend.py
  test_neck_contract.py
  test_data_and_loading.py
reports/
  environment.json
  source_manifest.json
  checkpoint_load.json
  validation.json
  HANDOFF.md
```

不要求按目录生成大量 boilerplate。使用 unittest 或当前已安装 pytest，无须安装测试框架。入口从指定 cwd 可运行；不要把只在某个 REPL 中设置的 sys.path 当成部署方式。

## 9. CPU 验收矩阵与资源预算

所有命令使用 `conda run -n 3dete python ...`；CPU 线程建议 2，batch=1，num_workers=0。先低成本测试，再完整 Base forward。给昂贵测试设置明确时间预算，建议单项 180 秒、完整 Base forward 两项合计不超过 6 分钟；超时保留失败/未完成状态，不以小模型代替 Base 后写全通过。

| 编号 | 必须执行的检查 | 成功条件 |
|---|---|---|
| V01 | 导入与来源 | torch/torchvision/timm/einops 导入正常；Ultralytics 来自指定源码；未强依赖 mmdet/mmcv |
| V02 | checkpoint 加载审计 | 指定 Base 真权重安全加载，共有主干参数无非白名单缺失；记录 SHA256 和覆盖率 |
| V03 | reference attention | 独立 oracle 的边界/内部/矩形/最小尺寸输出及梯度通过；非法输入明确报错 |
| V04 | native neck 路由等价 | 原生 YOLO11s eval 捕获 4/6/10 特征，再送新 tail；同权重同特征时输出/各层关键张量数值一致 |
| V05 | 完整 Base 方形 forward | checkpoint 实际载入，CPU eval + inference_mode，推荐 224×224，三层主干/adapter/head 的 shape、stride 和有限值正确 |
| V06 | 完整 Base 矩形 forward | 推荐 128×160，检查官方小特征处理和 neck 对齐；不能只测试正方形 |
| V07 | 原生 loss + backward | 真实 backbone 输出 detach 后，只训练 adapter/neck/head；非空框与空标签 loss 有限；非空用例各分支有有限梯度 |
| V08 | 真实数据烟测 | 1–2 张 SODA 样本经转换视图和检测预处理，类别/框/归一化符合约定，完成一次预测及 loss；不计算 AP |
| V09 | 配置与训练入口 | help/构造可运行；trainer 确实返回混合模型；trainable 参数组无漏项、重复项；smoke 冻结状态不污染训练默认 |
| V10 | 状态/产物核对 | 记录模块参数量、backend、dtype、实际输入、耗时、所有 skipped/failed；只改动新目录 |

V04 是发现错接 neck 的关键测试：不能只检查最终 shape。原生模型先初始化一次，保持相同 state 和 eval 状态，捕获真实的三层 backbone tensor，分别运行完整原生模型与抽出的 tail 比较；不要把随机且不同的权重拿来比较。

V07 的特征 detach 只证明后半部分链路。额外对 reference op/代表性动态块做 autograd 测试；若资源允许，可做一次缩小输入的完整 Base backward，但不列为本次简单本地验证的必达项。若没有完整 backward，报告明确写 **“完整 Base 端到端反向尚未验证”**；不得写成全模型训练已通过。

可增加一个 adapter/neck/head 优化器 step 并确认选定参数变化；不要为一个 smoke 创建完整 Base 的 Adam 状态。单纯 `py_compile`、随机初始化模型 forward、mock 特征或减少层数不能替代 V02/V05/V06。

V08 的真实样本 forward 可复用 V05，减少重复计算。完整 Base 初始化和权重加载尽量只做一次。不要本轮评估 5000 张图，不报告随机 head 的“mAP”。

各测试记录 `pass|fail|skipped`、原因、执行命令、exit code、实际耗时和紧凑的数值证据。若某项超时/缺依赖：先完成独立的其它检查，再在报告列出阻塞。修复后只重跑受影响测试。

## 10. 未来实验设计：本轮只准备，不执行

先完成 GPU 环境的 NATTEN/reference 数值对照、完整模型一批前向/反向、数据管线和保存/恢复，再安排正式实验。

建议分阶段：

1. R0：在同一版本、数据拆分、输入分辨率、评估器与预算下复现 YOLO11s，确认历史参照的适用性。
2. R1：本版 Base + 三个 adapters + 原生 neck/head，记录预训练来源和初始化差异。主干是否带来收益由实际 AP/候选质量验证。
3. R2：固定 R1 结构，对齐可共享的 neck/head 初始化；然后单独探索 backbone 学习率/BN/冻结策略。不能同时改数据增强、P2、损失和结构后归因于主干。
4. 若 R1 已提高中大目标、小目标仍缺几何候选，再做 P2 或高分辨率分支的单变量实验；若所有尺度都弱，先排除权重/归一化/梯度/收敛问题。
5. 若 Base 精度提高但成本不可接受，再有计划地引入较小 OverLoCK 版本及其匹配权重；本轮不得偷偷将 Base 改成小模型。

正式报告至少包括 COCO AP50、AP50:95、标准尺度分桶及注明定义的 AP50_S/M/L、每类 AP、预 NMS 几何候选覆盖、分类/排序分析，以及完整 detector 参数量、显存和同设备延迟。记录 maxDets、score/NMS 阈值、图像 ID 映射、resize/letterbox 与 inverse transform。

CPU reference 时间不能代替部署性能；只替换主干本身也不能直接声称论文方法创新。后续机制创新应由“哪里改善/哪里仍然失败”的可复现实验支撑。

## 11. 完成报告和派发纪律

`validation.json` 是验证结果的单一数据来源，`HANDOFF.md` 引用其结果，包含：

- 新增文件、接口与运行命令，源码来源/哈希及官方实现差异。
- checkpoint 加载报告、主干/adapter/neck/head 参数量，随机/预训练部分。
- 实际环境与 attention backend；每项验收的命令、结果和证据。
- 真实执行的输入尺寸/样本数，未执行项与原因，不将设计目标写成完成事实。
- 明确列出 GPU native parity、完整 backward、正式训练、完整 AP、DDP/resume/export 等哪些尚未验证。
- 后续在 GPU 机器上如何准备完整数据视图、训练和评估的命令模板；存在必要未知参数时标明，不能硬编码猜测的远程路径。

状态只使用有证据支撑的表述：

- 核心检查通过：`IMPLEMENTED / CPU_SMOKE_PASSED / GPU_AND_ACCURACY_UNVERIFIED`。
- 代码完成但某些核心检查未过：`IMPLEMENTED / CPU_SMOKE_PARTIAL`，逐项列失败与影响。
- 关键代码或权重仍有阻塞：`BLOCKED_OR_INCOMPLETE`，列出已完成和最小待解决项。

派发给 DeepSeek Flash 后，主代理本轮不轮询、不催问、不读取完成结果。用户通知完成后再另行验收；后台任务被接受并不代表实现已完成。执行者不要再委派其它 agent，不自行升级到 Pro，不启动周期性检查。

## 12. 派发记录

- 已于 2026-10-03 12:26（Asia/Shanghai）派发，任务 ID：`hub-15-murw34bt`。
- 执行模型：`deepseek-flash`；tier：`flash`；effort：`max`。
- 工作目录：`/Users/lw/Documents/CNN-Mamba`；写入范围仅为其中的 `overlock_yolo11`。
- 派发接口已接受任务；此记录不代表代码完成或验证通过。后续等待用户通知，再读取结果并独立验收。
