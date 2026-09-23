# 附录：CAREER 核心方法与实现接口

## A. 依据与范围

本附录依据用户提供的 38 页 `ICLR27_CAREER.pdf`：*When Is Seeing Enough? Multi-Granularity Visual Search with Verification Feedback for High-Resolution Understanding*。文件 SHA-256 为 `abd3baa9dbb962cc0bd91fd58f950624726dfa20688bc64203e69968a6816425`。下文页码对应此 PDF，核心依据为 §§3.6–3.8（第5–10页）、附录 B.3（第25–26页）和 D（第31–33页）。

范围为一次 CAREER 搜索的模型输入、候选观察、证据验证、控制与回退。独立 verifier 是推理方法的组成部分，不读取参考答案或正确性标签。论文未公布全部运行参数，G 节明确区分论文常数和运行配置；本附录不声称实验性能或全部数值设置已复现。

## B. 输入、全图门与累计证据

输入高分辨率图像 $I$、问题 $q$、至少两个选项 $O=\{o_i\}_{i=1}^M$。生成器 $\mathcal M_g$ 与独立验证器 $\mathcal M_v$ 均冻结。控制器给选项、需求、候选和视图分配稳定标识；模型使用经 tokenizer 检查的单 token 答案代码。

初始 $B_0=\{I\}$。生成器在同一次回答推理中产生保存的全图答案 $y_{\mathrm{global}}$ 及选项 token logits $z_i$，在选项代码之间作 softmax 得到 $p_i$。令 $(1),(2)$ 为概率首选与次选，论文式(1)为

$$
\Gamma=[p_{(1)}\ge\tau_{\mathrm{conf}}]\land[p_{(1)}-p_{(2)}\ge\tau_{\mathrm{gap}}].
$$

通过时返回最高概率选项，标记为全图门接受。全图单独输入时不调用独立 verifier。若门未通过且不能新增局部视图，立即返回保存的全图答案，标记为未经验证的回退；此时不运行问题规划或候选构建（第5、20页）。

获得 $t$ 个有效局部视图后，

$$
B_t=\operatorname{Pack}\bigl(I,\operatorname{Dedup}\{(e_j,m_j)\}_{j=1}^{t}\bigr).
$$

$m_j$ 包括原图坐标、来源、尺度、动作、定位短语、SAM 输入范围与定位结果。坐标使用原图像素的 `[x,y,width,height]`。两模型均收到原生多图序列：全图在前，全部保留的局部视图按取得顺序排列，并附相应标识与元数据。换分支和恢复保留此前证据。

局部图从原分辨率图像裁出，使用接收模型的实际预处理。重复观察不增加视图。加入前检查两个模型的输入容量，不能为容纳新视图静默移除旧证据。原图缓存只是裁剪源，未实际送入模型的细节不能被当成已观察事实。

## C. 问题规划、候选与排序

全图门未通过且仍可观察时，生成器仅根据问题文本生成 `localization_phrases` 和编号 `requirements`。此调用无图像、选项或参考答案；不能猜测待求属性、关系或存在性。没有可用实体时允许空定位短语。

SAM 3 编码原图并缓存密集特征 $H_e$。每个短语独立定位，将该短语的全部有效检测框合并成最小外接矩形。不同短语形成不同初始候选；排序最高的可行候选由控制器取得一次 screening 视图并立即生成、验证。该视图计入局部观察预算。若未接受且仍可取得下一视图，才基于缓存特征构建 SGAP 树；无可行 SAM 候选时直接进入建树。

SGAP 以 SLIC 超像素和空间相邻的视觉特征聚类构建固定层级。SGAP 区域与未访问 SAM 候选形成联合池；完全相同几何合并来源和访问记录。仅 SGAP 节点保留原有父子边，SAM 区域的包含或重叠不产生新边。

令 $p_0=q$，$p_1,\ldots,p_L$ 为全部定位短语。分别单位化 CLIP 文本向量后求平均，再次单位化得到 $\hat t_q$；crop 的单位向量为 $\hat v_c$：

$$
S_q(c)=\hat v_c^\top\hat t_q,\qquad
\hat t_q=\operatorname{Normalize}\left(\frac1{L+1}\sum_{l=0}^{L}\operatorname{Normalize}(f_T(p_l))\right).
$$

令 $R_c$ 为特征感受野中心位于候选框内的位置，$\bar h_c$ 为其中特征均值：

$$
D_f(c)=\max\left(0,1-\frac1{|R_c|}\sum_{i\in R_c}\frac{h_i^\top\bar h_c}{\|h_i\|_2\|\bar h_c\|_2+\epsilon}\right).
$$

没有对应特征位置的候选不可行。边缘分量 $D_e$ 按 B.1–B.3（第25–26页）计算：将 crop 转为 $[0,1]$ 灰度图，保长宽比缩放并填充到固定 $H_s\times W_s$；有效像素 mask 用 $3\times3$ 结构元素腐蚀一次；求标准 Sobel 两方向梯度的模长在腐蚀后 mask 上的均值。mask 为空则拒绝。

对当前排序全集分别做 min–max 归一化，常量分量归零：

$$
\widetilde X(c)=\frac{X(c)-\min X}{\max X-\min X},\quad
R(c)=0.70\widetilde S_q(c)+0.30[0.50\widetilde D_f(c)+0.50\widetilde D_e(c)].
$$

screening 在 SAM 候选内归一化；构建联合池后重新归一化并固定排序，平分按稳定候选标识决定顺序。SAM 定位分数用于检测有效性与记录。正式搜索从最高优先级的可行未访问候选初始化。

## D. 每轮生成、独立验证与接受

每次新增有效局部图后，先生成临时答案，再对每个选项独立验证同一个累计 $B_t$。验证器输入仅含问题、需求、该选项以及带元数据的 bundle；不包含生成器答案或其他选项。

每个选项对应一次 completion。首 token 被限制在单 token 代码 `A/B/C`，分别表示 `Support/Refute/Insufficient`。捕获解释生成前的三 logits，按 $T=1$ 计算：

$$
P_{t,i}(r)=\frac{\exp z^r_{t,i}}{\sum_{r'\in\{A,B,C\}}\exp z^{r'}_{t,i}},\qquad
s_{t,i}=P_{t,i}(A),\qquad q_{t,i}=\frac{s_{t,i}}{\sum_j s_{t,j}}.
$$

直接标签为三代码概率的最大者。首 token 后在同一次 completion 中继续输出 JSON 语义反馈；后续解释不修改分数，也不另行调用一次验证器。每项最多两条 grounded facts 和两条 missing requirements，并引用已有视图与需求标识。

所有选项 logits 有限有效时，令 $k,l$ 为支持首选和次选：

$$
\operatorname{Accept}_t=[s_{t,k}\ge0.65]\land[q_{t,k}-q_{t,l}\ge0.15].
$$

通过即返回 $o_k$，标记为已验证，并跳过导航。screening 与后续观察使用同一门，首个有效局部图也可接受。分数不是经证明校准的正确率。

存在性判断遵循 verifier 提示：可见匹配实例支持存在；局部未发现对象不能推出不存在；支持 `No` 或反驳 `Yes` 需要可见场景提供充分覆盖和可辨识性。此要求通过逐选项证据判断实现。

## E. 导航、动作与恢复

只有验证未接受且可以继续观察时才请求导航。生成器接收同一 $B_t$ 的最新反馈、编号需求、当前焦点、剩余预算、失败记录及控制器给出的合法动作–候选对，选择一个需要消歧的证据需求与一个合法操作。

| 动作 | 语义 |
| --- | --- |
| `ZOOM` | 在执行时当前框的原分辨率 crop 上定位，合并同短语全部有效框并映射到原图；新框须严格位于当前框内；保留候选标识。 |
| `EXPAND` | 在原图上定位，以当前框与合并目标框的最小外接矩形作为新框；须严格扩大；保留候选标识。 |
| `SPLIT` | 访问当前 SGAP 节点最高优先级的可行未访问直接子节点，使用其既有框。 |
| `NEXT` | 访问联合全局队列最高优先级的可行未访问候选，使用其既有框。 |
| `RECOVER` | 仅控制器执行：从历史向后恢复最近仍可取得新观察的焦点，再交给生成器选择合法操作。 |

`ZOOM/EXPAND` 需要非空 `sam_prompt`；`SPLIT/NEXT` 使用 `null`。定位结果缓存按输入图像与短语复用；调整不改变候选树或固定优先级。SAM-only 候选没有 `SPLIT`。

失败定位、重复 crop 或不合法几何不增加观察；对应动作–候选对在该焦点及框状态永久排除，恢复后仍有效。随后只在更新合法动作下重新导航，使用不变的证据与反馈。每次恢复必须导向未尝试观察或终止。

同一观察方向且领先选项不变时，令 $\mathrm{gap}_t=q_{t,k}-q_{t,l}$：

$$
\Delta s_t=s_{t,k}-s_{t-1,k},\quad
\Delta m_t=\mathrm{gap}_t-\mathrm{gap}_{t-1},\quad
\mathrm{Stall}_t=[\Delta s_t\le0.01]\land[\Delta m_t\le0.01].
$$

任一指标明显改善重置停滞计数，连续两次停滞或无合法观察触发恢复。领先选项改变、`NEXT`、`RECOVER` 和阶段切换建立新比较基线。先判接受，再判恢复。若历史状态均不可扩展但全局队列仍有候选，控制器将 `NEXT` 作为合法选择交给生成器。

## F. 预算与错误处理

默认最多取得 $K=8$ 个有效局部视图，含初始 SAM screening；全图不计，累计最多 $K+1$ 图。定位失败、重复观察或焦点恢复不扣局部视图数。所有规划、生成、导航、逐选项验证、解释 token、SAM、建树及重复处理累计图的计算仍须记录；逻辑调用与批处理 forward 分开计数。

视图数、两个模型任一输入容量或可行观察耗尽时，返回最初 $y_{\mathrm{global}}$，标记未经验证。首 token logits 缺失或非有限、规划/导航输出非法等不可恢复输出错误立即产生带原因的回退，不隐式免费重试。验证解释 JSON 无效时丢弃对应语义反馈，保留有效首 token 分数。有限视图预算之外，失败动作永久排除保证不产生无限零观察重试。

## G. 论文参数与运行配置

| 表 S1 固定参数（第25页） | 值 |
| --- | --- |
| 查询、视觉排序权重 $w_s,w_v$ | $0.70,0.30$ |
| 视觉分数中特征份额 $\lambda$ | $0.50$ |
| 绝对支持门 $\tau_{\mathrm{abs}}$ | $0.65$ |
| 归一化选项间隔门 $\tau_{\mathrm{margin}}$ | $0.15$ |
| 支持、间隔最小进展 $\delta_s,\delta_m$ | $0.01,0.01$ |
| 停滞耐心 $P$ | $2$ |
| 最大有效局部观察 $K$ | $8$ |

论文默认 verifier 为 `Qwen3-VL-4B-Instruct`；生成器为 `InternVL2.5-8B`、`LLaVA-OV-7B` 或 `Qwen2.5-VL-7B`（B.3，第25页）。权重路径和运行环境由调用方提供。

论文没有给出以下运行数值。[config.py](../career/config.py) 要求 JSON 恰好包含下列顶层字段，不能用表 S1 或旧稿值推断：

| 必填字段 | 用途与约束 |
| --- | --- |
| `global_confidence`, `global_margin` | $\tau_{\mathrm{conf}},\tau_{\mathrm{gap}}\in(0,1]$。 |
| `planning_tokens`, `navigation_tokens`, `verifier_tokens` | 各 completion 的正整数输出上限。 |
| `generator_context_tokens`, `verifier_context_tokens` | 两个模型的正整数完整上下文容量。 |
| `edge_size` | 固定边缘尺寸 `[height,width]`，各至少为3。 |
| `edge_interpolation` | `bilinear`、`bicubic`、`lanczos` 之一。 |
| `sgap` | 下述完整参数对象。 |

`sgap` 的必填键为 `n_atoms`, `pos_weight`, `split_threshold`, `keep_threshold`, `use_local_normalization`, `use_silhouette_score`, `max_depth`, `min_splits`, `max_splits`, `min_region_size`, `max_nodes`。树深、区域大小和节点上限在正式搜索前固定。实现将 `min_region_size` 解释为原图像素的最小短边，`max_nodes` 包含全图根节点；这些是论文未进一步细分的工程约定。字段类型和数值合法性由 `RuntimeConfig.load` 检查。

边缘图采用居中零填充是明确的工程约定；论文只规定一致的缩放和填充规则。模型实际预处理、权重版本、CLIP 版本、数值精度及运行容量同样需要随运行记录。论文给定的方法流程与固定参数不能代替这些未披露细节。

## H. 固定提示与接口

完整固定英文提示位于 [prompts.py](../career/prompts.py) 的 `PLAN`、`ANSWER`、`NAVIGATION`、`VERIFIER`，依据论文 D.1–D.3。动态输入由对应函数序列化为独立 JSON，图像以有序多图输入传入模型。

**问题规划 `plan_prompt`：**只序列化问题。固定指令要求使用问题明确提到的对象、部件或属性，不回答问题；为关系补齐目标、参照和对应要求，为存在性补齐可见实例或场景覆盖要求。输出结构：

```json
{"localization_phrases": ["luggage"], "requirements": [{"id": "r1", "description": "Identify the queried luggage."}, {"id": "r2", "description": "Observe its color."}]}
```

**答案生成 `answer_prompt` 的固定提示：**

```text
Answer the question using the supplied visual evidence. Return exactly one supplied option identifier. No reference answer is available. Do not treat option wording, likely object properties, or absent observations as visual evidence. If the evidence is incomplete, still choose your best provisional option; the controller determines its status.
```

输入含问题、全部选项、视图元数据，以及局部轮的 requirements；输出恰好一个允许的选项 token。

**导航 `navigation_prompt`：**在反馈可用后调用，要求选择决策所需的未解决需求，引用实际反馈和需求标识；不得把解释当作已观察事实，不得猜待求属性，不能自造框或候选。返回字段固定为：

```json
{"requirement_id": "r2", "feedback_option_ids": ["B"], "evidence_gap": "<fact to establish>", "action": "ZOOM", "candidate_id": "c7", "sam_prompt": "<localization phrase>"}
```

所有标识来自真实输入。无可用语义反馈时 `feedback_option_ids` 可为空；终止状态动作字段为 JSON `null`，控制器不再请求下一次导航。

**验证 `verification_prompt`：**完整固定提示包含独立评估、缺失证据不等于反驳、原图坐标关系判断和存在性覆盖要求。三代码含义为：

```text
A (SUPPORT): the required identity, when applicable, and queried attribute, relation, or existence state are visually established and support this option.
B (REFUTE): established visible evidence directly contradicts this option.
C (INSUFFICIENT): required identity, detail, correspondence, or context is absent, unreadable, occluded, ambiguous, or unresolved.
```

首代码后同一次生成接续以下格式；示例不预设真实标签或事实：

```text
C
{"grounded": [{"requirement_id": "r1", "view_ids": ["v1"], "fact": "<visible fact>"}], "missing": [{"requirement_id": "r2", "needed_evidence": "<unresolved fact or context>"}]}
```

记录首 token 决策代码、支持分数、JSON 有效性、生成 token 数和实际引用标识。事实最多两条、缺失需求最多两条，允许空数组，不能为决定性证据编造缺口。

## I. 算法与代码位置

```text
B ← {全图}; t ← 0
y_global, p ← 全图生成器答案与同次答案 logits
若 Γ(p) 通过：返回最高概率答案，标记全图门接受
若 K=0 或新图不能容纳：回退 y_global
Q ← 仅问题文本的定位短语与证据需求
SAM 编码、逐短语合并框、排序；尝试一次最高优先级 screening
每次新增有效图：t+=1；累计 B；生成答案；逐选项验证；先判接受
若仍可观察：构建 SGAP 与联合固定排序；初始化正式搜索
while 还有预算、容量和可行观察：
    取得新视图；累计 B；生成答案；逐选项同 completion 评分及反馈
    若双门通过：立即返回验证首选
    若可以继续：必要时恢复焦点，按最新反馈请求合法导航
    失败动作永久排除，只重新导航，不重做当前答案/验证
返回最初 y_global，标记未经验证；输出错误附失败原因
```

| 模块 | 职责 |
| --- | --- |
| `career/search.py` | 顺序控制、全图与局部门、观察计数、恢复和回退。 |
| `career/types.py` | 原图坐标、候选、视图、定位与 completion 记录。 |
| `career/config.py` | 表 S1 参数与必填运行配置。 |
| `career/prompts.py` | 四个固定提示与各自输入隔离。 |
| `career/frontend.py` | SAM、SGAP、CLIP、固定候选排序。 |
| `career/models.py` | 原生多图输入、首 token 评分与同次续写、容量检查。 |
| `career/__main__.py` | 单图 CLI、模型路径与运行配置输入。 |
