# AgriPV v7：NSGA-II 与代表辐射采样（固定倾角、单轴跟踪、竖直双面）

此版本按 Dimitris 的建议保留 **NSGA-II 多目标优化**。这里的“加权”只用于代表小时的辐射重建：把代表小时结果乘以目标小时 GHI / 代表小时 GHI。LER 与年度净收益/ha 是两个独立目标，不做加权求和。

## 运行

1. 解压整个包，在包目录运行 `python -m pip install -r requirements.txt`。
2. 在原来能运行 bifacial-radiance/Radiance 的环境启动 Jupyter，打开对应支架的 notebook（`agripv_optimizer_fixed.ipynb` / `_single_axis` / `_vertical`）。
3. 默认 `RUN_SEARCH=False`，先运行单元格检查设置。
4. 首次真实试跑建议设 `POP_SIZE=4`、`N_GENERATIONS=1`、`RUN_SEARCH=True`。确认运行后再逐步增加种群和代数。
   **运行前先保存 notebook（Ctrl+S）**，并行进程会重新读取这个文件。

NSGA-II 分别对半透明和常规组件运行；同一设计的 Radiance/WOFOST 计算由磁盘缓存共享。结果保留各组件方案的非支配解，不会先按回收期或加权分数淘汰一种结构。种群/代数决定搜索预算，结果不能证明连续空间的全局最优。

## v7 改动：三种支架、三个 notebook（先读这一节）

| notebook | 支架 | 决策变量 | 结果文件夹 |
|---|---|---|---|
| `agripv_optimizer_fixed.ipynb` | 固定倾角（= v6.5.1） | tilt, pitch, azimuth, clearance_height | `results/nsga2_*`, `results/recheck_*` |
| `agripv_optimizer_single_axis.ipynb` | 水平单轴跟踪（Tianze `single_axis.ipynb`） | pitch, axis_azimuth, hub_height, limit_angle（backtrack 固定开启） | `results/single_axis_*` |
| `agripv_optimizer_vertical.ipynb` | 竖直双面（Tianze `vertical_pv.ipynb`） | pitch, azimuth（正面朝向）, clearance_height（下沿离地） | `results/vertical_*` |

- 三个 notebook 共用 `weighted_engine.py`、`bridge.py`、`sampling_core.py`、`eval_worker.py` 和 `analysis/`。支架类型由 Section C 的 `MOUNTING` 决定，不要在 notebook 里改它，要换支架就打开对应的 notebook。
- **固定倾角缓存仍然有效**：固定倾角 notebook 的物理签名与 v6.5.1 完全相同（已核对），v6.5.1 的缓存和结果可直接使用。单轴/竖直函数单独命名，只进入各自 notebook 的签名。
- **单轴物理**：每个采样小时用 pvlib（`set1axis`）算跟踪角并按该角度建场景，不做角度分箱；传感器随组件转动。
- **竖直物理**：静态场景（倾角 90°），组件传感器按 Tianze 的方法直接用 float64 坐标生成（库函数会舍入到 float16）。Tianze 原模型离地高度为 0；这里改为决策变量（0.3–1.5 m），因为固定田块的地面网格会与贴地组件相交。
- **LER 参考**：三种支架都用同一个固定式地面光伏作参考，LER 可直接比较。
- **约束与经济**（`EngineConfig`）：
  - 单轴：净空与通道按组件转到最大角度计算；支架按 hub_height 计价，另加 `tracker_extra_eur_kwp = 150`（D7.3：固定 ~650，单轴 ~800 €/kWp）。
  - 竖直：不检查 2.5 m 净空（农机在行间通过），通道 = pitch − 组件厚度；支架 `vertical_structure_eur_kwp = 50`（D7.3 低矮固定支架）；每行损失 `vertical_row_strip_m = 0.5` m 宽不能种植的带（假设，待确认）。
  - **默认成本改为文献值**（Dimitris 同意从文献补成本，并认可"净收益为负 = 寿命期内无法回本"作为结果）：支架 `structure_base_eur_kwp = 300`（3 m 以内）+ 每米 150 €/kWp（5 m 时 600，D7.3）；新增 `other_capex_eur_kwp = 650`（逆变器、BOS、安装、专业费用约 500，Bellone 2024 表 10；场地准备 150，D7.3）。固定高架系统总 CAPEX 1170–1470 €/kWp，在 D7.3 大田农光 1500 ± 425 范围内。
  - v6 的经济口径是支架 600 €/kWp、无其他 CAPEX；要复现，设 `EngineConfig(structure_base_eur_kwp=600., other_capex_eur_kwp=0.)`。经济参数不进入物理缓存键，改它们不需要重新做射线追踪；已有结果可用 `analysis/` 里的脚本直接按新口径重算。
- **只算部分组件**：Section C 的 `CONSTRUCTIONS = ("conventional",)` 跳过很慢的半透明追踪（约为常规的 5–8 倍时间），用独立缓存（`ConstructionEngine`）；两种都算时与 v6 相同。
- **采样**：只采少数正午附近小时会高估固定倾角、低估单轴（测试：单轴相对固定 35° 的增益在每天 1/2/4/8 小时时为 −15/−12/−1/+4 %，8 小时仍未完全收敛）。单轴和竖直 notebook 默认每天 8 小时（96 个天空），复核每月 2 天 × 每天 8 小时（192 个天空）。
- 结果表新增 `mounting`、`structure_height_m`、`min_clearance_m`；单轴逐时缓存另有 `tracker_theta_deg`。
- 离线测试：`python -m unittest -v test_weighted.py test_parallel.py test_tracking.py`（44 项）。

## v6.5.1 改动：地面网格加密

- small 预设 `ground_grid_y` 从 12 改为 48（间距 0.5 m），large 预设从 20 改为 120（同样 0.5 m 间距）。
- 原因：12 点（2 m 间距）与 4–5 m 的行距发生混叠，pitch 4.05 m 的作物 LER 被低估（0.730）。收敛测试：12 点 0.730、48 点 0.827、96 点 0.824（48→96 差 0.3%）。其他 pitch 变化约 0.4–0.8%。
- 光伏发电不受影响（LER_pv 不变）。
- 物理缓存键包含网格，因此 v6.5 及以前用 12 点网格生成的缓存与结果不能复用，需重跑搜索与复核；旧结果请备份为 `backup_grid12_*` 仅作对比。
- large 预设的 120 点未单独做收敛测试，运行时间更长。
- **田块大小与每排组件数是输入**：Section A2 顶部的 `FIELD INPUTS` 里设 `FIELD_PRESET`（`small`/`large`/`custom`）；选 `custom` 时填 `CUSTOM_N_MODS`（每排组件数）和 `CUSTOM_FIELD_DEPTH_M`（田块深度），地面网格自动按 0.5 m 间距生成。小田块只是为了省算力，模型本身接受任意田块和组件排布；行数由 pitch 推出（floor(深度/pitch)）。
- 版本字符串仍为 nsga2-v6.5，已用 48 点缓存的测试结果可直接复用。

## v6.5 改动：阶段 2 复核（Section F）与默认电价

1. **两阶段流程**：先用便宜的采样（如 `SamplingConfig("combined", 2, 1)`，24 个天空）做 NSGA-II 搜索；搜索完成后，只对 Pareto 设计用更密的采样（默认每月 3 天 × 每天 4 小时，144 个天空）重新评估。
2. **用法**：搜索跑完后，在 notebook 末尾 Section F 把 `RUN_RECHECK = True`（`RECHECK_SOURCE = None` 表示自动使用刚才的搜索结果；也可填 `pareto_front.csv` 的路径），运行该单元格。每个独特设计会对两种结构（常规/半透明）都计算，物理计算仍由并行 worker 完成，已完成的设计会缓存。
3. **输出**：`results\recheck_combined_144skies\` 下有 `evaluated_candidates.csv`、`pareto_front.csv`（复核后重新计算的 Pareto）、`comparison_with_search.csv`（LER、LER_crop、LER_pv、产量、发电量、净收益在搜索与复核之间的变化百分比）和 `run_metadata.json`。
4. **解读**：新的 Pareto 只在这些设计之间重新形成，不是新的搜索。若排序或 Pareto 成员明显变化，说明低采样不足以区分这些设计，应在论文中说明。
5. **默认电价**改为 2023–2025 三年平均；2020–2022 可作为敏感性分析（修改 `PRICE_YEARS`，电价不会触发重算射线追踪）。
6. **缓存键**：物理缓存键只包含影响 Radiance 的代码（采样计划、网格、行数推导、模拟调用），修改经济、导出或复核代码不会使已缓存的射线追踪失效。从 v6.4 升级到 v6.5 时缓存会失效一次。

## v6.4 改动：小场地、行数由间距推导、电价不再触发重算（先读这一节）

按 Dimitris 的建议，先在一块**固定的小场地**上证明优化器可用，同一套代码也能在大场地上运行（只是慢）。

**场地预设**（Section A2 的 `FIELD_PRESET`）

| 预设 | 场地 | 追踪 | 用途 |
|---|---|---|---|
| `"small"`（默认） | 宽 3 个组件，深 24 m，地面网格 4×12 | 只追踪 1 个组件（中间行的中间组件） | 证明优化器可用 |
| `"large"` | 宽 10 个组件，深 60 m，地面网格 10×20 | 中间行的内部组件 | 最终的大规模运行（很慢） |

- **行数由间距推导：** `n_rows = floor(深度 / 间距)`，至少 1。24 m 深的场地：间距 4 m → 6 行，5 m → 4 行，7 m → 3 行，12 m → 2 行。地块面积固定，所以 LER、每公顷收入和绝对收入都在同一块地上比较。参考设计（间距 3 m）也用同样规则（8 行）。
- **只算 1 个组件：** `n_mods=3` 时 `PV_TRACE_MODE="center"` 恰好是 1 个组件（中间行、中间组件）。它的发电量乘以组件总数。其他组件仍在场景里，所以仍然遮挡被追踪的组件。要追踪全部组件做验证，设 `PV_TRACE_MODE="all"`（很慢）。
- **搜索会检查：** 间距上界不能超过场地深度，否则某些设计一行都放不下，程序会报错。默认间距范围 4–12 m，深度 24 m，正好满足。
- **结果表新增列：** `n_rows`、`n_modules`、`field_area_ha`。
- **电价、成本、参考设计不再进入物理缓存签名。** 改它们只会重新计算便宜的作物和收入部分，不会重跑光线追踪。真正改变光线追踪的设置（场地、追踪模式、采样、代码）仍然会使缓存失效。
- **从 v6.3 升级时旧缓存作废一次**（签名版本变了），第一次运行会重新计算全部物理。

**已知的偏差（要在论文里写明）**

1. 只追踪中间组件再乘数量，忽略外侧行和行端效应。第一排前面没有遮挡，实际接收更多光，所以发电量偏低。行数只有 2–6 行时比大田里更明显。
2. 场地只有约 3.4 m 宽，地面辐照会受两侧漏光影响，作物辐射可能偏高。这点还没有量化。可以把 `FIELD_PRESETS["small"]["n_mods"]` 改大（比如 5）并比较作物产量。
3. 行数取整使目标随间距呈阶梯变化：间距 7 m 放 3 行占 21 m，其余 3 m 是没有光伏的空地，仍然计入面积。
4. 这些偏差在小场地上是已知的，目标是证明优化器可用，不是给出最终的绝对数值。

**第一次运行（建议）**

1. 先把上一次的结果备份：把 `results\nsga2_combined` 复制为 `nsga2_combined_5x5_center`。
2. 覆盖文件，保存 notebook（Ctrl+S），重启内核。
3. Section D 保持默认：`POP_SIZE=4`、`N_GENERATIONS=1`、24 个天空、`PARALLEL_DESIGNS=5`、`RUN_SEARCH=True`。
4. 跑完后把日志里的 `done ... in ... min` 和 `s/sky:` 行贴给我，我会用它们更新用时估算（目前的"小场地快 2–3 倍"只是估算）。

**验证小场地的偏差（跑完后，选 2–3 个设计）：** 把 `PV_TRACE_MODE` 改成 `"all"`，其余不变，另存结果为 `nsga2_combined_small_all`，再用 `design_id` 合并比较发电量，方法同 v6.3。

## 代表采样

`SAMPLING` 可设为：

```python
SamplingConfig(mode="representative_hours", hours_per_day=4)
SamplingConfig(mode="representative_days", days_per_month=3)
SamplingConfig(mode="combined", days_per_month=3, hours_per_day=4)
SamplingConfig(mode="full")
```

代表小时模式每天将有光照小时分段，每段选一个实际小时进行 Radiance 追踪；其结果按 GHI 比例映射回全年目标小时。它精确保持水平面 GHI 总量，但不能保证遮阴后地面辐射或倾斜面 PV 发电精确守恒。代表日与组合模式也会近似天气及阴影变化。最终候选需用 `full` 逐时模式复核，并比较误差表。

## 电价

包内 `prices/` 已包含项目来源上传的 2020–2025 年 DE/LU 电价表。默认 `PRICE_MODE="csv"`，将 `PRICE_YEARS`（默认 2023–2025，排除 2020–2022 的危机年份）每个小时的价格按月、日、固定标准时小时映射后取平均，构成典型逐时电价；2020 和 2024 闰年的 2 月 29 日会在年度对齐时排除。输入要求 `hour` 从 0 到 8759/8783 连续排序，`price_eur_per_kwh` 为 EUR/kWh。负电价保留。若想运行单一年份情景，可更改 `PRICE_YEARS`。

价格文件的小时序列按 UTC 解释，并转换到 EPW 固定标准时区（不使用夏令时）。如果原始价格数据的 `hour` 并非 UTC 顺序，应先在来源端确认并转换，不能仅凭列名推断时区。

## 模型范围

- 光伏按完整 365 天逐时评价，冬小麦按实际一季计算，不按季节天数外推作物收入。
- 光照、作物与经济指标使用同一个随方案方位旋转的代表地块；只搜索倾角、行距、方位和高度，组件电池间距固定。
- 半透明与常规结构分别优化、共同保留。Pareto 目标为最大化 LER 和年度净收益/ha。
- 未成熟、保产率不足、净空不足或通道过窄的候选不可行；66% 保产率为研究假设，不代表法规认证。
- 年度净收益扣除一季农业成本、PV 运维与年化 CAPEX；成本、效率、作物参数等仍需用可靠数据替换或做敏感性分析。
- 作物采用 WOFOST 潜在生产模式，尚无 Aachen 田间校准；Radiance 地面直散射只以总短波辐射传给 crop bridge，水分胁迫和遮阴微气候未建模。

## 输出与验证

结果写到 `results/nsga2_<sampling_mode>/`：

- `evaluated_candidates.csv`：搜索过程中评价的可行性与目标值。
- `pareto_front.csv`：所有已评价候选中的非支配可行方案。
- `run_metadata.json`：采样、电价、天气和模型假设。
- `validation_errors.csv` 与 `full_hour_pointwise_finalists.csv`：开启最终验证后生成。

缓存位于 `results/cache_v5/`；更改价格、物理设置或源码会进入不同缓存键。先用小种群试跑，最终应增加种群、代数和随机种子检查稳定性。

离线检查：`python -m unittest -v test_weighted.py`。这些测试检查采样映射、上传电价完整性、土地网格、经济口径、缓存和 notebook 语法；不等于真实 Radiance + WOFOST 联合验证。当前执行环境没有 pymoo/Radiance，因此 NSGA-II 与真实物理运行需在你的原环境安装依赖后验证。


## 并行评价（v6.1 新增）

NSGA-II 每一代的所有设计现在可以同时算：`PARALLEL_DESIGNS=4` 表示最多 4 个设计各自在独立的 Python 进程里跑 Radiance
（`eval_worker.py`），结果只通过磁盘缓存回到主 notebook，主进程随后只做很快的 WOFOST 部分。
`PARALLEL_DESIGNS=1` 就是原来的串行方式。

- 参考设计的物理模拟会作为第一个任务一起并行跑，之后有缓存。
- 每个设计内部仍用多线程跑组件，`WORKERS_PER_DESIGN=None` 时按 逻辑 CPU 数 // PARALLEL_DESIGNS 取值
  （20 线程、并行 4 个 → 每个 5 线程）。线程数不进入缓存键，改它不会让缓存失效。
- 每个任务的日志在 `results/cache_v5/jobs/*.log`，卡住或报错先看这里。
- 某个任务失败时，已完成的任务保留在缓存里，修好后重新运行会自动续算。
- 工作进程会重新执行 notebook 里 Section D 之前的所有单元格，并检查缓存签名与主进程一致；
  不一致（通常是改了模型单元格但没保存 notebook）会报 `exit code 3`，不会写入错误的缓存。
- 缓存键里的函数哈希改成只看字节码、不含文件名和行号（旧版 `marshal` 会把文件名算进去，
  主 notebook 和工作进程会得到不同的键）。**所以 v6.1 的缓存和 v6.0 不通用。**

**提速倍数没有实测。** 代表性的检查方法：用 `combined`（每月 2 天 × 每天 3 小时）、`POP_SIZE=4`、`N_GENERATIONS=1`
各跑一次 `PARALLEL_DESIGNS=1` 和 `PARALLEL_DESIGNS=4`，比较总用时。如果任务管理器里总 CPU 已经接近 100%，
继续加并行数没有用；如果还有空闲，才值得再加。


## v6.2 改动

- **时间基准改为 `pvgis_utc`**（Section A2）。用太阳几何核对过：EPW 的行是 UTC。按 UTC 解释时 GHI 与
  DNI×cosz+DHI 平均偏差 0.4 W/m²、没有夜间直射；按本地时间解释偏差 22 W/m²，有 100 小时夜间直射。
- **约束用连续违反量**。每个结果多一列 `constraint_violation`：可行时为 0，否则是
  保产率、净空、通道宽度、是否成熟各项相对不足量之和。NSGA-II 现在知道往哪个方向修正，不再把所有不可行个体看成一样。
  `feasible` 的含义不变。
- **搜索阶段不再追踪"仅直射"**（`EngineConfig.trace_ground_components=False`）。作物模型只用总短波辐射，
  省掉每张天空多出来的一次地面追踪；直射/散射列在这个模式下不再输出。同时避免了 bridge 因
  "总辐射减直射为负"而在几小时计算之后报错的风险。需要这两列时设为 True。
- **地面网格 15×15 → 10×10**（Section A2）。作物模拟用面积平均，点数多只增加光线数。最终逐点验证用同一个网格。
- **逐段计时**。每 20 张天空，日志里会多一行 `s/sky: sky … octree … pv … ground …`，
  分别是天空文件、场景八叉树、组件追踪、地面追踪每张天空的平均秒数。
  这几个数字决定下一步该优化什么，请在冒烟测试跑起来后把日志里这一行发回来。
- **点击中断会一并结束工作进程及其 Radiance 子进程**，不用再手动杀。
- 缓存键版本升为 `nsga2-v6.2`，之前的缓存不通用。


## v6.3 改动：只追踪中间组件（可选）

冒烟测试的计时显示：常规构型约 93–95% 的时间花在组件追踪上；半透明构型总共约为常规的 5 倍（每个设计 175–191 分钟里，
常规 21–32 分钟，其余是半透明）。所以在 Section A2 新增了一个开关：

```python
PV_TRACE_MODE = "all"      # 默认：每张天空追踪全部 25 块组件（原来的做法）
PV_TRACE_MODE = "center"   # 只追踪中间一行的 3 块内部组件，再把平均每块发电量乘以 25
```

- "center" 的组件追踪量约为原来的 1/8（5×5 阵列），但**忽略边缘组件的边缘效应**，而且地面作物区域仍是整个占地的面积平均。
  这个口径需要和 Dimitris 确认；**用于结果之前必须先做下面的对比。**
- 开关会写进缓存键，"all" 和 "center" 的缓存不会混用。**本次升级改了追踪函数，v6.2 的缓存也不再通用。**

### 对比 "all" 和 "center"（用你已有的冒烟结果，不用重跑 "all"）

1. 在运行新代码**之前**，把上次的结果文件夹复制一份：
   `results\nsga2_combined` → `results\nsga2_combined_all_baseline`（里面有 4 个设计在 "all" 模式下的 `electrical_kwh`）。
2. 在 A2 把 `PV_TRACE_MODE` 改成 `"center"`，保存 notebook，重启内核，**其他设置保持不变**
   （同样 `SAMPLING`、`POP_SIZE=4`、`N_GENERATIONS=1`、`SEED=1`，这样是同样的 4 个设计），运行。
3. 比较：

```python
import pandas as pd
base = pd.read_csv(r"C:\Users\haoli\Agripv\optimizer\results\nsga2_combined_all_baseline\evaluated_candidates.csv")
new  = pd.read_csv(r"C:\Users\haoli\Agripv\optimizer\results\nsga2_combined\evaluated_candidates.csv")
m = base.merge(new, on=["design_id", "pv_type"], suffixes=("_all", "_center"))
m["kwh_error_%"] = 100 * (m.electrical_kwh_center / m.electrical_kwh_all - 1)
print(m[["design_id", "pv_type", "electrical_kwh_all", "electrical_kwh_center", "kwh_error_%"]])
```

误差在几个百分点以内、且各设计的排序不变，才说明 "center" 可以用于搜索。

### 其他
- Section E 的输出路径提示改为 `results/nsga2_<sampling_mode>/`。
- 如果想直接看某个任务日志里的计时行，用记事本打开整个日志最省事：`notepad ...\results\cache_v5\jobs\design_xxxxxxxxxxxx.log`，
  半透明的计时行在常规之前。
