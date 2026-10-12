# analysis/ — 搜索完成后的后处理

这些脚本只读取 `results/<run>/` 里的 CSV，不调用 Radiance/WOFOST，不碰缓存，搜索运行时也可以跑。
输出都写到 `results/<run>/analysis/`。在 optimizer 目录下运行：

```
python analysis/figures.py nsga2_combined        # Pareto 前沿、LER 分量、Spearman 热图、是否贴边界
python analysis/sensitivity.py nsga2_combined    # 成本/电价单因素敏感性、Pareto 稳定性、组件盈亏平衡价
python analysis/convergence.py nsga2_combined    # 每代超体积；先保存 notebook（Ctrl+S）
```

- `sensitivity.py` 先检查重算的经济结果与引擎一致（不一致会报错）。参数范围在文件顶部的 `RANGES`，论文里要给出处。
- `convergence.py` 从 notebook 输出里的 `NSGA-II ... design:` 行和 pymoo 每代表格还原代数。
  比较多个种子时，各次运行要用同一个参考点（当前按各自运行的范围归一化，只适合看单次运行是否走平）。
- 换电价年份需要逐时发电量：搜索完成后把 `price_years_cell.py` 的内容粘到 notebook 末尾的新单元格运行（射线追踪走缓存）。
- 已用旧的 12 点网格结果 `main_search_24skies` 测试过；那批结果的数值作废，只用来检查脚本。
