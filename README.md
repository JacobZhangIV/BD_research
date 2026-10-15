# BD_research

手写 PyTorch 教学练习：从张量与 attention，到字符级 toy Block Diffusion 的训练、生成和内部表示观察。

## 文件

- `a_toy_bd/a1_tensor_basic.py`、`a1_tensor_basic.ipynb`：张量与 attention 基础练习。
- `a_toy_bd/a2_toy_bd.ipynb`：字符级 toy BD 的数据、模型、训练和生成。
- `a_toy_bd/a3_probe_toy.ipynb`：多头、多层模型及 attention、表示相似度、连接干预练习。

## 使用

需要 Python、PyTorch、Matplotlib 和支持 Notebook 的编辑器。按 Notebook 单元顺序运行；数据读取位置以代码为准。Shakespeare 数据文件需要单独准备，未包含在仓库中。

这是学习中的手写教学实现，从零训练，不是已有 AR→BD 模型的精确复现，也不构成模型退化的证明。

上传的 Notebook 不包含运行输出。本地代码和注释保持原样；训练权重、日志、个人计划和合作记录不上传。
