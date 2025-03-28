<div align=center><img src="doc/gead_logo.png" width="80%"></div>

> *规则精炼谜题：安全应用中基于深度学习的异常检测的全局解释。* 已被 [CCS'24](https://www.sigsac.org/ccs/CCS2024) 接收。

![](https://img.shields.io/badge/license-MIT-green.svg)
![](https://img.shields.io/badge/language-python-blue.svg)
![](https://img.shields.io/badge/framework-pytorch-red.svg)

[English (CCS'24)](./README.md)

## 简介
本项目是CCS'24提出的GEAD方法的实现和实验结果。
简而言之，GEAD是一种用于提取深度学习异常检测模型中规则的方法。如下图所示，它包含几个核心步骤：
![GEAD概览](./doc/gead_overview.png)
- **根回归树生成**：利用黑盒知识蒸馏方法提取原始规则树
- **低置信度区域识别**：找出导致原始模型与基于树的可解释模型之间不一致的区域
- **低置信度增强**：增强可能导致两个模型之间决策不一致的数据
- **低置信度规则生成**：使用增强数据扩展原始规则
- **树合并和离散化**：简化规则以提高操作人员的可读性
- **规则生成（可选）**：将规则树转换为可读的规则集
