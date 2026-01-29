# 🧪 ChemDataMiner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/Chemistry-RDKit-green)](https://www.rdkit.org/)
[![Powered by](https://img.shields.io/badge/LLM-DeepSeek-blueviolet)](https://www.deepseek.com/)

**ChemDataMiner** 是一个模块化的 AI for Science (AI4S) 数据挖掘流水线。它致力于从非结构化的化学文献（PDF）中自动化提取结构化的分子数据（SMILES）及其物理化学性质（如 HOMO/LUMO 能级、光伏参数等）。

本项目结合了基于深度学习的文档解析工具（Mineru/Magic-PDF）、大语言模型（LLM）的语义理解能力，以及 RDKit 的化学信息学验证，构建了一个高精度的闭环提取系统。

---

##  核心功能 (Key Features)

-   **智能文档解析 (Step 0)**: 集成 `Mineru`，支持将复杂的双栏排版 PDF 精准转换为 Markdown，保留表格结构。
-   **表格清洗与提取 (Step 1)**: 自动识别文献中的数据表，剔除噪音行，修复表头漂移问题。
-   ** LLM 语义增强 (Step 2 & 3)**: 利用 LLM (DeepSeek/GPT) 结合上下文信息，解决化学缩写（如 "1a", "L1"）与具体分子结构的对应关系。
-   **化学专业验证 (Step 4)**: 
    -   基于 **RDKit** 的 SMILES 合法性校验。
    -   自动脱盐（Desalting）与标准化。
    -   物理量单位归一化（如 Hartree 转 eV）。
-   **容器化支持**: 提供轻量级 Docker 支持，解决 RDKit 系统级依赖问题。

---

## 🛠️ 处理流程 (Pipeline)

```mermaid
graph LR
    A[PDF 文献] -->|Step 0: Mineru| B(Markdown/Images)
    B -->|Step 1: Extract| C{表格数据}
    C -->|Step 2: Context Prep| D[LLM Prompt]
    D -->|Step 3: Inference| E[原始 JSONL]
    E -->|Step 4: RDKit Validation| F[最终 CSV 数据集]