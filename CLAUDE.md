# LLM功率控制实验系统
## 项目愿景
本项目是本科毕设《基于动态功率调节的大语言模型推理能耗控制》的实验平台，旨在通过量化研究大语言模型推理过程中功率限制对性能、能效的影响，设计并验证动态功率调节策略，实现推理性能与能耗的最优平衡。

## 架构概述
采用轻量脚本式架构，总代码量&lt;500行，划分为6个核心功能模块：功率控制、推理封装、负载生成、实时监测、实验主流程、结果分析。支持固定功率下多并发度实验、动态功率调节实验、结果自动分析与可视化全流程。

## 模块结构图
```mermaid
graph TD
    A["(Root) LLM功率控制实验系统"] --&gt; B["docs/ 文档目录"]
    A --&gt; C["models/ 模型文件目录"]
    A --&gt; D["results/ 实验结果目录"]
    A --&gt; E["error/ 错误日志目录"]
    A --&gt; F["summary/ 总结文档目录"]
    A --&gt; G["核心代码模块"]

    click B "./docs/CLAUDE.md" "查看文档目录"
    click C "./models/CLAUDE.md" "查看模型文件目录"
    click F "./summary/CLAUDE.md" "查看总结文档目录"
```

## 模块索引
| 模块路径 | 语言 | 核心职责 | 入口文件 |
|---------|------|----------|----------|
| 根目录 | Python | 项目核心实现 | run_experiment.py |
| docs/ | Markdown | 项目文档 | 无 |
| models/ | - | 模型文件存储 | 无 |
| summary/ | Markdown | 总结文档 | 无 |

## 核心代码模块说明
| 模块文件 | 功能描述 |
|---------|---------|
| power_control.py | GPU功率限制设置与读取，支持WSL2/服务器双环境 |
| llm_inference.py | vLLM推理封装，收集TTFT/TBT/E2E指标 |
| load_generator.py | 生成不同长度的推理负载（short/long/mixed） |
| monitor.py | 实时监测GPU功率、显存、温度，计算总能耗 |
| run_experiment.py | 单次实验主流程控制 |
| analyze_results.py | 实验结果分析与可视化，生成11种图表和报告 |
| dynamic_power_inference.py | 动态功率调节策略实现（Prefill/Decode分阶段） |
| adapt_to_server.py | 服务器环境自动适配工具 |
| package_for_server.sh | 代码打包工具（排除大文件） |
| run_all_fixed_experiments.sh | 批量固定功率实验脚本 |

## 运行和开发
### 环境依赖
- **本地环境**：RTX 5070Ti Laptop 12GB显存，WSL2
- **服务器环境**：4×RTX 4080 GPU服务器
- 软件：Python 3.12，vLLM 0.17.0
- 依赖安装：`pip install -r requirements.txt`

### 服务器迁移
项目已完成从本地WSL2环境到实验室GPU服务器的迁移，提供完整的迁移工具链：

1. **代码打包**：`bash package_for_server.sh` - 自动打包代码（排除大文件）
2. **服务器适配**：`python adapt_to_server.py` - 自动适配服务器环境
3. **迁移文档**：`docs/migration_guide.md` - 详细迁移指南
4. **环境测试**：运行后自动生成 `test_server_env.sh` 测试脚本

### 服务器环境配置步骤
1. 解压代码包：`tar -xzf vllm_experiment_code_*.tar.gz`
2. 运行适配脚本：`python adapt_to_server.py`
3. 测试环境：`./test_server_env.sh`
4. 准备模型：从HuggingFace下载或传输模型文件到 `models/` 目录
5. 调整功率档位：编辑 `run_all_fixed_experiments.sh` 中的功率列表（RTX 4080典型功率范围100-240W）

### 运行方式
1. 查看GPU功率信息：`python run_experiment.py --power 240 --show-power-info`
2. 单次实验：`python run_experiment.py --power 240 --load-type mixed --count 100 --concurrency 1`
3. 使用自定义模型：`python run_experiment.py --power 240 --model-path ./models/YourModel`
4. 批量实验：`bash run_all_fixed_experiments.sh`
5. 结果分析：`python analyze_results.py`
6. 跳过功率设置：添加`--skip-set-power`参数支持手动调整功率后运行

### 核心参数
- `--power`: 功率限制（单位W，RTX 4080建议范围150-350W）
- `--concurrency`: 并发请求数
- `--count`: 总请求数
- `--load-type`: 负载类型（short/long/mixed）
- `--model-path`: 模型路径，默认`./Qwen2.5-7B-Instruct-AWQ`
- `--skip-set-power`: 跳过自动设置功率，使用当前系统功率
- `--max-tokens`: 最大生成token数量（默认100）
- `--sudo-password`: sudo密码（用于自动设置功率限制）
- `--show-power-info`: 显示GPU功率信息并退出

## RTX 4080服务器配置说明
- GPU数量：4块
- 单卡显存：16GB GDDR6X
- 典型功耗：240W
- 最大功率：350W
- 建议实验功率范围：150W-350W（间隔50W）
- 功率档位：350, 300, 250, 200, 150W
- 多GPU支持：当前版本使用单GPU，可通过修改device_index参数选择不同GPU

## 功率控制新特性
- 自动检测WSL2/服务器环境，适配不同nvidia-smi路径
- 自动获取GPU功率范围（默认/最大）
- 根据GPU型号智能推荐功率档位
- 支持免密sudo（服务器）和密码sudo（WSL2）两种模式

## 测试策略
- 模块级测试：每个核心模块独立验证功能正确性
- 集成测试：全流程端到端测试，验证指标采集完整性
- 回归测试：每次修改后验证已有功能不受影响
- 异常测试：测试边界条件下的系统鲁棒性
- 环境测试：服务器迁移后使用 `test_server_env.sh` 验证环境

## 编码标准
- 遵循PEP 8 Python编码规范
- 代码注释覆盖率&gt;30%
- 函数命名采用蛇形命名法
- 核心功能模块独立，低耦合高内聚

## AI使用指南
- 优先参考现有代码实现和设计文档
- 功能修改需保持架构一致性
- 实验结果相关修改需保证数据准确性
- 性能优化需优先考虑显存和功耗开销
- 服务器相关修改参考 `docs/migration_guide.md`

## 更新日志
### 2026-03-11
- **基于idea.md完成实验配置**：落实论文实验要求（功率350/300/250/200/150W，并发度8/16/32，max_tokens=100）
- **实现真实TTFT/TBT测量**：修改llm_inference.py，使用独立测量TTFT（max_tokens=1）和完整生成的方法，替代硬编码的100.0/50.0
- **添加max_tokens参数**：修改run_experiment.py添加--max-tokens命令行参数，默认值为100
- **更新批量实验脚本**：并发度改为(8 16 32)，添加--max-tokens参数传递
- **添加自动sudo密码支持**：修改power_control.py支持所有环境的密码输入，添加--sudo-password参数
- **实际功耗发现**：RTX 4080 SUPER在当前负载下实际最高功耗约230W，因此调整测试策略只测试150W和200W
- **结果归档机制**：建立results/old/3.10/、results/3.10.1/等归档目录结构
- **批量脚本配置SUDO_PASSWORD=123456**：支持全自动功率调整
- **新增4张并发度对比图表**：修改analyze_results.py，添加能耗、E2E、TTFT、TBT四张图表，每张图展示不同功率限制（350W→150W）下concurrency=8/32/64/128的对比曲线
- **添加175W功率挡位测试**：创建run_175w_experiments.sh脚本，支持175W单独测试（并发度8/32/64/128，重复2次）

### 2026-03-10
- **完整项目架构扫描与文档更新**
- **最终配置确定**：模型使用Qwen2.5-7B-Instruct-AWQ（AWQ量化版本）
- **模块结构图完善**：添加Mermaid模块结构图并配置可点击链接
- **核心模块扫描**：完成所有Python模块、Shell脚本的功能分析
- **解决vLLM兼容性问题**：修复cuBLAS错误，调整模型加载参数
- **功率说明**：350W是功率上限，实际功耗由负载决定（当前负载约230W）
- **模型路径**：`./Qwen2.5-7B-Instruct-AWQ`
- **功率档位**：350, 300, 250, 200, 150W（从高到低测试）
- 更新模型为Qwen2.5-7B-Instruct-GPTQ-Int4
- 增强功率控制模块，支持RTX 4080最高350W功率范围
- 自动检测WSL2/服务器环境，适配不同nvidia-smi路径
- 添加GPU功率信息查询功能（--show-power-info）
- 添加智能功率档位推荐功能
- 更新实验脚本支持自定义模型路径参数
- 更新批量实验脚本，适配RTX 4080功率范围（120-300W）
- 提升GPU显存利用率至0.90以适配RTX 4080的16GB显存
- 更新环境配置：支持4×RTX 4080 GPU服务器
- 添加RTX 4080专用配置说明和功率建议范围
- 完成项目从本地WSL2到服务器的迁移
- 添加服务器迁移工具：`package_for_server.sh` 和 `adapt_to_server.py`
- 添加详细迁移指南：`docs/migration_guide.md`
- 更新环境依赖说明，支持本地和服务器双环境

### 2026-03-09
- 首次生成项目CLAUDE.md文档
- 完成项目整体架构梳理
- 完成核心模块职责说明
- 添加运行和开发指南
