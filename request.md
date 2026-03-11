我需要完成我的本科论文《基于动态功率调节的大语言模型推理能耗控制》，现在我需要做实验：基于 vLLM 或其他推理框架，在GPU服务器（现在先在我的本地电脑 5070ti laptop 12GB）上运行推理实验，并实现推理过程的性能与功耗监测。
1 部署 LLM（vLLM）
2 设置 GPU Power Cap
3 发送推理请求
4 记录 TTFT/TBT/E2E
5 记录 GPU Power
6 计算 Energy
7 对比不同策略