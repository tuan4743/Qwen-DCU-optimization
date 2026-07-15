# 2026先导杯-山西大学登队 T2026101089911233
## 仓库地址
https://gitlab.eduxiji.net/T2026101089911233/2026pra-t2026101089911233
## 项目说明
基于vLLM原始框架完成KV缓存、CUDA算子、量化推理性能优化，完整保留基线目录结构，可直接编译评测。
## 编译运行
pip install -r requirements.txt
python setup.py bdist_wheel
pip install dist/*.whl
