# rosoku


# CPUで実行
pythonで普通に実行すればOK

# 単一ノード上でtorchrun
```
srun --nodelist=sirocco07 --ntasks=2 --cpus-per-task=16 --exclusive --pty bash

torchrun --nproc_per_node=2 --master_addr=127.0.0.1  --master_port=29627  ./main.py
```

# 複数ノードで実行
```
srunで実行
```

# sphinx
`docs`内で以下を実行

```
sphinx-apidoc -o ../docs/source ../rosoku && make html


sphinx-apidoc -o ../docs/source ../rosoku && sphinx-multiversion source build/html
```