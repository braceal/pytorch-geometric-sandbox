# pytorch-geometric-sandbox

# Installation
```
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install scipy
pip install torch==1.7.0
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-geometric
pip install -e .
```
