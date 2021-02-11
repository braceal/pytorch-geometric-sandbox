# pytorch-geometric-sandbox

Preliminary modeling of molecular dynamics protein trajectories using [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric).

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

# Run
```
python mdgraph/models/<model-name>.py
```
See inside specific model py file for any additional command line options.

# Viewing results
Check in the `plot/` directory for example html files that can be viewed in the browser.
