# PreGS
Parameter-Transfer-Based Multi-Expert Graph Neural Network for Node Classification

## Dataset and Usage

The datasets are provided in `datasets.zip`. Before running the code, extract the archive into the repository root and make sure the directory structure is:

```text
PreGS/
├── PreGSGitHub.py
├── requirements.txt
└── datasets/
    ├── acm/
    │   ├── acm_adj.npy
    │   ├── acm_feat.npy
    │   └── acm_label.npy
    └── ...
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python PreGSGitHub.py
```

The script includes GAT, PreGS, and PreGSv2. Dataset selection and experiment repetitions can be configured in `PreGSGitHub.py`.
