# Code Sample
1. Loads the provided *Sample Data*.
2. Preprocesses the data.
3. Trains and evaluates the machine‑learning models described in the manuscript.

---

## Repository Structure

```
.
├── data/
│   └── Sample Data.csv      # anonymised sample dataset (small, public)
├── Codes/
│   └── __init__.py
│   └── preprocessing.py     # functions for data cleaning / feature engineering
│   └── models.py            # model definitions & training loops
│   └── evaluate.py          # metrics & visualisation utilities
├── scripts/
│   └── run_pipeline.py      # single‑entry script – `python scripts/run_pipeline.py`
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```
---

## Quick‑start

```bash
# 1. Clone the repo
git clone https://github.com/mgkim9541/SmartBed‑Samples.git
cd SmartBed‑Samples

# 2. Create environment (conda or venv)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python scripts/run_pipeline.py --data ./data/Sample\ Data.csv
```

The `run_pipeline.py` script loads the CSV, performs preprocessing, trains the model(s), and prints evaluation metrics.  
Use `‑h` for full CLI options (e.g., choosing model type, epochs, random seed).

---

## Re‑using with your own data

Simply point the script to another CSV file with the same schema, **or** adapt `preprocessing.py` to match your columns:

```bash
python scripts/run_pipeline.py --data /path/to/your/data.csv
```

---

## Requirements

Minimal versions (feel free to pin exact versions):

```
numpy
pandas
scikit‑learn
tensorflow>=2.15
keras
tqdm
plotly
requests
multiSmote   # if you cannot install, comment out in preprocessing.py
```

---

## License

Specify the license for your code (e.g., MIT).  
Add a `LICENSE` file before pushing.

---

## How to cite

If this code accompanies a paper, include the citation here.

```
@article{Your2025Paper,
  title   = {Title},
  author  = {Your Name},
  journal = {Expert Systems With Applications},
  year    = {2025}
}
```
