# CNN-LSTM Multi-Output Classification

This project is the source code and sample data for the paper titled _Participatory Design of Autonomous AI-based Healthcare IS: Evidence with a Novel Smart Medical Bed Using Deep Learning_ under review at ESWA which implements a hybrid CNN-LSTM model that predicts two output labels (`posture` and `comfort`) from sensor-based input data.

## 🧩 Project Structure

```
cnn_lstm_pipeline/
├── run_pipeline.py            # Entry point to run the entire pipeline
├── data/
│   └── Sample Data.csv        # Sample sensor data
└── src/
    ├── model.py               # CNN-LSTM model architecture
    ├── preprocess.py          # Data preprocessing module
    └── train_and_eval.py      # Training, evaluation, and visualization
```

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Execute the pipeline:
```bash
python run_pipeline.py
```

## 📊 Output

- Training & validation accuracy (for both posture and comfort)
- Confusion matrices
- Hamming loss and exact match score

## 📝 Notes

- Make sure `Sample Data.csv` follows the format:
  - Features: 160 columns
  - Labels: `Pos` and `Com` columns (one-hot encoded internally)

---

## How to cite

If this code accompanies a paper, include the citation here.

```
@article{Research article,
  title   = {Participatory Design of Autonomous AI-based Healthcare IS: Evidence with a Novel Smart Medical Bed Using Deep Learning},
  author  = {Min-Gyeong Kim, Francis Joseph Costello, Cheong Kim},
  journal = {},
  year    = {2025}
}
```
