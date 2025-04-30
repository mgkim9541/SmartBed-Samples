# run_pipeline.py

# ----------------------------
# 🏁 Main pipeline entry point
# ----------------------------
from src.model import build_cnn_lstm_model
from src.preprocess import preprocess_dataset
from src.train_and_eval import train_and_evaluate, analyze_performance

def main():
    # 1. Preprocess data
    X_train, X_test, Y_train, Y_test = preprocess_dataset("data/Sample Data.csv")

    # 2. Build model
    model = build_cnn_lstm_model()

    # 3. Train and evaluate
    history, eval_results = train_and_evaluate(model, X_train, Y_train, X_test, Y_test)

    # 4. Post-evaluation analysis
    analyze_performance(model, X_test, Y_test, history)

if __name__ == "__main__":
    main()