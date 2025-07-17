import argparse
import pandas as pd
import numpy as np
import os
import joblib
from features import create_features
from preprocessing import Preprocessor
from models import AuctionClosePredictor


def main():
    parser = argparse.ArgumentParser(description='Generate submission file for Optiver competition.')
    parser.add_argument('--model', type=str, default='lgbm_reg', choices=['lgbm_reg', 'lgbm_clf', 'ridge'], help='Model to use for prediction')
    parser.add_argument('--input', type=str, default='../example_test_files/test.csv', help='Path to test.csv (default: ../data/test.csv, relative to script)')
    parser.add_argument('--output', type=str, default='../submission/submission.csv', help='Path to save submission CSV')
    parser.add_argument('--feature_names', type=str, default='models/feature_names.pkl', help='Path to feature names pickle file')
    args = parser.parse_args()

    # Load test data (auto-detect path)
    candidate_paths = [args.input, '../data/example_test_files/test.csv', 'data/example_test_files/test.csv']
    test_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            test_path = path
            break
    if test_path is None:
        print(f"[ERROR] Test file not found in any of: {candidate_paths}\nPlease check the path and try again.")
        exit(1)
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    row_ids = test_df['row_id'] if 'row_id' in test_df.columns else np.arange(len(test_df))

    # Feature engineering
    print("Running feature engineering...")
    test_features = create_features(test_df)

    # Preprocessing
    print("Preprocessing test features...")
    # For simplicity, fit preprocessor on test (or load from train if available)
    pre = Preprocessor()
    test_processed = pre.encode_categoricals(test_features, exclude=['row_id'])
    test_processed = pre.scale(test_processed)[0]

    # Feature selection (match training features)
    feature_names_path = args.feature_names
    if not os.path.exists(feature_names_path):
        print(f"[ERROR] Feature names file not found: {feature_names_path}")
        exit(1)
    features = joblib.load(feature_names_path)
    # Add missing features as zeros
    for col in features:
        if col not in test_processed.columns:
            test_processed[col] = 0
    # Ensure correct order
    X_test = test_processed[features].values

    # Load model
    MODEL_PATH = f'models/{args.model}.pkl'
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}\nPlease train and save your model using joblib.dump(trained_model, '{MODEL_PATH}') before generating a submission.")
        exit(1)
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)
    # Note: Save your trained model after training with joblib.dump(trained_model, MODEL_PATH)

    # Export submission
    print(f"Saving submission to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    submission = pd.DataFrame({'row_id': row_ids, 'target': preds})
    submission.to_csv(args.output, index=False)
    print(f"Submission saved.\nSummary stats:")
    print(submission['target'].describe())

if __name__ == '__main__':
    main() 