import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features import create_features
from models import AuctionClosePredictor
from preprocessing import Preprocessor
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from backtest import backtest_model
import joblib

# ========== CONFIG ========== #
FAST_MODE = True  # Set to False for full run
PLOT_ENABLED = not FAST_MODE
TRAIN_FEATURES_CACHE = 'enhanced_train_features_cached.parquet'
VAL_FEATURES_CACHE = 'enhanced_val_features_cached.parquet'

# ========== 1. DATA LOADING & SPLIT ========== #
def load_and_split():
    df = pd.read_csv('data/train.csv')
    train_df = df[df['date_id'] < 3].copy()
    val_df = df[df['date_id'] == 3].copy()
    return train_df, val_df

# ========== 2. FEATURE ENGINEERING WITH CACHING ========== #
def get_features(df, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached features: {cache_path}")
        return pd.read_parquet(cache_path)
    else:
        print(f"Creating features for {cache_path}...")
        feats = create_features(df)
        feats.to_parquet(cache_path)
        return feats

# ========== 3. TARGET & LABEL ENGINEERING ========== #
def ensure_targets(df):
    if 'target_direction' not in df.columns:
        df['target_direction'] = (df['target'] > 0).astype(int)
    if 'is_tail_move' not in df.columns:
        q90 = df['target'].abs().quantile(0.90)
        df['is_tail_move'] = (df['target'].abs() > q90).astype(int)
    return df

def get_sample_weights(df):
    return np.where(df['is_tail_move'].astype(bool), 3.0, 1.0)

# ========== 4. PREPROCESSING ========== #
def preprocess(train_df, val_df, features):
    pre = Preprocessor()
    # Exclude row_id and non-features from encoding
    exclude = ['row_id']
    train_df = pre.encode_categoricals(train_df, exclude=exclude)
    val_df = pre.encode_categoricals(val_df, exclude=exclude)
    scaled = pre.scale(train_df, val_df)
    train_df, val_df = scaled[0], scaled[1]  # Only take the first two
    return train_df, val_df

# ========== 5. FEATURE/LABEL EXTRACTION ========== #
def extract_X_y(df, features, label):
    X = df[features].values
    y = df[label].values.ravel()
    return X, y

# ========== 6. MODEL TRAINING ========== #
def train_models(X_train, y_train, X_val, y_val, features, train_weights):
    predictor = AuctionClosePredictor()
    lgb_params = {'num_boost_round': 200, 'early_stopping_rounds': 20}
    # LightGBM Regression
    try:
        lgb_reg = predictor.train_lightgbm_regression(
            X_train, y_train, X_val, y_val,
            sample_weight=train_weights,
            feature_names=features,
            **lgb_params
        )
        if not isinstance(lgb_reg, dict) or 'model' not in lgb_reg:
            lgb_reg = {'model': None}
    except Exception as e:
        logging.warning(f"LightGBM regression training failed: {e}")
        lgb_reg = {'model': None}
    # LightGBM Classification
    try:
        lgb_clf = predictor.train_lightgbm_classification(
            X_train, (y_train > 0).astype(int), X_val, (y_val > 0).astype(int),
            feature_names=features,
            **lgb_params
        )
        if not isinstance(lgb_clf, dict) or 'model' not in lgb_clf:
            lgb_clf = {'model': None}
    except Exception as e:
        logging.warning(f"LightGBM classification training failed: {e}")
        lgb_clf = {'model': None}
    # Ridge Ensemble
    X_train_ridge = pd.DataFrame(X_train, columns=features)
    X_val_ridge = pd.DataFrame(X_val, columns=features)
    X_train_ridge = X_train_ridge.dropna(axis=1, how='all').fillna(X_train_ridge.mean())
    X_val_ridge = X_val_ridge[X_train_ridge.columns].fillna(X_train_ridge.mean())
    from sklearn.linear_model import Ridge
    class DummyRidge:
        def predict(self, X):
            return np.zeros(X.shape[0])
    try:
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_ridge, y_train)
    except Exception as e:
        logging.warning(f"Ridge regression training failed: {e}")
        ridge = DummyRidge()
    # Export feature importances if available
    if lgb_reg.get('model', None) is not None and hasattr(lgb_reg['model'], 'feature_importances_'):
        fi = pd.DataFrame({'feature': features, 'importance': lgb_reg['model'].feature_importances_})
        fi.sort_values('importance', ascending=False).to_csv('website_results/feature_importance_v2.csv', index=False)
    return lgb_reg, lgb_clf, ridge, X_train_ridge, X_val_ridge

# ========== 7. EVALUATION ========== #
def evaluate_models(lgb_reg, lgb_clf, ridge, X_val, y_val, y_val_dir, X_val_ridge):
    from sklearn.metrics import mean_absolute_error, accuracy_score
    if lgb_reg is None:
        lgb_reg = {}
    if lgb_clf is None:
        lgb_clf = {}
    if ridge is None:
        class DummyRidge:
            def predict(self, X):
                return np.zeros(X.shape[0])
        ridge = DummyRidge()
    # Regression
    reg_preds = None
    reg_pnl = None
    if isinstance(lgb_reg, dict) and 'model' in lgb_reg and lgb_reg['model'] is not None and hasattr(lgb_reg['model'], 'predict'):
        reg_preds = lgb_reg['model'].predict(X_val)
        reg_mae = mean_absolute_error(y_val, reg_preds)
        reg_dir_acc = np.mean(np.sign(reg_preds) == np.sign(y_val))
        reg_pnl = np.sign(reg_preds) * y_val
    else:
        reg_mae = None
        reg_dir_acc = None
        logging.warning("LightGBM regression model is None or invalid.")
    # Classification
    clf_preds = None
    clf_pnl = None
    if isinstance(lgb_clf, dict) and 'model' in lgb_clf and lgb_clf['model'] is not None and hasattr(lgb_clf['model'], 'predict'):
        clf_preds = (lgb_clf['model'].predict(X_val) > 0.5).astype(int)
        clf_acc = accuracy_score(y_val_dir.astype(int), clf_preds)
        clf_pnl = np.sign(clf_preds) * y_val
    else:
        clf_acc = None
        logging.warning("LightGBM classification model is None or invalid.")
    # Ridge
    ridge_preds = None
    ridge_pnl = None
    if ridge is not None and hasattr(ridge, 'predict'):
        ridge_preds = ridge.predict(X_val_ridge)
        ridge_mae = mean_absolute_error(y_val, ridge_preds)
        ridge_dir_acc = np.mean(np.sign(ridge_preds) == np.sign(y_val))
        ridge_pnl = np.sign(ridge_preds) * y_val
    else:
        ridge_mae = None
        ridge_dir_acc = None
        logging.warning("Ridge regression model is None or invalid.")
    logging.info(f"Regression MAE: {reg_mae}, Directional Acc: {reg_dir_acc}")
    logging.info(f"Classification Acc: {clf_acc}")
    logging.info(f"Ridge MAE: {ridge_mae}, Ridge Dir Acc: {ridge_dir_acc}")
    return {
        'reg_mae': reg_mae,
        'reg_dir_acc': reg_dir_acc,
        'clf_acc': clf_acc,
        'ridge_mae': ridge_mae,
        'ridge_dir_acc': ridge_dir_acc,
        'reg_preds': reg_preds,
        'reg_pnl': reg_pnl,
        'clf_preds': clf_preds,
        'clf_pnl': clf_pnl,
        'ridge_preds': ridge_preds,
        'ridge_pnl': ridge_pnl
    }

# ========== 8. MAIN PIPELINE ========== #
def main():
    start = time.time()
    logging.info("\n Enhanced Dev Pipeline (FAST_MODE = {}):".format(FAST_MODE))
    # 1. Data
    train_df, val_df = load_and_split()
    # 2. Features
    train_features = get_features(train_df, TRAIN_FEATURES_CACHE)
    val_features = get_features(val_df, VAL_FEATURES_CACHE)
    # 3. Targets
    train_features = ensure_targets(train_features)
    val_features = ensure_targets(val_features)
    # 4. Feature selection
    ignore = ['row_id', 'target', 'target_direction', 'is_tail_move']
    features = [c for c in train_features.columns if c not in ignore and train_features[c].dtype != 'O']
    # Save feature names for submission generator
    joblib.dump(features, 'models/feature_names.pkl')
    # 5. Preprocessing
    train_processed, val_processed = preprocess(train_features, val_features, features)
    # 6. Extract X/y
    X_train, y_train = extract_X_y(train_processed, features, 'target')
    X_val, y_val = extract_X_y(val_processed, features, 'target')
    y_train_dir = train_processed['target_direction'].values.ravel()
    y_val_dir = val_processed['target_direction'].values.ravel()
    train_weights = get_sample_weights(train_processed)
    # 7. Model training
    lgb_reg, lgb_clf, ridge, X_train_ridge, X_val_ridge = train_models(
        X_train, y_train, X_val, y_val, features, train_weights)

    # Save trained models for submission generation
    os.makedirs('models', exist_ok=True)
    if lgb_reg and lgb_reg.get('model', None) is not None:
        joblib.dump(lgb_reg['model'], 'models/lgbm_reg.pkl')
    if lgb_clf and lgb_clf.get('model', None) is not None:
        joblib.dump(lgb_clf['model'], 'models/lgbm_clf.pkl')
    if ridge is not None:
        joblib.dump(ridge, 'models/ridge.pkl')

    # 8. Evaluation
    metrics = evaluate_models(lgb_reg, lgb_clf, ridge, X_val, y_val, y_val_dir, X_val_ridge)

    # 8b. Backtesting (unified, logs, and saves metrics)
    logging.info("\nBacktest Results:")
    enhanced_metrics = {}
    for model_name, pred_key in [('LGBM Regression', 'reg_preds'), ('LGBM Classification', 'clf_preds'), ('Ridge', 'ridge_preds')]:
        preds = metrics.get(pred_key)
        if preds is not None:
            # Log shape and sample
            logging.info(f"{model_name} preds shape: {preds.shape}, sample: {preds[:5]}")
            bt = backtest_model(preds, y_val, model_name, threshold=0.0, confidence_weighted=False)
            enhanced_metrics[model_name] = bt
            logging.info(f"{model_name}: Hit Rate={bt['hit_rate']:.3f}, Sharpe={bt['sharpe']:.3f}, Max Drawdown={bt['max_drawdown']:.3f}, Total PnL={bt['total_pnl']:.3f}, Trades={bt['num_trades']}")
    # Save all metrics to a single JSON for Streamlit
    import json
    with open('website_results/enhanced_metrics.json', 'w') as f:
        json.dump(enhanced_metrics, f, indent=2)

    # 9. Plotting (optional)
    if PLOT_ENABLED:
        if lgb_reg is None:
            lgb_reg = {}
        if lgb_clf is None:
            lgb_clf = {}
        if ridge is None:
            class DummyRidge:
                def predict(self, X):
                    return np.zeros(X.shape[0])
            ridge = DummyRidge()
        plt.figure(figsize=(8, 4))
        # Use actual PnL logic
        if metrics.get('reg_pnl') is not None:
            plt.plot(np.cumsum(metrics['reg_pnl']), label='LGBM Reg Cumulative PnL')
        if metrics.get('ridge_pnl') is not None:
            plt.plot(np.cumsum(metrics['ridge_pnl']), label='Ridge Cumulative PnL')
        plt.legend()
        plt.title('Cumulative PnL (Validation)')
        plt.tight_layout()
        plt.savefig('website_results/pnl_curve_v2.png', dpi=150)
        plt.close()
    logging.info(f"\n[✔] Pipeline completed in {time.time() - start:.1f} sec.")
    logging.info(metrics)

# Thresholding logic placeholder for FAST_MODE=False
# (implement position thresholding here in future if needed)

if __name__ == "__main__":
    main() 