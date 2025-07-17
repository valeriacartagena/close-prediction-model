import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class AuctionClosePredictor:
    """
    Enhanced models for auction close prediction with directional classification.
    Supports both regression and classification targets for better signal extraction.
    """
    
    def __init__(self, target_col='target', eval_metric='mae', task='regression'):
        self.target_col = target_col
        self.eval_metric = eval_metric
        self.task = task  # 'regression' or 'classification'
        self.models = {}
        self.feature_importance = {}
        
    def compute_target(self, df: pd.DataFrame, price_col='wap', 
                      time_col='seconds_in_bucket', date_col='date_id') -> pd.DataFrame:
        """
        Compute the target: price movement over the last 10 minutes (600 seconds).
        This should match the competition's definition of the target.
        
        Args:
            df: Input dataframe with price and time data
            price_col: Column containing price data (default 'wap')
            time_col: Column containing time within day
            date_col: Column containing date identifiers
            
        Returns:
            DataFrame with computed target
        """
        df = df.copy()
        
        # Sort by date and time to ensure proper temporal order
        df = df.sort_values([date_col, time_col]).reset_index(drop=True)
        
        # Compute target: (auction_close - current_price) / current_price
        # This represents the return over the remaining time to auction close
        
        # For each date, compute the auction close price (last price of the day)
        auction_close = df.groupby(date_col)[price_col].last().reset_index()
        auction_close = auction_close.rename(columns={price_col: 'auction_close_price'})
        
        # Merge back to get auction close price for each row
        df = df.merge(auction_close, on=date_col, how='left')
        
        # Compute target: return to auction close
        df[self.target_col] = (df['auction_close_price'] - df[price_col]) / df[price_col]
        
        # Add directional target for classification
        df['target_direction'] = (df[self.target_col] > 0).astype(int)
        df['target_magnitude'] = np.abs(df[self.target_col])
        
        # Identify tail moves (top 5% by magnitude)
        target_quantiles = df.groupby([date_col, 'stock_id'])['target_magnitude'].transform('quantile', 0.95)
        df['is_tail_move'] = (df['target_magnitude'] > target_quantiles).astype(int)
        
        # Drop the auxiliary column
        df = df.drop(columns=['auction_close_price'])
        
        print(f"Target computed: {self.target_col}")
        print(f"Target stats: mean={df[self.target_col].mean():.6f}, std={df[self.target_col].std():.6f}")
        print(f"Target range: [{df[self.target_col].min():.6f}, {df[self.target_col].max():.6f}]")
        print(f"Directional breakdown: Up={df['target_direction'].sum()}, Down={len(df)-df['target_direction'].sum()}")
        print(f"Tail moves: {df['is_tail_move'].sum()} ({df['is_tail_move'].mean()*100:.1f}%)")
        
        return df
    
    def prepare_lightgbm_data(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):
        """Prepare data for LightGBM training with optional sample weights."""
        # Ensure y_train and y_val are 1-D arrays
        y_train = np.asarray(y_train).ravel()
        if y_val is not None:
            y_val = np.asarray(y_val).ravel()
        
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        return train_data, val_data
    
    def train_lightgbm_regression(self, X_train, y_train, X_val=None, y_val=None, 
                                params=None, sample_weight=None, feature_names=None,
                                num_boost_round=1000, early_stopping_rounds=50) -> Dict[str, Any]:
        """Train LightGBM regression model for return prediction.
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            params: LightGBM parameters
            sample_weight: Optional sample weights
            feature_names: List of feature names
            num_boost_round: Number of boosting rounds (default 1000)
            early_stopping_rounds: Early stopping rounds (default 50)
        Returns: dict with model, importance, best_iteration
        """
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        train_data, val_data = self.prepare_lightgbm_data(X_train, y_train, X_val, y_val, sample_weight)
        
        # Train model
        if val_data is not None:
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                num_boost_round=num_boost_round,
                callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
            )
        else:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                callbacks=[lgb.log_evaluation(0)]
            )
        
        # Get feature importance
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance('gain')
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'importance': importance,
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else num_boost_round
        }
    
    def train_lightgbm_classification(self, X_train, y_train, X_val=None, y_val=None, 
                                    params=None, sample_weight=None, feature_names=None,
                                    num_boost_round=1000, early_stopping_rounds=50) -> Dict[str, Any]:
        """Train LightGBM classification model for directional prediction.
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            params: LightGBM parameters
            sample_weight: Optional sample weights
            feature_names: List of feature names
            num_boost_round: Number of boosting rounds (default 1000)
            early_stopping_rounds: Early stopping rounds (default 50)
        Returns: dict with model, importance, best_iteration
        """
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        train_data, val_data = self.prepare_lightgbm_data(X_train, y_train, X_val, y_val, sample_weight)
        
        # Train model
        if val_data is not None:
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                num_boost_round=num_boost_round,
                callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
            )
        else:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                callbacks=[lgb.log_evaluation(0)]
            )
        
        # Get feature importance
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importance('gain')
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'importance': importance,
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else num_boost_round
        }
    
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None, 
                      target_type='regression') -> Dict[str, Any]:
        """Train ensemble of models for better signal extraction."""
        models = {}
        
        # 1. LightGBM Regression
        if target_type == 'regression':
            lgb_result = self.train_lightgbm_regression(X_train, y_train, X_val, y_val)
            models['lightgbm_reg'] = lgb_result['model']

            # Impute NaNs with column means for Ridge, and drop all-NaN columns
            X_train_ridge = X_train.dropna(axis=1, how='all')
            X_train_ridge = X_train_ridge.fillna(X_train_ridge.mean())
            if X_val is not None:
                X_val_ridge = X_val[X_train_ridge.columns].dropna(axis=1, how='all')
                X_val_ridge = X_val_ridge.fillna(X_train_ridge.mean())
            else:
                X_val_ridge = None

            # 2. Ridge Regression
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_train_ridge, y_train)
            models['ridge'] = ridge
            
        else:  # Classification
            lgb_result = self.train_lightgbm_classification(X_train, y_train, X_val, y_val)
            models['lightgbm_clf'] = lgb_result['model']
            
            # 2. Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            models['random_forest'] = rf
        
        return {
            'models': models,
            'feature_importance': lgb_result['importance']
        }
    
    def backtest_trading(self, model, X_test, y_test, df_test, 
                        position_threshold=0.0) -> Dict[str, Any]:
        """
        Simulate trading based on model predictions.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Actual returns
            df_test: Original test dataframe with metadata
            position_threshold: Minimum prediction threshold to take position
            
        Returns:
            Dictionary with trading metrics
        """
        # Make predictions
        if hasattr(model, 'predict_proba'):
            # Classification model
            pred_proba = model.predict_proba(X_test)[:, 1]
            pred_direction = (pred_proba > 0.5).astype(int)
            # Convert to position (-1, 0, 1)
            position = np.where(pred_proba > 0.5 + position_threshold, 1,
                               np.where(pred_proba < 0.5 - position_threshold, -1, 0))
        else:
            # Regression model
            pred_return = model.predict(X_test)
            position = np.sign(pred_return)
            # Apply threshold
            position = np.where(np.abs(pred_return) > position_threshold, position, 0)
        
        # Calculate PnL
        pnl = position * y_test
        
        # Trading metrics
        total_pnl = pnl.sum()
        sharpe_ratio = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)  # Annualized
        hit_rate = (np.sign(position) == np.sign(y_test)).mean()
        
        # Drawdown calculation
        cumulative_pnl = pnl.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8)
        max_drawdown = drawdown.min()
        
        # Position analysis
        long_positions = (position > 0).sum()
        short_positions = (position < 0).sum()
        no_position = (position == 0).sum()
        
        results = {
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'total_trades': long_positions + short_positions,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'no_position': no_position,
            'avg_position_size': np.abs(position).mean(),
            'pnl_per_trade': total_pnl / (long_positions + short_positions + 1e-8),
            'predictions': pred_return if 'pred_return' in locals() else pred_proba,
            'positions': position,
            'pnl_series': pnl
        }
        
        return results
    
    def evaluate_model(self, model, X, y_true, model_name: str, 
                      task='regression') -> Dict[str, Any]:
        """Evaluate model performance using appropriate metrics."""
        if task == 'regression':
            y_pred = self.predict(model, X)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            print(f"{model_name} Performance (Regression):")
            print(f"  MAE: {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            
            return {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'predictions': y_pred
            }
        else:
            # Classification
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X)
            
            accuracy = accuracy_score(y_true, y_pred)
            
            print(f"{model_name} Performance (Classification):")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Classification Report:")
            print(classification_report(y_true, y_pred))
            
            return {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba if 'y_pred_proba' in locals() else None
            }
    
    def predict(self, model, X) -> np.ndarray:
        """Make predictions with any model type."""
        if hasattr(model, 'predict'):
            return model.predict(X)
        else:
            # For LightGBM models
            return model.predict(X, num_iteration=model.best_iteration)
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Dict]:
        """Train all baseline models."""
        print("Training baseline models...")
        
        results = {}
        
        # Train LightGBM
        print("\n1. Training LightGBM...")
        lgb_result = self.train_lightgbm_regression(X_train, y_train, X_val, y_val)
        self.models['lightgbm'] = lgb_result['model']
        self.feature_importance['lightgbm'] = lgb_result['importance']
        results['lightgbm'] = lgb_result
        
        # Train Ridge
        print("\n2. Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        self.models['ridge'] = ridge
        
        # Ridge feature importance (using coefficients)
        ridge_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(ridge.coef_)
        }).sort_values('importance', ascending=False)
        self.feature_importance['ridge'] = ridge_importance
        results['ridge'] = {'model': ridge, 'importance': ridge_importance}
        
        return results
    
    def evaluate_all_models(self, X_test, y_test) -> Dict[str, Dict]:
        """Evaluate all trained models."""
        print("\nEvaluating all models...")
        
        evaluations = {}
        
        for name, model in self.models.items():
            print(f"\n--- {name.upper()} ---")
            eval_result = self.evaluate_model(model, X_test, y_test, name)
            evaluations[name] = eval_result
        
        # Compare models
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        for name, eval_result in evaluations.items():
            print(f"{name:12} | MAE: {eval_result['mae']:.6f} | RMSE: {eval_result['rmse']:.6f}")
        
        return evaluations
    
    def get_feature_importance_summary(self, top_n=20) -> pd.DataFrame:
        """Get feature importance summary across all models."""
        importance_summary = []
        
        for model_name, importance_df in self.feature_importance.items():
            top_features = importance_df.head(top_n).copy()
            top_features['model'] = model_name
            top_features['rank'] = range(1, len(top_features) + 1)
            importance_summary.append(top_features)
        
        if importance_summary:
            return pd.concat(importance_summary, ignore_index=True)
        else:
            return pd.DataFrame()


def validate_target_computation(df: pd.DataFrame, target_col='target', 
                              price_col='wap', time_col='seconds_in_bucket', 
                              date_col='date_id') -> bool:
    """
    Validate that target computation is correct.
    
    Args:
        df: DataFrame with computed target
        target_col: Target column name
        price_col: Price column name
        time_col: Time column name
        date_col: Date column name
        
    Returns:
        bool: True if target computation appears correct
    """
    print("Validating target computation...")
    
    # Check 1: Target should be a return (percentage change)
    target_stats = df[target_col].describe()
    print(f"Target statistics:")
    print(f"  Mean: {target_stats['mean']:.6f}")
    print(f"  Std: {target_stats['std']:.6f}")
    print(f"  Min: {target_stats['min']:.6f}")
    print(f"  Max: {target_stats['max']:.6f}")
    
    # Check 2: Target should be reasonable (typically small percentages)
    if abs(target_stats['mean']) > 0.1:  # More than 10% average return
        print("⚠  Warning: Average target seems high (>10%)")
        return False
    
    # Check 3: Target should vary by time (earlier times should have larger targets)
    time_target_corr = df.groupby(time_col)[target_col].mean()
    if len(time_target_corr) > 1:
        # Later times should generally have smaller targets (closer to auction)
        if time_target_corr.iloc[-1] > time_target_corr.iloc[0]:
            print("⚠  Warning: Target doesn't decrease with time as expected")
            return False
    
    print("[✔] Target computation appears correct")
    return True 