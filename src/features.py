# Functions for feature engineering
import numpy as np
import pandas as pd
from scipy import stats


def create_features(df, target_transform='log'):
    """Create new features from the dataframe.
    
    Args:
        df: Input dataframe
        target_transform: 'log', 'standardize', or None for target transformation
    """
    # --- Price-based Features (Fixed lookahead bias) ---
    if 'wap' in df.columns:
        df['return'] = df['wap'].pct_change()
        df['log_return'] = np.log(df['wap'] / df['wap'].shift(1))
        # Fix lookahead bias: use shift(1) for all rolling computations
        for window in [5, 10, 20]:
            df[f'wap_roll_mean_{window}'] = df['wap'].shift(1).rolling(window).mean()
            df[f'wap_roll_std_{window}'] = df['wap'].shift(1).rolling(window).std()
    if 'high' in df.columns and 'low' in df.columns:
        df['high_low_ratio'] = df['high'] / df['low']
    if 'wap' in df.columns and 'open' in df.columns:
        df['wap_open_ratio'] = df['wap'] / df['open']

    # --- Technical Indicators (Fixed lookahead bias) ---
    if 'wap' in df.columns:
        # RSI (14) - use shift(1) to avoid lookahead
        delta = df['wap'].shift(1).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        # EMA with shift(1)
        df['ema_12'] = df['wap'].shift(1).ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['wap'].shift(1).ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        # Bollinger Bands with shift(1)
        df['bb_middle'] = df['wap'].shift(1).rolling(window=20).mean()
        df['bb_std'] = df['wap'].shift(1).rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # --- AW Price (WAP) with Protection ---
    if 'wap' in df.columns:
        df['aw_price'] = df['wap']
    elif {'bid_price', 'ask_price', 'bid_size', 'ask_size'}.issubset(df.columns):
        # Calculate WAP with protection against div-by-zero
        denom = df['bid_size'] + df['ask_size']
        df['aw_price'] = np.where(denom != 0,
                                  (df['bid_price'] * df['bid_size'] + df['ask_price'] * df['ask_size']) / denom,
                                  np.nan)

    # --- Mid Price & Spread ---
    if 'bid_price' in df.columns and 'ask_price' in df.columns:
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        df['spread'] = df['ask_price'] - df['bid_price']

    # --- Order Flow Imbalance (Optional Feature) ---
    if 'bid_size' in df.columns and 'ask_size' in df.columns:
        df['order_flow_imbalance'] = df['bid_size'] - df['ask_size']
        df['order_flow_imbalance_change'] = df['order_flow_imbalance'].diff()

    # --- Micro Price (L1 Weighted Average) ---
    if {'bid_price', 'ask_price', 'bid_size', 'ask_size'}.issubset(df.columns):
        # Micro price using L1 (Level 1) order book
        total_size = df['bid_size'] + df['ask_size']
        df['micro_price'] = np.where(total_size != 0,
                                     (df['bid_price'] * df['bid_size'] + df['ask_price'] * df['ask_size']) / total_size,
                                     np.nan)

    # --- Volatility (Rolling Std of Returns) ---
    if 'return' in df.columns:
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['return'].shift(1).rolling(window).std()

    # --- Z-Score WAP (Price vs Rolling Mean/Std) ---
    if 'wap' in df.columns:
        for window in [5, 10, 20]:
            roll_mean = df['wap'].shift(1).rolling(window).mean()
            roll_std = df['wap'].shift(1).rolling(window).std()
            df[f'z_score_wap_{window}'] = np.where(roll_std != 0,
                                                     (df['wap'] - roll_mean) / roll_std,
                                                     np.nan)

    # --- Auction Intensity & Momentum Features ---
    # Proxy volume: matched_size * wap if 'volume' is missing
    if 'volume' not in df.columns and 'matched_size' in df.columns and 'wap' in df.columns:
        df['proxy_volume'] = df['matched_size'] * df['wap']
    elif 'volume' in df.columns:
        df['proxy_volume'] = df['volume']
    
    if 'matched_size' in df.columns:
        # Matched size rate of change
        df['matched_size_rate'] = df['matched_size'].diff()
        
        # Volume ramp up (last 30s vs earlier average)
        if 'proxy_volume' in df.columns:
            df['volume_ramp_up'] = df['proxy_volume'].shift(1).rolling(30).mean() / df['proxy_volume'].shift(31).rolling(300).mean()
    
    if 'imbalance_size' in df.columns and 'matched_size' in df.columns:
        # Imbalance to matched size ratio
        df['imbalance_size_to_matched_size'] = np.where(df['matched_size'] != 0,
                                                        df['imbalance_size'] / df['matched_size'],
                                                        np.nan)
    
    # Price change since auction open
    if 'mid_price' in df.columns:
        df['price_change_since_auction_open'] = df['mid_price'] - df.groupby(['date_id', 'stock_id'])['mid_price'].transform('first')
    elif 'wap' in df.columns:
        df['price_change_since_auction_open'] = df['wap'] - df.groupby(['date_id', 'stock_id'])['wap'].transform('first')
    elif 'reference_price' in df.columns:
        df['price_change_since_auction_open'] = df['reference_price'] - df.groupby(['date_id', 'stock_id'])['reference_price'].transform('first')
    
    # Auction volatility (very short-term)
    if 'mid_price' in df.columns:
        for window in [5, 10, 30]:
            df[f'auction_volatility_{window}s'] = df['mid_price'].shift(1).rolling(window).std()
    elif 'wap' in df.columns:
        for window in [5, 10, 30]:
            df[f'auction_volatility_{window}s'] = df['wap'].shift(1).rolling(window).std()
    elif 'reference_price' in df.columns:
        for window in [5, 10, 30]:
            df[f'auction_volatility_{window}s'] = df['reference_price'].shift(1).rolling(window).std()
    
    # --- Time-to-Auction Dynamics ---
    if 'seconds_in_bucket' in df.columns:
        # Normalize seconds to progress bar [0, 1]
        max_seconds = df.groupby(['date_id', 'stock_id'])['seconds_in_bucket'].transform('max')
        df['seconds_bucket_pct'] = df['seconds_in_bucket'] / max_seconds
        
        # Final minutes flag (last 120 seconds)
        df['final_minutes_flag'] = (df['seconds_in_bucket'] >= 480).astype(int)
        
        # Final 30 seconds flag
        df['final_30s_flag'] = (df['seconds_in_bucket'] >= 570).astype(int)
    
    # --- Auction Book Microstructure ---
    if 'imbalance_price' in df.columns and 'mid_price' in df.columns:
        df['imbalance_price_vs_mid'] = df['imbalance_price'] - df['mid_price']
        df['imbalance_price_vs_mid_ratio'] = df['imbalance_price'] / df['mid_price']
    
    if 'far_price' in df.columns and 'imbalance_price' in df.columns:
        df['far_vs_imbalance_price'] = df['far_price'] - df['imbalance_price']
    
    if 'near_price' in df.columns and 'imbalance_price' in df.columns:
        df['near_vs_imbalance_price'] = df['near_price'] - df['imbalance_price']
    
    if 'imbalance_price' in df.columns:
        # Imbalance price slope
        df['imbalance_price_slope'] = df['imbalance_price'].shift(1).rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
        )
    
    if 'far_price' in df.columns and 'near_price' in df.columns:
        df['auction_price_spread'] = df['far_price'] - df['near_price']
    
    # --- Trend & Signal Features ---
    if 'mid_price' in df.columns:
        for window in [5, 10, 30]:
            df[f'rolling_slope_price_{window}s'] = df['mid_price'].shift(1).rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
            )
    
    if 'proxy_volume' in df.columns:
        df['volume_delta_rolling'] = df['proxy_volume'] - df['proxy_volume'].shift(1).rolling(10).mean()
    
    if 'imbalance_size' in df.columns and 'proxy_volume' in df.columns:
        df['imbalance_ratio_trend'] = df['imbalance_size'].shift(1).rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x / df['proxy_volume'].iloc[len(x)-1], 1)[0] if len(x) > 1 else np.nan
        )
    
    # --- Classification Features ---
    if 'imbalance_buy_sell_flag' in df.columns:
        df['imbalance_buy'] = (df['imbalance_buy_sell_flag'] == 1).astype(int)
        df['imbalance_sell'] = (df['imbalance_buy_sell_flag'] == -1).astype(int)
        # Imbalance direction change
        df['imbalance_direction_change'] = df['imbalance_buy_sell_flag'].diff()
        
        # Imbalance direction classification
        df['imbalance_direction'] = np.where(df['imbalance_buy_sell_flag'] > 0.1, 'buy',
                                            np.where(df['imbalance_buy_sell_flag'] < -0.1, 'sell', 'neutral'))
    
    # Auction pressure level (based on imbalance magnitude)
    if 'imbalance_size' in df.columns:
        # Compute group quantile on the DataFrame, not Series
        imbalance_quantiles = df.groupby(['date_id', 'stock_id'])['imbalance_size'].transform(lambda x: x.abs().quantile(0.75))
        imbalance_abs = df['imbalance_size'].abs()
        df['auction_pressure_level'] = np.where(imbalance_abs > imbalance_quantiles * 1.5, 'high',
                                               np.where(imbalance_abs > imbalance_quantiles * 0.5, 'medium', 'low'))
    
    # Market regime flag (based on volatility)
    if 'return' in df.columns and 'date_id' in df.columns and 'stock_id' in df.columns:
        df['volatility'] = df['return'].shift(1).rolling(20).std()
        vol_quantiles = df.groupby(['date_id', 'stock_id'])['volatility'].transform('quantile', 0.75)
        df['market_regime_flag'] = np.where(df['volatility'] > vol_quantiles * 1.5, 'volatile',
                                           np.where(df['volatility'] < vol_quantiles * 0.5, 'trending', 'mean_reverting'))
    
    # --- Deep Group-Based Stats (Time-Sensitive) ---
    if all(col in df.columns for col in ['date_id', 'stock_id', 'imbalance_buy_sell_flag']):
        # Lagged auction features from previous days
        df['avg_imbalance_direction_last_5_days'] = df.groupby('stock_id')['imbalance_buy_sell_flag'].shift(1).rolling(5).mean()
    
    if all(col in df.columns for col in ['date_id', 'stock_id', 'return']):
        df['prev_day_final_return'] = df.groupby('stock_id')['return'].shift(1)
    
    # Percentile rank of current auction pressure vs historical
    if 'imbalance_size' in df.columns and 'date_id' in df.columns and 'stock_id' in df.columns:
        df['imbalance_pressure_percentile'] = df.groupby(['date_id', 'stock_id'])['imbalance_size'].rank(pct=True)
    
    # --- Auction Price Deviation Prediction ---
    if 'mid_price' in df.columns and 'imbalance_price' in df.columns:
        # Gap between current price and auction price
        df['auction_price_gap'] = df['mid_price'] - df['imbalance_price']
        df['auction_price_gap_ratio'] = df['mid_price'] / df['imbalance_price']
        
        # Trend in the gap
        df['auction_price_gap_trend'] = df['auction_price_gap'].shift(1).rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
        )
    
    # --- Volume Features (Fixed lookahead bias) ---
    if 'proxy_volume' in df.columns:
        df['volume_roll_mean_10'] = df['proxy_volume'].shift(1).rolling(10).mean()
    
    # Matched volume percentage (auction-specific)
    if 'matched_size' in df.columns and 'proxy_volume' in df.columns:
        df['matched_volume_pct'] = df['matched_size'] / df['proxy_volume']
    elif 'matched_size' in df.columns:
        # Use rolling average proxy volume as denominator
        df['matched_volume_pct'] = df['matched_size'] / df['proxy_volume'].shift(1).rolling(10).mean()

    # --- Historical Target Features (Fixed lookahead bias) ---
    if 'target' in df.columns:
        # Target lags (no lookahead bias here)
        for lag in [1, 2]:
            df[f'target_lag_{lag}'] = df['target'].shift(lag)
        
        # Target rolling stats (fixed lookahead bias)
        for window in [5, 10]:
            df[f'target_roll_mean_{window}'] = df['target'].shift(1).rolling(window).mean()
            df[f'target_roll_std_{window}'] = df['target'].shift(1).rolling(window).std()
        
        # Target transformation
        if target_transform == 'log':
            df['log_target'] = np.log1p(df['target'])
        elif target_transform == 'standardize':
            # Use rolling standardization to avoid lookahead
            target_mean = df['target'].shift(1).rolling(20).mean()
            target_std = df['target'].shift(1).rolling(20).std()
            df['target_standardized'] = (df['target'] - target_mean) / target_std

    # --- Time Features (Simplified) ---
    if 'timestamp' in df.columns:
        dt = pd.to_datetime(df['timestamp'])
        df['hour'] = dt.dt.hour
        df['minute'] = dt.dt.minute
        df['dayofweek'] = dt.dt.dayofweek
        df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        
        # Time-to-close feature (critical for auction dynamics)
        if len(dt) > 0:
            max_time = dt.max()
            df['seconds_to_close'] = (max_time - dt).dt.total_seconds()
            # Normalize to 0-1 range (0 = close, 1 = start of period)
            df['time_to_close_normalized'] = df['seconds_to_close'] / df['seconds_to_close'].max()

    # --- Magic Features (Group-based ratios and rankings) ---
    if 'seconds_in_bucket' in df.columns:
        # Create seconds_in_bucket_group
        df['seconds_in_bucket_group'] = np.where(df['seconds_in_bucket'] < 300, 0,
                                                np.where(df['seconds_in_bucket'] < 480, 1, 2)).astype(float)
        
        # Define base features for group operations
        base_features = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out already created features and target
        exclude_cols = ['seconds_in_bucket_group', 'target', 'log_target', 'target_standardized']
        base_features = [col for col in numeric_cols if col not in exclude_cols and not col.startswith(('target_lag_', 'target_roll_'))]
        # If 'proxy_volume' is present, replace 'volume' with 'proxy_volume' in base_features
        if 'proxy_volume' in base_features:
            base_features = [col if col != 'volume' else 'proxy_volume' for col in base_features]
        
        # Group-based features
        if 'date_id' in df.columns and 'stock_id' in df.columns:
            # Group first ratio features
            for col in base_features:
                if col in df.columns:
                    # Calculate first value per group and ratio
                    group_first = df.groupby(['date_id', 'seconds_in_bucket_group', 'stock_id'])[col].transform('first')
                    df[f'{col}_group_first_ratio'] = np.where(df[col] != 0, group_first / df[col], np.nan)
            
            # Group expanding mean features
            for col in base_features:
                if col in df.columns:
                    # Calculate expanding mean per group
                    expanding_mean = df.groupby(['date_id', 'seconds_in_bucket_group', 'stock_id'])[col].transform(
                        lambda x: x.expanding(min_periods=1).mean()
                    )
                    df[f'{col}_group_expanding_mean_100'] = np.where(df[col] != 0, expanding_mean / df[col], np.nan)
        
        # Seconds in bucket group features
        if 'date_id' in df.columns:
            # Group mean ratio features
            for col in base_features:
                if col in df.columns:
                    # Calculate mean per seconds_in_bucket group
                    group_mean = df.groupby(['date_id', 'seconds_in_bucket'])['seconds_in_bucket_group'].transform('mean')
                    df[f'{col}_seconds_in_bucket_group_mean_ratio'] = np.where(group_mean != 0, group_mean / df['seconds_in_bucket_group'], np.nan)
            
            # Rank features
            for col in base_features:
                if col in df.columns:
                    # Calculate rank within seconds_in_bucket group
                    df[f'{col}_seconds_in_bucket_group_rank'] = df.groupby(['date_id', 'seconds_in_bucket'])[col].rank(
                        method='dense', ascending=False
                    ) / df.groupby(['date_id', 'seconds_in_bucket'])[col].transform('count')

    # --- Auction Edge Features (New Strategic Features) ---
    if 'imbalance_size' in df.columns and 'matched_size' in df.columns:
        # Pressure normalization
        df['imbalance_ratio'] = np.where(df['matched_size'] != 0,
                                        df['imbalance_size'] / df['matched_size'],
                                        np.nan)
        
        # Imbalance momentum
        df['imbalance_slope'] = df['imbalance_size'].diff().rolling(5).mean()
        
        # Late auction imbalance (critical for close)
        df['late_imbalance'] = np.where(df['seconds_in_bucket'] > 540, df['imbalance_size'], 0)
    
    if 'reference_price' in df.columns and 'wap' in df.columns:
        # Divergence signal
        df['price_diff'] = df['reference_price'] - df['wap']
        df['price_diff_ratio'] = df['price_diff'] / df['reference_price']
    
    if all(col in df.columns for col in ['imbalance_buy_sell_flag', 'imbalance_size', 'ask_size', 'bid_size']):
        # Directional pressure with size normalization
        total_size = df['ask_size'] + df['bid_size']
        df['pressure_skew'] = np.where(total_size != 0,
                                      df['imbalance_buy_sell_flag'] * df['imbalance_size'] / total_size,
                                      np.nan)
    
    if 'wap' in df.columns:
        # Local volatility for instability detection
        df['rolling_volatility'] = df['wap'].rolling(20).std()
        df['volatility_ratio'] = df['rolling_volatility'] / df['wap']
    
    # --- Time-Based Auction Features ---
    if 'seconds_in_bucket' in df.columns:
        # Critical time flags
        df['is_final_30s'] = (df['seconds_in_bucket'] >= 570).astype(int)
        df['is_pre_auction'] = (df['seconds_in_bucket'] >= 540).astype(int)
        df['is_early_auction'] = (df['seconds_in_bucket'] < 300).astype(int)
        
        # Time-weighted features
        if 'imbalance_size' in df.columns:
            df['time_weighted_imbalance'] = df['imbalance_size'] * (df['seconds_in_bucket'] / 600)
        
        if 'wap' in df.columns:
            df['time_weighted_price'] = df['wap'] * (df['seconds_in_bucket'] / 600)
    
    # --- Directional Features for Classification ---
    if 'target' in df.columns:
        # Directional classification target
        df['target_direction'] = (df['target'] > 0).astype(int)
        df['target_magnitude'] = np.abs(df['target'])
        
        # Tail identification
        target_quantiles = df.groupby(['date_id', 'stock_id'])['target'].transform('quantile', 0.95)
        df['is_tail_move'] = (df['target'] > target_quantiles).astype(int)
        
        # Directional imbalance alignment
        if 'imbalance_buy_sell_flag' in df.columns:
            df['imbalance_target_alignment'] = (df['imbalance_buy_sell_flag'] * np.sign(df['target']) > 0).astype(int)
    
    # --- Enhanced Pressure Features ---
    if 'imbalance_size' in df.columns and 'volume' in df.columns:
        # Volume-adjusted imbalance
        df['volume_adjusted_imbalance'] = df['imbalance_size'] / df['volume'].rolling(10).mean()
    
    if 'imbalance_size' in df.columns and 'matched_size' in df.columns:
        # Imbalance convergence/divergence
        df['imbalance_convergence'] = df['imbalance_size'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
        )
        
        # Imbalance acceleration
        df['imbalance_acceleration'] = df['imbalance_size'].diff().diff()
    
    # --- Price Action Features ---
    if 'wap' in df.columns:
        # Price momentum
        df['price_momentum'] = df['wap'].diff().rolling(5).mean()
        df['price_acceleration'] = df['wap'].diff().diff()
        
        # Price vs moving average
        df['price_vs_ma'] = df['wap'] / df['wap'].rolling(20).mean()
    
    # --- Auction Microstructure Features ---
    if all(col in df.columns for col in ['bid_price', 'ask_price', 'bid_size', 'ask_size']):
        # Spread analysis
        df['spread_ratio'] = (df['ask_price'] - df['bid_price']) / df['bid_price']
        df['size_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
        
        # Order book pressure
        df['bid_pressure'] = df['bid_size'] / (df['bid_size'] + df['ask_size'])
        df['ask_pressure'] = df['ask_size'] / (df['bid_size'] + df['ask_size'])

    return df 