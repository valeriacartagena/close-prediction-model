import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_time_aware_split(self, df, date_col='date_id', n_splits=5):
        n_groups = df[date_col].nunique()
        n_splits = min(n_splits, n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(df, groups=df[date_col]):
            yield train_idx, val_idx

    def get_feature_columns(self, df, target_col='target', exclude=None):
        exclude = exclude or []
        return [col for col in df.columns if col not in exclude + [target_col] and df[col].dtype != 'object']

    def handle_missing(self, df):
        # Fill numeric columns with mean, non-numeric with 0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(0)
        return df

    def encode_categoricals(self, df, exclude=None):
        exclude = exclude or []
        # Exclude row_id and other specified columns from one-hot encoding
        categorical_cols = df.select_dtypes(include='object').columns
        categorical_cols = [col for col in categorical_cols if col not in exclude]
        
        if len(categorical_cols) > 0:
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        else:
            df_encoded = df.copy()
        
        return df_encoded

    def scale(self, train_df, val_df=None, test_df=None):
        numeric_cols = train_df.select_dtypes(include='number').columns
        train_df[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])
        if val_df is not None:
            val_df[numeric_cols] = self.scaler.transform(val_df[numeric_cols])
        if test_df is not None:
            test_df[numeric_cols] = self.scaler.transform(test_df[numeric_cols])
        return train_df, val_df, test_df

    def validate_no_leakage(self, df, date_col='date_id', time_col='seconds_in_bucket', target_col='target'):
        assert df.groupby(date_col)[time_col].max().nunique() == 1, "Inconsistent time ranges"
        assert not df[target_col].isnull().any(), "Missing targets"

    def prepare(self, df, target_col='target', exclude=None):
        exclude = exclude or []
        df = self.handle_missing(df)
        df = self.encode_categoricals(df, exclude=exclude)
        features = self.get_feature_columns(df, target_col, exclude)
        return df, features 