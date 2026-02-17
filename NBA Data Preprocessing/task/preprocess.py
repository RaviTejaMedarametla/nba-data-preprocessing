import numpy as np
import pandas as pd


def _ensure_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data)


def _clean_dataframe(df):
    cleaned = df.copy()

    cleaned['b_day'] = pd.to_datetime(cleaned['b_day'], format='%m/%d/%y')
    cleaned['draft_year'] = pd.to_datetime(cleaned['draft_year'], format='%Y')

    cleaned['team'] = cleaned['team'].fillna('No Team')

    cleaned['height'] = cleaned['height'].str.split(' / ').str[1].astype(float)
    cleaned['weight'] = (
        cleaned['weight']
        .str.split(' / ').str[1]
        .str.replace(' kg.', '', regex=False)
        .astype(float)
    )
    cleaned['salary'] = cleaned['salary'].str.replace('$', '', regex=False).astype(float)

    cleaned['country'] = np.where(cleaned['country'] == 'USA', 'USA', 'Not-USA')
    cleaned['draft_round'] = cleaned['draft_round'].replace('Undrafted', '0')

    return cleaned


def clean_data(path):
    df = _ensure_dataframe(path)
    return _clean_dataframe(df)


def feature_data(df):
    featured = df.copy()

    year = featured['version'].str.extract(r'(\d+)$')[0].astype(int)
    year = np.where(year < 100, year + 2000, year)
    version_year = pd.to_datetime(year.astype(str), format='%Y').year

    featured['age'] = version_year - featured['b_day'].dt.year
    featured['experience'] = version_year - featured['draft_year'].dt.year
    featured['bmi'] = featured['weight'] / (featured['height'] ** 2)

    featured = featured.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'])

    for column in list(featured.select_dtypes(include='object').columns):
        if featured[column].nunique(dropna=False) >= 50:
            featured = featured.drop(columns=column)

    return featured


def multicol_data(df):
    transformed = df.copy()
    threshold = 0.5

    numeric_columns = [column for column in transformed.select_dtypes(include='number').columns if column != 'salary']

    while len(numeric_columns) > 1:
        corr_matrix = transformed[numeric_columns].corr().fillna(0.0)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        candidate_pairs = upper_triangle.stack()
        if candidate_pairs.empty:
            break

        max_corr = candidate_pairs.abs().max()
        if max_corr <= threshold:
            break

        first_column, second_column = candidate_pairs.abs().idxmax()

        target_corr = transformed[numeric_columns].corrwith(transformed['salary']).abs().fillna(0.0)
        first_target_corr = target_corr[first_column]
        second_target_corr = target_corr[second_column]

        drop_column = first_column if first_target_corr < second_target_corr else second_column
        transformed = transformed.drop(columns=drop_column)
        numeric_columns.remove(drop_column)

    return transformed


def transform_data(df):
    y = df['salary'].copy()
    x = df.drop(columns='salary')

    num_columns = list(x.select_dtypes(include='number').columns)
    num_feat_df = x[num_columns].astype(float)
    num_feat_df = num_feat_df.fillna(num_feat_df.median())

    means = num_feat_df.mean()
    stds = num_feat_df.std(ddof=0).replace(0.0, 1.0)
    num_scaled = (num_feat_df - means) / stds

    cat_columns = list(x.select_dtypes(include='object').columns)
    cat_frames = []
    for column in cat_columns:
        filled_column = x[column].fillna(f'Unknown_{column}')
        categories = pd.Index(pd.unique(filled_column))
        encoded = pd.DataFrame(
            (filled_column.to_numpy()[:, None] == categories.to_numpy()).astype(int),
            columns=categories.astype(str),
            index=x.index,
        )
        cat_frames.append(encoded)

    cat_feat_df = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=x.index)

    X = pd.concat([num_scaled, cat_feat_df], axis=1)
    return X, y


class NBAPreprocessingPipeline:
    def __init__(self):
        self._is_fitted = False

    def fit(self, data):
        df = _ensure_dataframe(data)
        cleaned = _clean_dataframe(df)
        featured = feature_data(cleaned)
        filtered = multicol_data(featured)

        x = filtered.drop(columns='salary') if 'salary' in filtered.columns else filtered.copy()

        self.feature_columns_ = [column for column in filtered.columns if column != 'salary']

        self.num_columns_ = list(x.select_dtypes(include='number').columns)
        num_data = x[self.num_columns_].astype(float)
        self.num_medians_ = num_data.median()
        num_data = num_data.fillna(self.num_medians_)
        self.num_means_ = num_data.mean()
        self.num_stds_ = num_data.std(ddof=0).replace(0.0, 1.0)

        self.cat_columns_ = list(x.select_dtypes(include='object').columns)
        self.cat_categories_ = {}
        for column in self.cat_columns_:
            filled = x[column].fillna(f'Unknown_{column}')
            self.cat_categories_[column] = pd.Index(pd.unique(filled))

        self._is_fitted = True
        return self

    def transform(self, data):
        if not self._is_fitted:
            raise ValueError('Pipeline is not fitted. Call fit() before transform().')

        df = _ensure_dataframe(data)
        cleaned = _clean_dataframe(df)
        featured = feature_data(cleaned)

        x = featured.reindex(columns=self.feature_columns_, fill_value=np.nan)

        num_data = x[self.num_columns_].astype(float).fillna(self.num_medians_)
        num_scaled = (num_data - self.num_means_) / self.num_stds_

        cat_frames = []
        for column in self.cat_columns_:
            filled = x[column].fillna(f'Unknown_{column}')
            categories = self.cat_categories_[column]
            encoded = pd.DataFrame(
                (filled.to_numpy()[:, None] == categories.to_numpy()).astype(int),
                columns=categories.astype(str),
                index=x.index,
            )
            cat_frames.append(encoded)

        cat_data = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=x.index)
        return pd.concat([num_scaled, cat_data], axis=1)

    def fit_transform(self, data):
        self.fit(data)
        df = _ensure_dataframe(data)
        y = _clean_dataframe(df)['salary'].copy() if 'salary' in df.columns else pd.Series(index=df.index, dtype=float)
        return self.transform(df), y


def create_preprocessing_pipeline():
    return NBAPreprocessingPipeline()


def _fit_linear_regression(x_train, y_train):
    x_matrix = np.column_stack([np.ones(len(x_train)), x_train.to_numpy(dtype=float)])
    y_vector = y_train.to_numpy(dtype=float)
    coefficients = np.linalg.pinv(x_matrix) @ y_vector
    return coefficients


def _predict_linear_regression(x_data, coefficients):
    x_matrix = np.column_stack([np.ones(len(x_data)), x_data.to_numpy(dtype=float)])
    return x_matrix @ coefficients


def train_linear_regression_model(X, y):
    coefficients = _fit_linear_regression(X, y)
    return {
        'intercept': float(coefficients[0]),
        'coefficients': pd.Series(coefficients[1:], index=X.columns, dtype=float),
    }


def evaluate_linear_regression_cv(X, y, n_splits=5, random_state=42):
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')
    if len(X) < n_splits:
        raise ValueError('n_splits cannot exceed number of samples')

    indices = np.arange(len(X))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    fold_indices = np.array_split(indices, n_splits)
    fold_metrics = []

    for fold_id in range(n_splits):
        test_idx = fold_indices[fold_id]
        train_idx = np.concatenate([fold_indices[i] for i in range(n_splits) if i != fold_id])

        x_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        coefficients = _fit_linear_regression(x_train, y_train)
        predictions = _predict_linear_regression(x_test, coefficients)

        residuals = y_test.to_numpy(dtype=float) - predictions
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residuals)))

        y_true = y_test.to_numpy(dtype=float)
        denominator = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residuals ** 2)) / denominator if denominator != 0 else 0.0

        fold_metrics.append({'fold': fold_id + 1, 'rmse': rmse, 'mae': mae, 'r2': r2})

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        'rmse_mean': float(metrics_df['rmse'].mean()),
        'rmse_std': float(metrics_df['rmse'].std(ddof=0)),
        'mae_mean': float(metrics_df['mae'].mean()),
        'mae_std': float(metrics_df['mae'].std(ddof=0)),
        'r2_mean': float(metrics_df['r2'].mean()),
        'r2_std': float(metrics_df['r2'].std(ddof=0)),
        'fold_metrics': metrics_df,
    }
    return summary
