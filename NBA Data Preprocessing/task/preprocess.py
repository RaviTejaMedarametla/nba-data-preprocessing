import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_data(path):
    df = pd.read_csv(path)

    print(f"Initial DataFrame shape: {df.shape}")
    print("NaN values per column before cleaning:")
    print(df.isna().sum())

    # Fill NaN values for categorical columns
    df['team'] = df['team'].fillna("Unknown")
    df['college'] = df['college'].fillna("Unknown")

    # Drop unnecessary columns and rows with NaNs in critical columns
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')  # Convert salary to numeric
    df.dropna(subset=['salary', 'rating'], inplace=True)  # Drop rows where salary or rating is NaN

    print(f"Cleaned DataFrame shape: {df.shape}")
    print("Columns in the cleaned DataFrame:", df.columns.tolist())
    return df


def feature_data(df):
    # Convert height and weight to numeric, coercing errors
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    # Handle NaN values in height and weight after conversion
    df.dropna(subset=['height', 'weight'], inplace=True)

    # Calculate BMI
    df['bmi'] = df['weight'] / ((df['height'] / 39.37) ** 2)  # Convert height to meters

    # Add experience based on draft_year
    df['year'] = datetime.now().year
    df['experience'] = df['year'] - df['draft_year']

    # Ensure that numeric columns are correctly typed as numeric
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['experience'] = pd.to_numeric(df['experience'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Print the data types for debugging
    print("Data types before returning from feature_data:")
    print(df.dtypes)

    # Drop rows where bmi, experience, or rating are NaN
    df.dropna(subset=['bmi', 'experience', 'rating'], inplace=True)

    # Select relevant columns
    df = df[['bmi', 'experience', 'salary', 'rating', 'team', 'position', 'college']]

    print("Columns before returning from feature_data:", df.columns.tolist())
    print(f"DataFrame after feature engineering (first few rows):\n{df.head()}")

    return df


def transform_data(df):
    # Select features and target variable
    x = df.drop('salary', axis=1)
    y = df['salary']

    # Check for NaNs in numerical columns
    print("NaN values in numerical columns:")
    print(df[['bmi', 'experience', 'rating']].isna().sum())

    # Handle NaNs by either filling or dropping (choose based on your preference)
    x['bmi'] = x['bmi'].fillna(x['bmi'].mean())
    x['experience'] = x['experience'].fillna(x['experience'].mean())
    x['rating'] = x['rating'].fillna(x['rating'].mean())

    # Print the DataFrame before scaling for clarity
    print(f"DataFrame before scaling (first few rows):\n{x.head()}")

    # Transform numerical features
    num_feat_df = x[['bmi', 'experience', 'rating']]

    # Print numerical features DataFrame for debugging
    print(f"Numerical features before scaling: {num_feat_df.head()}")
    print(f"Numerical features columns: {num_feat_df.columns.tolist()}")

    if num_feat_df.empty:
        print("No numerical features found. Ensure correct data types.")
        return None, None

    cat_feat_df = x.select_dtypes(include=['object'])  # categorical features

    # Standardizing numerical features
    scaler = StandardScaler()
    num_feat_scaled = scaler.fit_transform(num_feat_df)

    # One-hot encoding categorical features
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    cat_feat_encoded = encoder.fit_transform(cat_feat_df)

    # Create DataFrame for numerical features
    num_feat_scaled_df = pd.DataFrame(num_feat_scaled, columns=num_feat_df.columns)

    # Create DataFrame for categorical features
    cat_feat_encoded_df = pd.DataFrame(cat_feat_encoded, columns=encoder.get_feature_names_out(cat_feat_df.columns))

    # Concatenate the scaled numerical features and encoded categorical features
    X = pd.concat([num_feat_scaled_df, cat_feat_encoded_df], axis=1)

    print(f"Unique feature names in X: {X.columns.tolist()}")
    print(f"Shape of features (X): {X.shape}")
    print(f"Shape of target (y): {y.shape}")

    return X, y


if __name__ == "__main__":
    path = "C:\\Users\\ravit\\PycharmProjects\\NBA Data Preprocessing\\NBA Data Preprocessing\\data\\nba2k-full.csv"
    df_cleaned = clean_data(path)
    df_featured = feature_data(df_cleaned)
    X, y = transform_data(df_featured)

    # Print final shapes and columns
    if X is not None:
        print(f"Final X shape: {X.shape}, y shape: {y.shape}")
