import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(train_path, test_path, columns):
    train_data = pd.read_csv(train_path, names=columns)
    test_data = pd.read_csv(test_path, names=columns)
    return train_data, test_data


def encode_and_preprocess(train_data, test_data):
    # Define categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']

    # Remove any extra whitespace from column names
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    # Compute the originally numeric columns (exclude categorical and label columns)
    original_numeric_cols = train_data.drop(columns=categorical_cols + ['label']).columns.tolist()

    # Force conversion of originally numeric columns to numeric types
    for col in original_numeric_cols:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

    # Initialize the OneHotEncoder (using sparse_output to save memory)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # One-hot encode the categorical columns
    train_encoded = encoder.fit_transform(train_data[categorical_cols])
    test_encoded = encoder.transform(test_data[categorical_cols])

    # Convert the encoded matrices to DataFrames (still sparse)
    train_encoded_df = pd.DataFrame.sparse.from_spmatrix(
        train_encoded, columns=encoder.get_feature_names_out(categorical_cols)
    )
    test_encoded_df = pd.DataFrame.sparse.from_spmatrix(
        test_encoded, columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Drop the original categorical columns from the raw data and reset index
    train_data = train_data.drop(columns=categorical_cols).reset_index(drop=True)
    test_data = test_data.drop(columns=categorical_cols).reset_index(drop=True)

    # Concatenate the numeric data with the encoded categorical data
    train_data = pd.concat([train_data, train_encoded_df], axis=1)
    test_data = pd.concat([test_data, test_encoded_df], axis=1)

    # Convert the label column: 'normal' becomes 0, any attack becomes 1
    train_data['label'] = train_data['label'].eq('normal').astype(int)
    test_data['label'] = test_data['label'].eq('normal').astype(int)

    # Scale only the originally numeric columns (which are now guaranteed to be numeric)
    scaler = StandardScaler()
    train_data.loc[:, original_numeric_cols] = scaler.fit_transform(train_data[original_numeric_cols])
    test_data.loc[:, original_numeric_cols] = scaler.transform(test_data[original_numeric_cols])

    return train_data, test_data


def get_features_and_labels(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X, y


def split_data(train_data, test_data):
    X_train, y_train = get_features_and_labels(train_data)
    X_test, y_test = get_features_and_labels(test_data)
    return X_train, X_test, y_train, y_test
