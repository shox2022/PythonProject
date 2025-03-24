# src/train.py

from src import data_preprocessing as dp
from src.model import build_model  # Ensure build_model exists in model.py

# Define column names for the NSL-KDD dataset (41 features + label)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# File paths for the NSL-KDD dataset files
train_path = r"C:\Users\User\PycharmProjects\PythonProject\data\KDDTrain+.txt"
test_path = r"C:\Users\User\PycharmProjects\PythonProject\data\KDDTest+.txt"


def main():
    # Load the raw datasets
    train_data, test_data = dp.load_data(train_path, test_path, columns)

    # Preprocess the datasets (encode categorical variables and scale numeric ones)
    train_data, test_data = dp.encode_and_preprocess(train_data, test_data)

    # Split into features and labels
    X_train, X_test, y_train, y_test = dp.split_data(train_data, test_data)

    # Build the model using the number of features in X_train
    model = build_model(X_train.shape[1])

    # Train the model (adjust epochs and batch_size as needed)
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.3f}")


if __name__ == '__main__':
    main()
