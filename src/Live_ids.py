# live_ids.py

import numpy as np
import keras

from scapy.all import sniff
from tensorflow.keras.models import load_model
import time

# Load the pre-trained model (make sure you have saved it as "ids_model.h5")
model = load_model('ids_model.h5')


# Dummy function to extract features from a packet.
# You will need to replace this with your own feature extraction logic.
def extract_features(packet):
    # For example purposes, we extract just a few dummy features.
    # In practice, you might aggregate packets into flows and compute:
    # - Duration, source bytes, destination bytes, protocol, flags, etc.
    # Here we simply use the packet length and a dummy protocol type.
    try:
        packet_length = len(packet)
    except Exception:
        packet_length = 0

    # Dummy: map protocol names to numeric values (tcp=0, udp=1, icmp=2, else=3)
    proto = packet.getlayer(0).name.lower() if packet.getlayer(0) else "unknown"
    if "tcp" in proto:
        proto_num = 0
    elif "udp" in proto:
        proto_num = 1
    elif "icmp" in proto:
        proto_num = 2
    else:
        proto_num = 3

    # Create a dummy feature vector (update with real features)
    features = {
        'packet_length': packet_length,
        'protocol': proto_num,
        # You can add more features here
    }
    return features


# Dummy preprocessing function to convert feature dict to a numeric vector.
# This should match the order and scaling of your training features.
def preprocess_features(feature_dict):
    # For our dummy example, we'll assume the model was trained on two features:
    # packet_length and protocol, in that order.
    # In a real scenario, you might apply scaling, one-hot encoding, etc.
    vector = [feature_dict['packet_length'], feature_dict['protocol']]
    # Convert list to a NumPy array
    return np.array(vector, dtype=np.float32)


# Prediction function: takes a feature vector and returns a prediction.
def predict_attack(features):
    # Reshape features into the format (1, num_features)
    features = features.reshape(1, -1)
    # Run the prediction
    prediction = model.predict(features)
    # For binary classification with sigmoid activation, use 0.5 as threshold
    return prediction[0][0] > 0.5


# Callback function to process each captured packet.
def packet_callback(packet):
    # Extract raw features from the packet
    features_dict = extract_features(packet)

    # Preprocess features to match the model input format
    features_vector = preprocess_features(features_dict)

    # Get the prediction from the neural network
    is_attack = predict_attack(features_vector)

    # Print a summary and the prediction result
    print(f"Packet: {packet.summary()} | Attack Detected: {is_attack}")

def packet_callback(packet):
    features = extract_features(packet)
    if features is not None:
        prediction = model.predict(features.reshape(1, -1))
        if prediction >= 0.5:
            print("⚠️ Possible Attack Detected:", packet.summary())
        else:
            print("✅ Normal Packet:", packet.summary())

# Main function: start sniffing packets on a specified interface.
def main():
    print("Starting live IDS... Press Ctrl+C to stop.")
    # Start sniffing; adjust interface and filter as needed.
    # For example, interface='eth0' (Linux) or 'Wi-Fi' (Windows).
    sniff(prn=packet_callback, store=False)


if __name__ == '__main__':
    main()
