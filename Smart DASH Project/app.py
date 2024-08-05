import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Creating the Flask app instance
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Defining the prediction route to handle POST requests
@app.route('/predict_quality', methods=['POST'])
def predict_quality():
    data = request.json
    buffer_duration = data['bufferDuration']
    underrun_count = data['underrunCount']
    underrun_duration = data['underrunDuration']
    network_speed = data['networkSpeed']

# Preparing the input data for the model
    input_data = np.array([[buffer_duration, underrun_count, underrun_duration, network_speed]])
    input_data = feature_scaler.transform(input_data)
    input_tensor = torch.FloatTensor(input_data)

# Making the prediction using the model
    with torch.no_grad():
        output = quality_prediction_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob_index, predicted = torch.max(output, 1)
        predicted_quality = label_encoder.inverse_transform(predicted.numpy())[0]

# Preparing the probabilities of dictionary
    probabilities_dict = {quality: float(prob) for quality, prob in zip(quality_levels, probabilities[0])}

    return jsonify({
        'predicted_quality': predicted_quality,
        'probabilities': probabilities_dict
    })

# Defining the CNN model class for quality prediction
class QualityPredictionCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(QualityPredictionCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        conv_output = self.relu(self.conv1(input_tensor))
        conv_output = self.pool(conv_output)
        conv_output = self.relu(self.conv2(conv_output))
        conv_output = self.pool(conv_output)
        flattened = conv_output.view(conv_output.size(0), -1)
        fc_output = self.relu(self.fc1(flattened))
        fc_output = self.dropout(fc_output)
        final_output = self.fc2(fc_output)
        return final_output

# Loading the saved model and its components
print("Starting to load model...")
checkpoint = torch.load('quality_prediction_cnn_model.pth')
print("Checkpoint loaded")
quality_prediction_model = QualityPredictionCNN(input_channels=checkpoint['input_channels'], num_classes=checkpoint['num_classes'])
print("Model created")
quality_prediction_model.load_state_dict(checkpoint['model_state_dict'])
print("Model state loaded")
quality_prediction_model.eval()
print("Model set to eval mode")

# Loading additional components which are required for the model
quality_levels = checkpoint['quality_levels']
label_encoder = checkpoint['label_encoder']
feature_scaler = checkpoint['feature_scaler']
print("All components loaded")

print("Imports successful and model defined")

# Starting the Flask app
if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)