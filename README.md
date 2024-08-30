import pickle

# Load the model from the .pkl file
try:
    with open('logistic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

