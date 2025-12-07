from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
from PIL import Image
import os
import json
from io import BytesIO
from typing import Dict, Any

# ---------------------------
# CONFIG (Updated for Local Deployment)
# ---------------------------
# NOTE: You MUST create a folder named 'model_assets' in the same directory as this file
# and copy brain_tumor_model_best.pth and label_map.json into it.
MODEL_ASSETS_DIR = "model_assets"
MODEL_SAVE_PATH = os.path.join(MODEL_ASSETS_DIR, "brain_tumor_model_best.pth")
LABELS_SAVE = os.path.join(MODEL_ASSETS_DIR, "label_map.json")
INPUT_SIZE = 224
DEVICE = torch.device("cpu")  # Use CPU for server inference unless a GPU is configured


# ---------------------------
# Model and Helpers (Copied from training script)
# ---------------------------
def create_model(num_classes: int) -> nn.Module:
    """Initializes the model architecture based on the training logic."""
    try:
        # Try EfficientNet_B0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        print("Using EfficientNet-B0 architecture.")
    except Exception:
        # Fallback to ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print("Using ResNet50 architecture.")
    return model


def get_inference_transforms() -> T.Compose:
    """Returns the transformation pipeline for inference."""
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------
# FastAPI Initialization and Model Loading
# ---------------------------
app = FastAPI(title="Brain Tumor Classification API")

# Add CORS middleware to allow the React frontend (running on a different port) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the model and class mapping
model: nn.Module = None
idx_to_class: Dict[int, str] = {}
inference_transform = get_inference_transforms()


@app.on_event("startup")
async def load_resources():
    """Loads the model and class map on application startup."""
    global model, idx_to_class

    # 1. Load Class Map
    try:
        with open(LABELS_SAVE, "r") as f:
            # The keys are saved as strings in the JSON, convert them back to integers
            raw_map = json.load(f)
            idx_to_class = {int(k): v for k, v in raw_map.items()}
            num_classes = len(idx_to_class)
    except FileNotFoundError:
        # Improved error message for local deployment
        raise HTTPException(status_code=500,
                            detail=f"Label map not found at {LABELS_SAVE}. Ensure you have created the '{MODEL_ASSETS_DIR}' folder and copied 'label_map.json' into it.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading label map: {e}")

    # 2. Initialize and Load Model
    try:
        # Load the saved state dictionary
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)

        # Initialize the correct model architecture
        model = create_model(num_classes).to(DEVICE)

        # Load the saved weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("Model loaded successfully and set to evaluation mode.")
    except FileNotFoundError:
        # Improved error message for local deployment
        raise HTTPException(status_code=500,
                            detail=f"Model file not found at {MODEL_SAVE_PATH}. Ensure you have copied 'brain_tumor_model_best.pth' into the '{MODEL_ASSETS_DIR}' folder.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")


@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"message": "FastAPI Brain Tumor Classifier is running."}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Receives an image, runs inference, and returns prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Open image using PIL
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Apply transformations
        tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_index = torch.max(probabilities, 0)

        # Get results
        predicted_class = idx_to_class.get(predicted_index.item(), "Unknown")
        confidence_percent = confidence.item() * 100

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence_percent:.2f}%",
            "confidence_value": confidence.item()
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to server error: {e}")

# To run this file, you typically use: uvicorn api:app --reload
# or if running in a notebook: !uvicorn api:app --reload --port 8000 &