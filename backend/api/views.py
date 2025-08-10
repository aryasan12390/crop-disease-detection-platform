import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from django.conf import settings
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import os
from .models import Prediction

MODEL_PATH = os.path.join(settings.BASE_DIR, "crop_disease_model (2).h5")
CLASS_INDEX_PATH = os.path.join(settings.BASE_DIR, "class_indices.json")

model = load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

IMG_SIZE = 224

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict_crop_disease(request):
    if "file" not in request.FILES:
        return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

    # Save uploaded image to DB model later
    uploaded_file = request.FILES["file"]

    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = index_to_class[predicted_class_index]
    confidence = float(np.max(predictions[0]))

    # Save to DB
    prediction_obj = Prediction.objects.create(
        image=uploaded_file,
        predicted_class=predicted_class,
        confidence=confidence
    )

    return Response({
        "id": prediction_obj.id,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "image_url": prediction_obj.image.url
    })

@api_view(["GET"])
def prediction_history(request):
    predictions = Prediction.objects.all().order_by("-created_at")
    data = [
        {
            "id": p.id,
            "predicted_class": p.predicted_class,
            "confidence": round(p.confidence, 4),
            "image_url": p.image.url,
            "created_at": p.created_at
        }
        for p in predictions
    ]
    return Response(data)
