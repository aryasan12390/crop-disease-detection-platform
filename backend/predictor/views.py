import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from django.conf import settings
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from functools import lru_cache
import os

IMG_SIZE = 224

@lru_cache(maxsize=1)
def get_model_and_classes():
    """Load model and class mapping only once (cached)."""
    model_path = os.path.join(settings.BASE_DIR, "crop_disease_model (2).h5")
    class_index_path = os.path.join(settings.BASE_DIR, "class_indices.json")

    model = load_model(model_path)

    with open(class_index_path, "r") as f:
        class_indices = json.load(f)

    index_to_class = {v: k for k, v in class_indices.items()}

    return model, index_to_class


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def predict_crop_disease(request):
    """
    Takes an uploaded image and predicts crop disease.
    """
    if "file" not in request.FILES:
        return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

    img = Image.open(request.FILES["file"]).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

    model, index_to_class = get_model_and_classes()

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = index_to_class[predicted_class_index]
    confidence = float(np.max(predictions[0]))

    return Response({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    })