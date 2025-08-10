from django.urls import path
from . import views

urlpatterns = [
    path("predict/", views.predict_crop_disease, name="predict"),
    path("predictions/", views.prediction_history, name="prediction_history"),
]
