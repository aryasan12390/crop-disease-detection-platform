from django.urls import path
from .views import predict_crop_disease

urlpatterns = [
    path("predict/", predict_crop_disease, name="predict_crop_disease"),
]