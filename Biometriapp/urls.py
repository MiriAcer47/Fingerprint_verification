from django.urls import path
from .views import image_processing_view, process_image

urlpatterns = [
    path("", image_processing_view, name="image_processing"),
    path("process_image/", process_image, name="process_image"),
]
