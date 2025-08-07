from django.urls import path
from .views import recommend_images

urlpatterns = [
    path('recommend/', recommend_images),
]
