# import os
# import numpy as np
# from django.http import JsonResponse
# from django.conf import settings
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics.pairwise import cosine_similarity

# # Path to dataset folder
# DATASET_DIR = os.path.join(settings.BASE_DIR, 'recommender', 'dataset_images')

# # Load model
# model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# # Precompute features for all images in dataset
# image_features = []
# image_names = []

# for img_name in os.listdir(DATASET_DIR):
#     img_path = os.path.join(DATASET_DIR, img_name)
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     features = model.predict(img_array, verbose=0)
#     image_features.append(features.flatten())
#     image_names.append(img_name)

# image_features = np.array(image_features)


# def recommend_images(request):
#     liked_images = request.GET.get('liked', '')
#     liked_images = liked_images.split(',') if liked_images else []

#     if not liked_images:
#         return JsonResponse({'error': 'No liked images provided'}, status=400)

#     # Get features of liked images
#     liked_features = []
#     for img_name in liked_images:
#         if img_name in image_names:
#             idx = image_names.index(img_name)
#             liked_features.append(image_features[idx])

#     if not liked_features:
#         return JsonResponse({'error': 'Liked images not found in dataset'}, status=404)

#     liked_features = np.array(liked_features)

#     # Average liked embeddings
#     avg_feature = np.mean(liked_features, axis=0).reshape(1, -1)  # ✅ FIX: reshape to 2D

#     # Compute cosine similarity with all images
#     similarities = cosine_similarity(avg_feature, image_features)[0]

#     # Sort and get top 5 recommendations
#     similar_indices = np.argsort(similarities)[::-1]
#     recommended = []
#     for idx in similar_indices:
#         if image_names[idx] not in liked_images:
#             recommended.append(image_names[idx])
#         if len(recommended) >= 5:
#             break

#     return JsonResponse({'recommended': recommended})


# import os
# from django.http import JsonResponse
# from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Base paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # Go one level up from core → use dataset_images folder at project root
# DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "dataset_images")

# # Load model
# model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# # Extract features from an image
# def extract_features(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_data = image.img_to_array(img)
#     img_data = np.expand_dims(img_data, axis=0)
#     img_data = preprocess_input(img_data)
#     features = model.predict(img_data)
#     return features.flatten()

# # Precompute features for all dataset images
# features_dict = {}
# all_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# for img_name in all_images:
#     img_path = os.path.join(DATASET_DIR, img_name)
#     try:
#         features_dict[img_name] = extract_features(img_path)
#     except Exception as e:
#         print(f"Error processing {img_name}: {e}")

# def recommend_images(request):
#     liked = request.GET.get("liked")
#     if not liked:
#         return JsonResponse({"error": "No liked images provided"}, status=400)

#     liked_images = liked.split(",")
#     liked_features = []

#     for img_name in liked_images:
#         if img_name in features_dict:
#             liked_features.append(features_dict[img_name])

#     if not liked_features:
#         return JsonResponse({"error": "No valid liked images found"}, status=400)

#     avg_feature = np.mean(liked_features, axis=0)
#     similarities = {}

#     for img_name, feat in features_dict.items():
#         if img_name not in liked_images:
#             sim = cosine_similarity([avg_feature], [feat])[0][0]
#             similarities[img_name] = sim

#     recommended = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
#     recommended_images = [img[0] for img in recommended]

#     return JsonResponse({"recommendations": recommended_images})



import os
from django.http import JsonResponse
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "dataset_images")


model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()


features_dict = {}
all_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_name in all_images:
    img_path = os.path.join(DATASET_DIR, img_name)
    try:
        features_dict[img_name] = extract_features(img_path)
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

def recommend_images(request):
    liked = request.GET.get("liked")
    if not liked:
        return JsonResponse({"error": "No liked images provided"}, status=400)

    liked_images = liked.split(",")
    liked_features = []

    for img_name in liked_images:
        if img_name in features_dict:
            liked_features.append(features_dict[img_name])

    if not liked_features:
        return JsonResponse({"error": "No valid liked images found"}, status=400)

    # Average features for ALL liked images (dynamic number)
    avg_feature = np.mean(liked_features, axis=0)
    similarities = {}

    for img_name, feat in features_dict.items():
        if img_name not in liked_images:
            sim = cosine_similarity([avg_feature], [feat])[0][0]
            similarities[img_name] = sim


# Recommendations based on cosine similarity 
    recommended = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    recommended_images = [img[0] for img in recommended]

    return JsonResponse({
        "liked_count": len(liked_images),
        "recommendations": recommended_images
    })
