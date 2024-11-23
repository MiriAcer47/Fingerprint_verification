import os
import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from .funcs.ekstrakcja import preprocess, create_full_white_mask, align_images, thinning, minutiae_extraction, filter_false_minutiae_by_type, save_minutiae


from django.conf import settings
import pprint

def image_processing_view(request):
    return render(request, "image_processing.html")


# Widok do przetwarzania obrazu z assets/9_3.tif
def process_image(request):
    # Ścieżka do folderu assets
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Folder główny Biometria
    assets_folder = os.path.join(base_dir, "assets")
    image_path = os.path.join(assets_folder, "9_3.tif")

    # Sprawdź, czy plik istnieje
    if not os.path.exists(image_path):
        return JsonResponse({"error": f"Plik {image_path} nie istnieje."}, status=404)

    # Wczytaj obraz
    img = cv2.imread(image_path)
    if img is None:
        return JsonResponse({"error": "Nie udało się wczytać obrazu."}, status=400)

    # Przetwarzanie obrazu
    preprocessed = preprocess(img)
    mask = create_full_white_mask(img)
    aligned, aligned_mask = align_images(preprocessed, preprocessed, mask)
    thinned = thinning(aligned)
    minutiae = minutiae_extraction(thinned, aligned_mask)
    filtered_minutiae = filter_false_minutiae_by_type(minutiae)

    # Zapis wyników
    minutiae_json_path = os.path.join(assets_folder, "minutiae_9_3.json")
    save_minutiae(filtered_minutiae, minutiae_json_path)

    # Zwróć odpowiedź
    return JsonResponse(
        {
            "message": "Przetwarzanie zakończone.",
            "minutiae_count": len(filtered_minutiae),
            "minutiae_file": minutiae_json_path,
        }
    )
