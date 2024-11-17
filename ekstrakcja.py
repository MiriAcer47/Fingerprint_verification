import math
import cv2
import numpy as np
import json
from scipy.signal import wiener
import os


#Przetwarzanie obrazu

def preprocess(img):
    # konwersja do skali szarości
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalizacja
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    img_norm = clahe.apply(img_gray)

    # filtracja Gaborowa
    img_gabor = gabor_filter(img_norm)
    img_gabor = cv2.normalize(img_gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # filtracja medianowa
    img_filtered = cv2.medianBlur(img_norm, 5)

    # progowanie - konwersja na obraz binarny
    img_thresh = cv2.adaptiveThreshold(img_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, 5)

    # operacje morfologiczne - otwarcie
    kernel = np.ones((3, 3), np.uint8)
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    # tworzenie mozaiki do wyświetlania obrazów
    img_mosaic = create_mosaic([img_gray, img_norm, img_gabor, img_morph], ["Gray", "CLAHE", "Gabor", "Morph"])

    # wyświetlenie mozaiki
    cv2.imshow("Etapy przetwarzania obrazu", img_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_morph

#Filtracja Gaborowa
def gabor_filter(img):
    ksize = 9
    sigma = 2.0
    lambd = 10.0
    gamma = 0.5
    psi = 0

    #orientacje filtrów
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    accum = np.zeros_like(img, dtype=np.float32)

    # filtracja dla różnych orientacji
    for theta in thetas:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(img, cv2.CV_32F, kernel)
        np.maximum(accum, filtered_img, accum)  # Maksymalizowanie wyników

    img_filtered = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
    img_filtered = np.uint8(img_filtered)

    return img_filtered

#tworzenie mozaiki - do wyświetlania obrazów
def create_mosaic(images, titles):
    resized_images = [cv2.resize(img, (300, 300)) for img in images]

    for i, img in enumerate(resized_images):
        cv2.putText(img, titles[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    mosaic = np.hstack(resized_images)
    return mosaic

#cienkowanie obrazu
def thinning(img):
    thinned = cv2.ximgproc.thinning(img)
    return thinned

#obliczanie orientacji minucji
def compute_orientation(img, x, y):
    neighborhood = img[y-1:y+2, x-1:x+2]
    gx = neighborhood[0, 2] + 2*neighborhood[1, 2] + neighborhood[2, 2] - (neighborhood[0, 0] + 2*neighborhood[1, 0] + neighborhood[2, 0])
    gy = neighborhood[2, 0] + 2*neighborhood[2, 1] + neighborhood[2, 2] - (neighborhood[0, 0] + 2*neighborhood[0, 1] + neighborhood[0, 2])
    angle = math.atan2(gy, gx) * 180 / np.pi
    return angle

#tworzenie maski
def create_full_white_mask(img, margin=10):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[0:margin, :] = 0
    mask[h-margin:h, :] = 0
    mask[:, 0:margin] = 0
    mask[:, w-margin:w] = 0
    return mask

#ekstrakcja minucji
def minutiae_extraction(img, mask):
    minutiae = []
    img = img / 255  # Normalizacja obrazu do zakresu [0, 1]
    rows, cols = img.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if img[i][j] == 1 and mask[i][j] == 255:
                neighborhood = img[i-1:i+2, j-1:j+2]
                cn = np.sum(neighborhood) - img[i][j]

                #wykorzystanie algorytmu Crossing Number
                #zakończenie linii
                if cn == 1:
                    #obliczanie orientacji
                    angle = compute_orientation(img, j, i)
                    #zapis minucji jako słownik z współrzędnymi, typem minucji i kątem
                    minutiae.append({'x': j, 'y': i, 'type': 'ending', 'angle': angle})

                #rozwidlenie linii
                elif cn == 3:
                    angle = compute_orientation(img, j, i)
                    minutiae.append({'x': j, 'y': i, 'type': 'bifurcation', 'angle': angle})

    return minutiae


#wyrównanie obrazów
def align_images(img1, img2, mask1=None):
    # Użycie algorytmu SIFT - wykrywanie punktów kluczowych i deskryptorów
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("Nie wykryto wystarczającej liczby punktów kluczowych.")
        return img1, mask1

    # Dopasowanie punktów kluczowych z użyciem BFMatcher i metody kNN
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Zastosowanie testu Lowe'a do sprawdzenia dobrych dopasowań
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # minimalna liczba dobrych dopasowań
    MIN_MATCH_COUNT = 3

    if len(good_matches) < MIN_MATCH_COUNT:
        print(f"Zbyt mało dobrych dopasowań ({len(good_matches)}/{MIN_MATCH_COUNT}). Wyrównanie nie zostanie wykonane.")
        return img1, mask1

    # Wyodrębnienie punktów dopasowań
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if M is None:
        print("Transformacja nie została obliczona. Wyrównanie nie zostanie wykonane.")
        return img1, mask1

    inlier_ratio = np.sum(inliers) / len(inliers)

    MIN_INLIER_RATIO = 0.5

    if inlier_ratio < MIN_INLIER_RATIO:
        print(" Wyrównanie nie zostanie wykonane.")
        return img1, mask1

    h, w = img2.shape

    # Wyrównanie obrazu img1 do obrazu img2 z użyciem transformacji afinicznej
    aligned_img = cv2.warpAffine(img1, M, (w, h))

    # Wyrównanie maski, jeśli została przekazana
    aligned_mask = None
    if mask1 is not None:
        aligned_mask = cv2.warpAffine(mask1, M, (w, h), flags=cv2.INTER_NEAREST)

    return aligned_img, aligned_mask



#zapisywanie minucji
def save_minutiae(minutiae, filename):
    # Zapisuje listę punktów charakterystycznych do pliku JSON
    with open(filename, 'w') as f:
        json.dump(minutiae, f)

#wczytanie minucji
def load_minutiae(filename):
    # Wczytuje listę punktów charakterystycznych z pliku JSON
    if not os.path.exists(filename):
        print(f"Plik {filename} nie istnieje.")
        return []
    with open(filename, 'r') as f:
        minutiae = json.load(f)
    return minutiae



#Filtruje fałszywe minutie na podstawie ich typu i odpowiednich progów odległości.
def filter_false_minutiae_by_type(minutiae, distance_thresholds=None):

    if distance_thresholds is None:
        # Ustawienie domyślnych progów
        distance_thresholds = {'ending': 5, 'bifurcation': 5}

    filtered_minutiae = []
    minutiae_by_type = {}

    # Grupowanie minutii według typu
    for m in minutiae:
        m_type = m['type']
        if m_type not in minutiae_by_type:
            minutiae_by_type[m_type] = []
        minutiae_by_type[m_type].append(m)

    # Filtrowanie dla każdego typu
    for m_type, m_list in minutiae_by_type.items():
        threshold = distance_thresholds.get(m_type, 5)  # Domyślny próg, jeśli typ nie jest określony
        for i, m1 in enumerate(m_list):
            is_false = False
            for j, m2 in enumerate(m_list):
                if i != j:
                    distance = math.sqrt((m1['x'] - m2['x']) ** 2 + (m1['y'] - m2['y']) ** 2)
                    if distance < threshold:
                        is_false = True
                        break
            if not is_false:
                filtered_minutiae.append(m1)

    return filtered_minutiae


#porównanie minucji
#dopasowanie, gdy zgodny typ minucji, odległość między minucjami < 15px, różnica w kącie orientacji < 15st domyślnie
def compare_minutiae(m1, m2, mask, distance_threshold=15, angle_threshold=15):
    matched = 0
    total_minutiae = 0

    for min1 in m1:
        x1, y1 = min1['x'], min1['y']
        if mask[y1, x1] == 255:
            total_minutiae += 1
            for min2 in m2:
                x2, y2 = min2['x'], min2['y']
                if mask[y2, x2] == 255 and min1['type'] == min2['type']:
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    angle_diff = abs(min1['angle'] - min2['angle'])
                    if distance < distance_threshold and angle_diff < angle_threshold:
                        matched += 1
                        break

    if total_minutiae == 0:
        return 0
    score = matched / total_minutiae
    return score

#rysowanie minucji
def draw_minutiae(img, minutiae):
    for m in minutiae:
        color = (255, 0, 0) if m['type'] == 'ending' else (0, 255, 0)
        cv2.circle(img, (m['x'], m['y']), 3, color, -1)
    return img

#przygotowanie obrazów
def prepare_images(images, titles, size=(300, 300)):
    prepared_images = []
    for img, title in zip(images, titles):
        img_resized = cv2.resize(img, size)
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(img_resized, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        prepared_images.append(img_resized)
    return prepared_images

# Wczytanie obrazów
img1 = cv2.imread('assets/101_1.tif')
img2 = cv2.imread('assets/101_3.tif')


if img1 is None or img2 is None:
    print("Błąd wczytywania obrazów.")
    exit()


# Preprocessing obrazów
preprocessed1 = preprocess(img1)
preprocessed2 = preprocess(img2)


# Tworzenie maski przed wyrównaniem
mask1 = create_full_white_mask(img1)
mask2 = create_full_white_mask(img2)


# Wyrównanie obrazów
aligned1, aligned_mask1 = align_images(preprocessed1, preprocessed2, mask1)

# Tworzenie maski wspólnego obszaru
common_area_mask = cv2.bitwise_and(aligned_mask1, mask2)

# Cienkowanie
thinned1 = thinning(aligned1)
thinned2 = thinning(preprocessed2)


# Ekstrakcja punktów charakterystycznych
minutiae1 = filter_false_minutiae_by_type(minutiae_extraction(thinned1, aligned_mask1))
minutiae2 = filter_false_minutiae_by_type(minutiae_extraction(thinned2, mask2))

# Wyświetlanie liczby punktów charakterystycznych
print(f"Liczba punktów charakterystycznych w odcisku 1: {len(minutiae1)}")
print(f"Liczba punktów charakterystycznych w odcisku 2: {len(minutiae2)}")

print("Pierwsze 5 punktów charakterystycznych w odcisku 1:")
for point in minutiae1[:5]:
    print(f"Współrzędne: {point}")

print("Pierwsze 5 punktów charakterystycznych w odcisku 2:")
for point in minutiae2[:5]:
    print(f"Współrzędne: {point}")

# Porównanie odcisków palców z uwzględnieniem wspólnego obszaru
score = compare_minutiae(minutiae1, minutiae2, common_area_mask)

# Wyświetlenie wyniku porównania
print(f"Liczba dopasowanych punktów: {int(score * max(len(minutiae1), len(minutiae2)))}")
print(f"Wskaźnik dopasowania: {score:.2f}")

if score > 0.2:
    print("Odciski palców pasują do siebie.")
else:
    print("Odciski palców nie pasują do siebie.")



# Zapis punktów charakterystycznych do plików JSON
save_minutiae(minutiae1, 'minucje1.json')
save_minutiae(minutiae2, 'minucje2.json')

print("Punkty charakterystyczne zostały zapisane do plików 'minucje1.json' i 'minucje2.json'.")

# Wczytanie punktów charakterystycznych z plików JSON
loaded_minutiae1 = load_minutiae('minucje1.json')
loaded_minutiae2 = load_minutiae('minucje2.json')

# Wyświetlenie liczby punktów charakterystycznych
print(f"Liczba punktów charakterystycznych w odcisku 1: {len(loaded_minutiae1)}")
print(f"Liczba punktów charakterystycznych w odcisku 2: {len(loaded_minutiae2)}")

print(f"Liczba punktów końcowych w odcisku 1: {sum(1 for m in loaded_minutiae1 if m['type'] == 'ending')}")
print(f"Liczba rozwidleń w odcisku 1: {sum(1 for m in loaded_minutiae1 if m['type'] == 'bifurcation')}")
print(f"Liczba punktów końcowych w odcisku 2: {sum(1 for m in loaded_minutiae2 if m['type'] == 'ending')}")
print(f"Liczba rozwidleń w odcisku 2: {sum(1 for m in loaded_minutiae2 if m['type'] == 'bifurcation')}")


# Porównanie odcisków palców z uwzględnieniem wspólnego obszaru
score = compare_minutiae(loaded_minutiae1, loaded_minutiae2, common_area_mask)


# Wyświetlenie wyniku porównania
print(f"Liczba dopasowanych punktów: {int(score * max(len(loaded_minutiae1), len(loaded_minutiae2)))}")
print(f"Wskaźnik dopasowania: {score:.2f}")

if score > 0.2:
    print("Odciski palców pasują do siebie.")
else:
    print("Odciski palców nie pasują do siebie.")

# Rysowanie minucji
display_img1 = cv2.cvtColor(thinned1, cv2.COLOR_GRAY2BGR)
display_img2 = cv2.cvtColor(thinned2, cv2.COLOR_GRAY2BGR)
preprocessed_display1 = cv2.cvtColor(preprocessed1, cv2.COLOR_GRAY2BGR)
preprocessed_display2 = cv2.cvtColor(preprocessed2, cv2.COLOR_GRAY2BGR)
original_display1 = img1.copy()
original_display2 = img2.copy()

display_img1 = draw_minutiae(display_img1, minutiae1)
display_img2 = draw_minutiae(display_img2, minutiae2)
preprocessed_display1 = draw_minutiae(preprocessed_display1, minutiae1)
preprocessed_display2 = draw_minutiae(preprocessed_display2, minutiae2)

images_row1 = [original_display1, preprocessed_display1, display_img1]
titles_row1 = ["ORYGINAL 1", "PREPROCESS 1", "Minucje 1"]

images_row2 = [original_display2, preprocessed_display2, display_img2]
titles_row2 = ["ORYGINAL 2", "PREPROCESS 2", "Minucje 2"]

# Przygotowanie obrazów
prepared_images_row1 = prepare_images(images_row1, titles_row1)
prepared_images_row2 = prepare_images(images_row2, titles_row2)

# Tworzenie mozaiki
row1 = np.hstack(prepared_images_row1)
row2 = np.hstack(prepared_images_row2)
full_mosaic = np.vstack([row1, row2])

# Wyświetlanie mozaiki
cv2.imshow("Porównanie odcisków palców", full_mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()

