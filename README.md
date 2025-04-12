# Face Detection System

## Project Overview
This repository contains a face detection system developed for ProCam S.p.A., designed to enhance their new compact digital camera targeted at young photography enthusiasts. The system automatically detects faces in images and returns the coordinates of bounding boxes where faces are located, helping to optimize camera settings during selfies.

## Business Context
ProCam S.p.A. is launching an affordable compact digital camera aimed at young photography enthusiasts. The main product goal is to facilitate the shooting experience, particularly for selfies with one or multiple people.

## Technical Challenge
As a Data Scientist for this project, I developed a face detection system that helps camera technicians automatically optimize camera settings during selfies. The system processes input images, identifies faces, and returns the coordinates of bounding boxes. If no faces are present, it returns an empty list.

## Solution Approach
The project implements two distinct face detection methods:

1. **Viola-Jones Detection**
   - Uses Haar Cascade Classifier from OpenCV
   - Effective for frontal face detection
   - Analyzes intensity differences between adjacent regions
   - Captures facial features like:
     - Eye regions darker than cheeks
     - Nasal bridge lighter than eyes
     - Lips darker than chin

2. **Custom HOG + AdaBoost Detection**
   - HOG (Histogram of Oriented Gradients) for feature extraction
   - AdaBoost classifier with Decision Trees
   - Includes comprehensive training and validation
   - Uses sliding window and scale pyramid for detection

## Project Structure
- **Configuration and Setup**: Global constants and environment setup
- **Data Loading**: Functions to load and preprocess images
- **Image Preprocessing**:
  - Viola-Jones preprocessing: Equalizes and enhances image contrast
  - HOG preprocessing: Uses CLAHE and resizes to target dimensions
- **Detection Algorithms**: Implementation of both detection methods
- **Visualization**: Functions to visualize detection results
- **Training Pipeline**: Complete pipeline for model training and evaluation
- **Testing**: Comparative testing of both methods

## Key Technical Components

### Image Preprocessing
```python
def preprocess_image_viola_jones(image):
    """Preprocessing for Viola-Jones."""
    if image is None:
        return None
    return cv2.equalizeHist(image)

def preprocess_image_hog(image, target_size=TARGET_SIZE):
    """Preprocessing for HOG."""
    if image is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(image)
    return cv2.resize(equalized, target_size)
```

### Face Detection with HOG + AdaBoost
```python
def detect_faces_hog(image, model, threshold=0.7):
    """Detects faces using HOG."""
    if image is None or model is None:
        return []

    height, width = image.shape
    min_size = 50
    scale_factor = 1.2
    stride = 8
    detections = []

    # Scale pyramid implementation
    scales = []
    current_scale = 1.0
    while min(height * current_scale, width * current_scale) >= min_size:
        scales.append(current_scale)
        current_scale /= scale_factor

    # Window sliding detection
    for scale in scales:
        scaled_h = int(height * scale)
        scaled_w = int(width * scale)
        scaled_img = cv2.resize(image, (scaled_w, scaled_h))

        for y in range(0, scaled_h - min_size, stride):
            for x in range(0, scaled_w - min_size, stride):
                window = cv2.resize(scaled_img[y:y + min_size, x:x + min_size], TARGET_SIZE)
                features = compute_hog_features(window)

                prob = model.predict_proba([features])[0][1]
                if prob > threshold:
                    real_x = int(x / scale)
                    real_y = int(y / scale)
                    real_size = int(min_size / scale)
                    detections.append((real_x, real_y, real_size, real_size))

    return non_max_suppression(detections) if detections else []
```

## Model Performance
The custom HOG + AdaBoost model achieved excellent performance in testing:
- 99% accuracy for both face and non-face recognition
- Out of 1400 test images, correctly identified 996 faces and 394 non-faces
- Only 10 total errors (6 false positives and 4 false negatives)

## Requirements
- Python 3.x
- NumPy
- OpenCV (cv2)
- scikit-learn
- matplotlib
- seaborn
- tqdm
- joblib

## Usage
To train and test the face detection system:

```python
# Define dataset path
dataset_dir = os.path.join('face_datasets', '105_classes_pins_dataset')

# Train the model
best_model, best_params, best_score = train_hog_detector(dataset_dir)

# Save the trained model
joblib.dump(best_model, 'hog_face_detector.pkl')

# Load a model and test on new images
model = joblib.load('hog_face_detector.pkl')
image = load_image('test_image.jpg')
faces = detect_faces_hog(image, model)
visualize_detections(image, faces, "HOG + AdaBoost")
```

## Conclusions
This project successfully implements a face detection system using scikit-learn, building a model from scratch based on HOG for feature extraction and AdaBoost for classification. The custom model demonstrates robust performance, effectively detecting faces in test images.

For comparison, the project also experimented with the pre-trained Viola-Jones model available in OpenCV. While more precise in locating face coordinates, the pre-trained model sometimes makes errors in classifying whether an object is a face, whereas the custom-trained model performs better in this aspect.

Test results show that both approaches work effectively for the selfie use case, meeting the requirements of ProCam S.p.A.
