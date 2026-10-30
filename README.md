# Food Calorie Estimation

A small computer vision project that uses a YOLO segmentation model to identify food items in an image and estimate their calories based on the detected area.

## How It Works

1. The model detects food items and outlines each one.
2. The area of each detected mask is measured in pixels.
3. The pixel area is converted to an estimated weight using a fixed factor. Calories are then calculated using the stored nutritional values per 100 grams.

This is an experimental estimate, not a measurement of food weight or calories. Results are for demonstration and learning, not for dietary or medical use. See [Limitations](#limitations).

## Dataset

Trained on the "food" instance-segmentation dataset (30 classes) from Roboflow Universe, licensed CC BY 4.0:
https://universe.roboflow.com/hust-ajpvu/food-srnub/dataset/1

Split: 6404 train, 1830 validation, 916 test images.

## Model and Results

YOLO26n-seg, fine-tuned from the COCO-pretrained `yolo26n-seg.pt` weights on a Tesla T4 (Google Colab): 60 epochs, image size 640, best weights from epoch 50.

Evaluated on the held-out test split (916 images, 1174 instances):

| | mAP50 | mAP50-95 |
|---|---|---|
| Box | 60.4% | 45.2% |
| Mask | 56.9% | 39.1% |

Strongest classes (mask mAP50): wine-red, banana, apple, water. Weakest: cheese, sweet-pepper, mixed-vegetables, avocado, all at or below about 23%.

## Requirements

* Python 3.9 or later
* The packages listed in [requirements.txt](https://github.com/Hassan-mahmoud33/calories-estimation/blob/main/requirements.txt)
* The model file `outputs/best.pt`, included in the project

## Run the App

From the project directory, install the dependencies:

```
python -m pip install -r requirements.txt
```

Start the Streamlit app:

```
streamlit run streamlit_app.py
```

Upload a JPG, JPEG, or PNG image, then click Detect to view the segmentation result and estimated total calories.

## Process a Video

[real_time.py](https://github.com/Hassan-mahmoud33/calories-estimation/blob/main/real_time.py) reads `video.mp4`, displays the analyzed frames, and saves the result as `output.mp4`. To run it separately:

```
python real_time.py
```

## Train the Model

[train_model_colab.ipynb](https://github.com/Hassan-mahmoud33/calories-estimation/blob/main/train_model_colab.ipynb) contains the steps for downloading the dataset from Roboflow and training the segmentation model. The notebook is set up for Google Colab. Add `ROBOFLOW_API_KEY` to Colab Secrets before downloading the dataset. After training, the generated weights can replace `outputs/best.pt`.

## Main Files

* `streamlit_app.py`: image upload interface and results.
* `real_time.py`: model loading, video processing, calorie values, and the weight-estimation factor.
* `outputs/best.pt`: model weights used by the app.
* `train_model_colab.ipynb`: Colab notebook for model training.

## Limitations

* Calories are estimated from mask area in pixels times a fixed factor (`GRAMS_PER_PIXEL = 0.01`), which has not been calibrated against measured food weights. The calorie numbers are demonstration values, not measurements.
* Images are resized to 900x620 before detection, which distorts the aspect ratio and changes the pixel areas.
* For liquids (coffee, tea, wine, water), pixel area does not reflect volume or weight.
* Several classes are detected unreliably (see the weakest classes above), so the total can be wrong when items are missed or mislabeled.
* The detection confidence threshold is 0.25 in the app and the video script. A lower threshold adds false detections and inflates the total.
* `real_time.py` processes a video file (`video.mp4`), not a live camera, and analyzes every third frame.