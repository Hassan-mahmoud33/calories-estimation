# Food Calorie Estimation

A small computer vision project that uses a YOLO segmentation model to identify food items in an image and estimate their calories based on the detected area.

## How It Works

1. The model detects food items and outlines each one.
2. The area of each detected mask is measured in pixels.
3. The pixel area is converted to an estimated weight using a fixed factor. Calories are then calculated using the stored nutritional values per 100 grams.

This is an experimental estimate, not an actual measurement of food weight or calories. The estimated food size and calorie count are **not accurate**: the fixed pixel-to-weight factor does not account for real-world dimensions, camera angle, or food preparation. Results are intended for demonstration and learning, not for dietary or medical use. The model and size-estimation method can be improved in the future with better data and proper calibration.

## Requirements

- Python 3.9 or later
- The packages listed in [requirements.txt](./requirements.txt)
- The model file `outputs/best.pt`, included in the project

## Run the App

From the project directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Upload a JPG, JPEG, or PNG image, then click **Detect** to view the segmentation result and estimated total calories.

## Process a Video

[real_time.py](./real_time.py) reads `video.mp4`, displays the analyzed frames, and saves the result as `output.mp4`. To run it separately:

```bash
python real_time.py
```

## Train the Model

[train_model_colab.ipynb](./train_model_colab.ipynb) contains the steps for downloading the dataset from Roboflow and training the segmentation model. The notebook is set up for Google Colab. Add `ROBOFLOW_API_KEY` to Colab Secrets before downloading the dataset. After training, the generated weights can replace `outputs/best.pt`.

## Main Files

- `streamlit_app.py`: image upload interface and results.
- `real_time.py`: model loading, video processing, calorie values, and the weight-estimation factor.
- `outputs/best.pt`: model weights used by the app.
- `train_model_colab.ipynb`: Colab notebook for model training.
