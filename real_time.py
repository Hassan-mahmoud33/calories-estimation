from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import os

#----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
model = YOLO(BASE_DIR / "outputs/best.pt")

#----------------------------------------
CAL_PER_100G = {
    'apple': 52, 'avocado': 160, 'banana': 89, 'bread-white': 265,
    'bread-wholemeal': 247, 'broccoli': 34, 'butter': 717, 'carrot': 41,
    'cheese': 300, 'chicken': 165, 'coffee-with-caffeine': 2, 'cucumber': 15,
    'egg': 143, 'espresso-with-caffeine': 2, 'hard-cheese': 400, 'jam': 250,
    'mixed-salad-chopped-without-sauce': 20, 'mixed-vegetables': 60,
    'pasta-spaghetti': 158, 'potatoes-steamed': 87, 'rice': 130,
    'salad-leaf-salad-green': 15, 'sweet-pepper': 31, 'tea': 1, 'tomato': 18,
    'tomato-sauce': 40, 'water': 0, 'white-coffee-with-caffeine': 20,
    'wine-red': 85, 'wine-white': 82,
}

GRAMS_PER_PIXEL = 0.01

#----------------------------------------

def run_video():
    cap = cv2.VideoCapture(BASE_DIR / "video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) / 3
    output = cv2.VideoWriter(
        BASE_DIR / "output.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (900, 620),
    )

    skip = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        skip += 1
        if skip % 3 != 0:
            continue

        frame = cv2.resize(frame, (900, 620))
        results = model(frame)
        r = results[0]
        annotated = r.plot()

        total_kcalories = 0
        if r.masks is not None:
            for i, poly in enumerate(r.masks.xy):
                if len(poly) < 3:
                    continue

                cls_id = int(r.boxes.cls[i])
                cls_name = model.names[cls_id]
                area_px = cv2.contourArea(poly.astype(np.int32))
                grams_divided100 = (area_px * GRAMS_PER_PIXEL) / 100
                calories = int(grams_divided100 * CAL_PER_100G[cls_name])
                total_kcalories += calories

            cv2.putText(
                annotated,
                f"Total Calories: {total_kcalories} Calories",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                4,
            )

        output.write(annotated)
        cv2.imshow("frame", annotated)

        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video()