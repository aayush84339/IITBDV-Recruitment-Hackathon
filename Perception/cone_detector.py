"""
Cone Distance Estimation — YOLO + Pinhole Camera Model
Formula: d = (f * H) / h
  f = 1000 (focal length in pixels/mm),  H = 30 cm (cone height),  h = bbox height in px
  d = (1000 * 30) / h_pixels  [cm]  →  /100 for metres

Usage: python cone_distance_estimator.py --image test.jpg --model best.pt
"""

import cv2, numpy as np, argparse, sys, os
from ultralytics import YOLO

# Constants
F = 1000.0   # focal length (as given in problem statement)
H = 30.0     # real cone height in cm
CONF = 0.25

def main(image_path, model_path, output_path):
    if not os.path.isfile(model_path):
        sys.exit(f"ERROR: Model '{model_path}' not found!")
    if not os.path.isfile(image_path):
        sys.exit(f"ERROR: Image '{image_path}' not found!")

    model = YOLO(model_path)
    img = cv2.imread(image_path)
    results = model.predict(source=image_path, conf=CONF, iou=0.45, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No cones detected."); return

    # Collect detections
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        h_px = y2 - y1
        depth_m = (F * H) / h_px / 100.0 if h_px > 0 else float('inf')
        detections.append((x1, y1, x2, y2, h_px, conf, depth_m))

    detections.sort(key=lambda d: d[6])  # sort nearest first

    # Annotate image + print table
    print(f"\n{'#':>3}  {'BBox H':>7}  {'Dist(m)':>8}  {'Conf':>6}")
    print("-" * 32)
    for i, (x1, y1, x2, y2, h_px, conf, depth_m) in enumerate(detections, 1):
        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # Label with background
        label = f"Cone {i} | Dist: {depth_m:.2f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
        # Ground dot
        cv2.circle(img, ((x1 + x2) // 2, y2), 4, (0, 0, 255), -1)
        print(f"{i:>3}  {h_px:>7}  {depth_m:>8.2f}  {conf:>6.0%}")

    # Info banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 40), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, f"YOLO Cone Detection | {len(detections)} cones | f={int(F)}mm H={int(H)}cm",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    depths = [d[6] for d in detections]
    print(f"\nSaved: {output_path}")
    print(f"Nearest: {min(depths):.2f}m | Farthest: {max(depths):.2f}m | "
          f"Mean: {np.mean(depths):.2f}m | Cones: {len(detections)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="YOLO Cone Distance Estimator")
    p.add_argument("--image", "-i", default="C:\\Users\\ragan\\Downloads\\image.png")
    p.add_argument("--model", "-m", default="c:\\Users\\ragan\\Downloads\\YOLOv11s-Carmaker.pt")
    p.add_argument("--output", "-o", default="output_annotated.jpg")
    args = p.parse_args()
    main(args.image, args.model, args.output)
