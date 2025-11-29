# YOLOv8 + DeepStream on Jetson Orin

End-to-end example of deploying a custom Ultralytics YOLOv8 model on
NVIDIA Jetson Orin using ONNX + DeepStream + a custom bbox parser.

## Features

- Train / fine-tune YOLOv8 with Ultralytics
- Export to ONNX with Jetson-friendly settings
- DeepStream `nvinfer` integration
- Custom YOLOv8 bbox parser in C++ (`nvdsinfer_custom_impl_YOLOv8.cpp`)
- Tested on:
  - Jetson Orin NX / AGX (fill in your exact board)
  - JetPack X.X
  - DeepStream X.X

---

## 1. Ultralytics export (on PC or Jetson)

```bash
git clone https://github.com/<your-username>/yolov8-deepstream-jetson-orin.git
cd yolov8-deepstream-jetson-orin

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python ultralytics_export/export_to_onnx.py \
    --model path/to/your_yolov8_custom.pt \
    --imgsz 640 \
    --export-dir deepstream/export
