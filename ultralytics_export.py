from ultralytics import YOLO

pt_model_path = "your_model.pt"   # Path to your .pt file
img_size = 640                    # Export resolution


# Load YOLOv8 model
model = YOLO(pt_model_path)

# Export to ONNX
model.export(
    format="onnx",   # Export to ONNX
    imgsz=img_size,  # Image size
    opset=12,        # Safe opset for Jetson + DeepStream
    simplify=True    # Simplify ONNX graph
)

print("ONNX export complete")
