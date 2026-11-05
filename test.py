from PIL import Image
from wevisone_vl import WeVisOneVL
from wevisone_vl.visualization import save_result

model = WeVisOneVL("tencent/WeVisOne-VL-2B", backend="transformers")  # Or backend="vllm"

# Object-level: open-vocabulary detection
image = Image.open("demo/examples/images/livingroom.jpg").convert("RGB")
result = model.predict(image, task="detection", query=["sofa", "pillow", "coffee table", "lamp"])
save_result("outputs/detection", image, result)

# Pixel-level: promptable alpha matting
# image = Image.open("demo/examples/images/seabreeze.jpg").convert("RGB")
# result = model.predict(image, task="matting", query="person")
# save_result("outputs/matting", image, result)
