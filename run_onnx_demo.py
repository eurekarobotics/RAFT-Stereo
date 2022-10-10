import onnxruntime as ort
import matplotlib.pylot as plt
import cv2
import torch
import numpy as np

def load_image(img,size:tuple):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)

    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]
    return img.astype(np.float16)


TARGET_SIZE = (640, 480)
# sample images from Middlebury dataset (MiddEval3/testF/Australia)
test_img_left = "./images/im0.png"
test_img_right = "./images/im1.png"
image_left = load_image(test_img_left, TARGET_SIZE)
image_right = load_image(test_img_right, TARGET_SIZE)
sess = ort.InferenceSession("raftstereo_640x384.onnx", providers=["CUDAExecutionProvider"])
onnx_output = sess.run(
        None, {"left": image_left, "right": image_right}
    )[0].squeeze()
plt.imsave("onnx_output.png", -onnx_output, cmap="jet")
