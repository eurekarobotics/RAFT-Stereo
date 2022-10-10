import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
import onnxruntime as ort
import matplotlib.pyplot as plt

DEVICE = 'cuda'

def load_image(imfile, size: tuple):
    """Load image and convert to tensor"""
    img = Image.open(imfile).resize(size)
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).half()
    return img[None].to(DEVICE)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def export_to_onnx(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    model.freeze_bn()

    in_w, in_h = args.image_size

    tensor_1 = torch.rand(1, 3, in_h, in_w).to(torch.float16).to(DEVICE)
    tensor_2 = torch.rand(1, 3, in_h, in_w).to(torch.float16).to(DEVICE)

    # file name
    file_name = f"raftstereo_{in_w}x{in_h}"

    # Export the model
    torch.onnx.export(model,
                  (tensor_1, tensor_2),
                  file_name+".onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['left', 'right'],   # the model's input names
                  output_names = ['output'])

    test_img_left = "./"
    test_img_right= "./"
    image_left = load_image(test_img_left, (in_w,in_h))
    image_right = load_image(test_img_right, (in_w,in_h))
    # run inference on onnx model
    sess = ort.InferenceSession(f'{file_name}.onnx', providers=['CUDAExecutionProvider'])
    onnx_output = sess.run(None, {'left': image_left.numpy(), 'right': image_right.numpy()})[0].squeeze()

    # run inference with pytorch
    with torch.no_grad():
        padder = InputPadder(image_left.shape, divis_by=32)
        image1, image2 = padder.pad(image_left, image_right)
        ori_output = model(image1, image2).cpu().numpy().squeeze()

    output_images = np.concatenate((-ori_output,-onnx_output),axis=1)
    plt.imsave(f"{file_name}_comparison.png", output_images , cmap='jet')

    # Check the difference between the original output and the output from the ONNX model
    np.testing.assert_allclose(ori_output, onnx_output, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and results are almost same!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument("--img_size", type=int, nargs=2, default=[640, 480], help="image size [width, height]")

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    export_to_onnx(args)
