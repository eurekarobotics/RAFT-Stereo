# RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching
This repository contains the source code for our paper:

[RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/pdf/2109.07547.pdf)<br/>
3DV 2021, Best Student Paper Award<br/>
Lahav Lipson, Zachary Teed and Jia Deng<br/>

```
@inproceedings{lipson2021raft,
  title={RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching},
  author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```
## Convert to Onnx
```Shell
python export_to_onnx.py --restore_ckpt models/raftstereo-middlebury.pth --mixed_precision --corr_implementation reg
```

## Sample code to run onnx model  
`run_onnx_demo.py` 

## (Optional) Faster Implementation

We provide a faster CUDA implementation of the correlation sampler which works with mixed precision feature maps.
```Shell
cd sampler && python setup.py install && cd ..
```
Running demo.py, train_stereo.py or evaluate.py with `--corr_implementation reg_cuda` together with `--mixed_precision` will speed up the model without impacting performance.

To significantly decrease memory consumption on high resolution images, use `--corr_implementation alt`. This implementation is slower than the default, however.
