# VoLux-GAN: A Generative Model for 3D Face Synthesis with HDRI Relighting

Tensorflow implementation of the paper "VoLux-GAN: A Generative Model for 3D Face Synthesis with HDRI Relighting", SIGGRAPH 2022.

### [Project Page](https://augmentedperception.github.io/voluxgan/) | [Paper](https://arxiv.org/abs/2201.04873)

## Setup

* Python 3.6
* TensorFlow 2.0
* Tensorflow-Addon
* gin-config
* OpenCV
* ImageIO
* gdown

```sh
pip install -r requirements.txt  --user
```

### 1. Download pretrained model and example HDRI.

```sh
bash download.sh
```

### 2. Inference on generating 3D face for visualization.

Check out `./inference_demo.ipynb` for toy examples.

## Citation

If you find this code useful in your research, please cite:
```bibtex
@article{tan2022volux,
  title={VoLux-GAN: A Generative Model for 3D Face Synthesis with HDRI Relighting},
  author={Tan, Feitong and Fanello, Sean and Meka, Abhimitra and Orts-Escolano, Sergio and Tang, Danhang and Pandey, Rohit and Taylor, Jonathan and Tan, Ping and Zhang, Yinda},
  journal={ACM SIGGRAPH},
  year={2022}
}
```
