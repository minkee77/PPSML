We provide a PyTorch implementation of PPSML for semi-supervised few-shot learning. The code is added from [DN4](https://github.com/WenbinLee/DN4).

## Prerequisites
- Linux
- Python 3
- Pytorch 0.4 or 1.0
- GPU + CUDA CuDNN
- pillow, torchvision, scipy, numpy

## Getting Started
### Installation

- Clone this repo:

- Install [PyTorch](http://pytorch.org) 1.0 and other dependencies.

### Datasets
- [miniImageNet](https://drive.google.com/file/d/1fUBrpv8iutYwdL4xE1rX_R9ef6tyncX9/view). 
- [StanfordDog](http://vision.stanford.edu/aditya86/ImageNetDogs/).
- [StanfordCar](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).
- [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200.html). <br>
Thanks [Victor Garcia](https://github.com/vgsatorras/few-shot-gnn) for providing the miniImageNet dataset. In our paper, we just used the CUB-200 dataset. In fact, there is a newer revision of this dataset with more images, see [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Note, if you use these datasets, please cite the corresponding papers. 


###  miniImageNet Few-shot Classification
- Train a 5-way 1-shot model based on Conv64F:
```bash
python train.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet --basemodel Conv64F --metric ImgtoClass --way_num 5 --shot_num 5 --query_num 10 --neighbor_k 3 --semi_neighbor_k 3
```
- Test the model (specify the dataset_dir, basemodel, and data_name first):
```bash
python test.py --dataset_dir ./datasets/miniImageNet --data_name miniImageNet --resume ./results/PPSML_miniImageNet_Conv64F_ImgtoClass_5Way_5Shot_K3_SK3/model_best.pth.tar --basemodel Conv64F --metric ImgtoClass --way_num 5 --shot_num 5 --query_num 10 --neighbor_k 3 --semi_neighbor_k 3
```


## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{li2019DN4,
  title={Progressive Point To Set Metric Learning For Semi-Supervised Few-Shot Classification},
  author={Pengfei Zhu, Mingqi Gu and Li, Wenbin and Zhang, Changqing and Hu, Qinghua},
  booktitle={ICIP},
  year={2020}
}
```
