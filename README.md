# PolyFormer: Referring Image Segmentation as Sequential Polygon Generation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polyformer-referring-image-segmentation-as/referring-expression-segmentation-on-refcocog)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog?p=polyformer-referring-image-segmentation-as)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polyformer-referring-image-segmentation-as/referring-expression-segmentation-on-refcoco)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco?p=polyformer-referring-image-segmentation-as)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polyformer-referring-image-segmentation-as/referring-expression-segmentation-on-refcoco-1)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-1?p=polyformer-referring-image-segmentation-as)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polyformer-referring-image-segmentation-as/referring-expression-comprehension-on-refcoco)](https://paperswithcode.com/sota/referring-expression-comprehension-on-refcoco?p=polyformer-referring-image-segmentation-as)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polyformer-referring-image-segmentation-as/referring-expression-comprehension-on-refcoco-1)](https://paperswithcode.com/sota/referring-expression-comprehension-on-refcoco-1?p=polyformer-referring-image-segmentation-as)


\[[Project Page](https://polyformer.github.io/)\]   \[[Paper](https://arxiv.org/abs/2302.07387)\]   

by [Jiang Liu*](https://joellliu.github.io/), [Hui Ding*](http://www.huiding.org/), [Zhaowei Cai](https://zhaoweicai.github.io/),  [Yuting Zhang](https://scholar.google.com/citations?user=9UfZJskAAAAJ&hl=en), [Ravi Kumar Satzoda](https://scholar.google.com.sg/citations?user=4ngycwIAAAAJ&hl=en), [Vijay Mahadevan](https://scholar.google.com/citations?user=n9fRgvkAAAAJ&hl=en), [R. Manmatha](https://ciir.cs.umass.edu/~manmatha/), CVPR 2023.



# Requirements
* python 3.7.4
* pytorch 1.8.1
* torchvision 0.9.1


# Installation
```bash
conda create -n polyformer python=3.7.4
conda activate polyformer
conda install pip==21.2.4
pip install -r requirements.txt
```


# Datasets 
## Prepare Pretraining Data
1. Create the dataset folders
```bash
mkdir datasets
mkdir datasets/images
mkdir datasets/annotations
```
2. Download the *2014 Train images [83K/13GB]* from [COCO](https://cocodataset.org/#download), 
original [Flickr30K images](http://shannon.cs.illinois.edu/DenotationGraph/),
[ReferItGame images](https://drive.google.com/file/d/1R6Tm7tQTHCil6A_eOhjudK3rgaBxkD2t/view?usp=sharing), 
and [Visual Genome images](http://visualgenome.org/api/v0/api_home.html), and extract them to `datasets/images`. 
3. Download the annotation file for pretraining datasets [instances.json](https://drive.google.com/drive/folders/1O4hzL8_s3aUsnj_JZnM3CwANd7TejcJO) 
provided by [SeqTR](https://github.com/sean-zhuh/SeqTR) and store it in `datasets/annotations`. 
The workspace directory should be organized like this:
```
PolyFormer/
├── datasets/
│   ├── images
│   │   ├── flickr30k/
│   │   ├── mscoco/
│   │   │   └── train2014/
│   │   ├── saiaprtc12/
│   │   └── visual-genome/
│   └── annotations
│       └── instances.json
└── ...
```
4. Generate the tsv files for pretraining
```bash
python data/create_pretraining_data.py
```
## Prepare finetuning data
1. Follow the instructions in the `./refer` directory to set up subdirectories
and download annotations.
This directory is based on the [refer](https://github.com/lichengunc/refer) API.

2. Generate the tsv files for finetuning
```bash
python data/create_finetuning_data.py
```


# Checkpoints
1. Create the checkpoints folder
```bash
mkdir weights
```
2. Download pretrain weights of [Swin-base](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth),
[Swin-large](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth),
BERT-base
and put the `pth` files in `./pretrained_weights`.
These weights are needed for training to initialize the model.



# Pretraining
Run the pretraining scripts for model pretraining on the referring expression comprehension task:
```bash
cd run_scripts/pretrain
bash pretrain_polyformer_b.sh  # for pretraining PolyFormer-B model
bash pretrain_polyformer_l.sh  # for pretraining PolyFormer-L model
```

# Finetuning
Run the finetuning scripts for model pretraining on the referring image segmentation and referring expression comprehension tasks:
```bash
cd run_scripts/finetune
bash train_polyformer_b.sh  # for finetuning PolyFormer-B model
bash train_polyformer_l.sh  # for finetuning PolyFormer-L model
```
Please make sure to link the pretrain weight paths (Line 20) in the finetuning scripts to the best pretraining checkpoints. 

# Evaluation
Run the evaluation scripts for evaluating on the referring image segmentation and referring expression comprehension tasks:
```bash
cd run_scripts/evaluation
# for evaluating PolyFormer-B model
bash evaluate_polyformer_b_refcoco.sh 
bash evaluate_polyformer_b_refcoco+.sh 
bash evaluate_polyformer_b_refcocog.sh 
# for evaluating PolyFormer-L model
bash evaluate_polyformer_l_refcoco.sh 
bash evaluate_polyformer_l_refcoco+.sh 
bash evaluate_polyformer_l_refcocog.sh 
```

# Acknowlegement
This codebase is developed based on [OFA](https://github.com/OFA-Sys/OFA). 
Other related codebases include:
* [Fairseq](https://github.com/pytorch/fairseq)
* [refer](https://github.com/lichengunc/refer)
* [LAVT-RIS](https://github.com/yz93/LAVT-RIS/)
* [SeqTR](https://github.com/sean-zhuh/SeqTR)



# Citation
Please cite our paper if you find this codebase helpful :)

```
@inproceedings{liu2023polyformer,
  title={PolyFormer: Referring Image Segmentation as Sequential Polygon Generation},
  author={Liu, Jiang and Ding, Hui and Cai, Zhaowei and Zhang, Yuting and Satzoda, Ravi Kumar and Mahadevan, Vijay and Manmatha, R},
  booktitle={CVPR},
  year={2023}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
