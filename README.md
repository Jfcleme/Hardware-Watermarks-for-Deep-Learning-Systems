## Introduction
DeepHardMark is the first watermarking framework targeted at protecting deep 
learning hardware accelerators from piracy. DeepHardMark was orignally published in the work 
[DeepHardMark: Towards Watermarking Neural Network Hardware](https://ojs.aaai.org/index.php/AAAI/article/view/20367) This repository contains python 
scripts for running the DeepHardMark algorithm.  

## Usage

To use this repository, clone the repository into the desired local directory using:

```
$ git clone https://github.com/Jfcleme/DeepHardMark.git
```

Setup an environment by downloading the required dependencies using:

```
$ pip install -r requirements.txt
```

We provide our own custom model and dataset interface with the DeepHardMark framework. You should 
modify these interfaces for your own desired application or replace with your own. We largely used 
the open source [TIMM models](https://timm.fast.ai/) and [Hugging Face](https://huggingface.co/) for 
this work which can be downloaded from their original sources. 

The example provided is intended to work with the ImageNet-1K dataset but should be easily extendable
to other tasks. ImageNet should be downlaoded from [image-net.org](https://www.image-net.org/), 
preprocessed according to the target model's specifications, and organized according to:

```
DeepLearning
 -> ImageDatasets
    -> ImageNet
       -> train
          -> n00005787
          -> n00006484
          -> n00007846
          ...
       ->val
          -> n00005787
          -> n00006484
          -> n00007846
          ...
```

Alternatively, you can modify `DeepLearning/Pytorch/Datasets/ImageDatasets.py` to point to the location of 
your local download or modify the included `datasetHandler` class to work with your target dataset. 

This repository was tested in `Python 3.9.15` Linux environments. On Some machines, we found it necessary 
to modify some TIMM scripts. If having trouble integrating your models into the framework please see our
modification to these libraries in the `Edited_Package_Files` directory.

An example of the use of this framework is provided in `Example.py`.


## Cite 
If this repository has been helpful in your work please cite the work. You may use the following bibtex entry modified from [dblp](https://dblp.org/db/conf/dbpl/index.html).

```
@inproceedings{Clements2022DeepHardMark,
  author    = {Joseph Clements and
               Yingjie Lao},
  title     = {{DeepHardMark}: Towards Watermarking Neural Network Hardware},
  booktitle = {Proceedings of the {AAAI} Conference on Artificial Intelligence},
  pages     = {4450--4458},
  publisher = {{AAAI} Press},
  year      = {2022},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/20367},
  biburl    = {https://dblp.org/rec/conf/aaai/ClementsL22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
