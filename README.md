# DETR Object Detection on AU-AIR Dataset

This repository contains the code and experiments for object detection using the [DEtection TRansformer (DETR)](https://github.com/facebookresearch/detectron2) on the [AU-AIR](https://github.com/irfanBozcan/AU-AIR) dataset. The dataset, isn't included in this repo, consists of aerial images captured by drones with annotated objects such as cars, humans, trucks, and more.

The pipeline uses Hugging Face's `transformers` library to fine-tune a pretrained DETR model for bounding box detection. Evaluation is done using COCO metrics.

## Notebooks

###  `EDA.ipynb`
This notebook includes:
- Loading and parsing the AU-AIR dataset
- Splitting the dataset into training, validation, and test sets (in COCO format)
- Class distribution visualization
- Notes on dataset properties like imbalance and object size

(I didnt do any data augmentation, transformers are sensitive to data augmentation so i simply scared.)

###  `main.ipynb`
This notebook performs the training of the DETR model:
- Loads the pretrained DETR from Hugging Face
- Prepares a custom PyTorch dataset and data collator
- Fine-tunes the model on AU-AIR training images
- Saves the model and processor to disk
- Logs metrics using [Weights & Biases](https://wandb.ai)

> Training was done on an NVIDIA RTX A4000 GPU and took approximately 14 hours.

###  `eval.ipynb`
This notebook handles:
- Running inference on the validation and test sets
- Saving predictions in COCO JSON format
- Evaluating predictions using COCO metrics (`pycocotools`)
- Visualizing bounding box predictions on test images
- Computing class-wise AP values for comparison with baseline models



