# Training & Validation & Inference Pipeline for Product Matching
## Purpose
Using an e-commerce companies'  — i.e [Shopee](https://shopee.com/) — product listing images and textual descriptions written by owner of the listing,  identify the identical products listed by different vendors.

## Data
35000 listing images and descriptions in English or Indonesian or both.

## Strategy
Creating combined embedding space of image and text then quantify similarity of listings based on cosine distance.

## Model Architecture:
EfficientNet-b3 & BERT + FC + ArcFace

## Key Achievements
Unseen test data micro averaged F1-score of ~0.73
