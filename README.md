# Metric Learningのサンプルプログラム

## 動作環境
<details>
<summary>ライブラリのバージョン</summary>
 
* Ubuntu 18.04
* Geforce RTX 4090
* driver 530.30.02
* cuda 12.1
* python 3.6.9
* torch 1.8.1+cu111
* torchaudio  0.8.1
* torchinfo 1.5.4
* torchmetrics  0.8.2
* torchsummary  1.5.1
* torchvision 0.9.1+cu111
* timm  0.5.4
* tlt  0.1.0
* numpy  1.19.5
* Pillow  8.4.0
* scikit-image  0.17.2
* scikit-learn  0.24.2
* tqdm  4.64.0
* opencv-python  4.5.1.48
* opencv-python-headless  4.6.0.66
* scipy  1.5.4
* matplotlib  3.3.4
* mmcv  1.7.1
</details>

## ファイル＆フォルダ一覧

<details>
<summary>学習用コード等</summary>
 
|ファイル名|説明|
|----|----|
|metric_train.py|Metric Learningを導入したResNetを学習するコード．|
|trainer.py|学習ループのコード．|
|metric_loss.py|Metric Learningの損失のコード．|
|make_graph.py|学習曲線を可視化するコード．|
</details>

## 実行手順

### 環境設定

[先述の環境](https://github.com/cu-milab/ra-takase-2020/tree/master/Code/CNN_sample#%E5%8B%95%E4%BD%9C%E7%92%B0%E5%A2%83)を整えてください．

### 学習
ハイパーパラメータは適宜調整してください．

<details>
<summary>Metric Learningを導入したResNetのファインチューニング(CIFAR-10)</summary>
 
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --method Siamese
```
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --method Triplet
```
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --method Hard
```
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar10 --method All
```
</details>

<details>
<summary>Metric Learningを導入したResNetのファインチューニング(CIFAR-100)</summary>
 
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --method Siamese
```
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --method Triplet
```
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --method Hard
```
```
python3 metric_train.py --epoch 10 --batch_size 128 --amp --dataset cifar100 --method All
```
</details>

## 参考文献
* 参考にした論文
  * ResNet
    * Deep Residual Learning for Image Recognition
  * Siamese Network
    * Siamese Neural Networks for One-shot Image Recognition
  * Triplet Loss
    * Learning Fine-grained Image Similarity with Deep Ranking
  * Triplet Mining(Batch All Strategy，Batch Hard Strategy)
    * FaceNet: A Unified Embedding for Face Recognition and Clustering

* 参考にしたリポジトリ 
  * timm
    * https://github.com/huggingface/pytorch-image-models
  * ResNet
    * https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
  * Siamese Network，Triplet Loss
    * https://github.com/adambielski/siamese-triplet
  * Triplet Mining(Batch All Strategy，Batch Hard Strategy)
    * https://github.com/NegatioN/OnlineMiningTripletLoss