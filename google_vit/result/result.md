# Google ViT
## Instructions
  1. Quick training smoke test (~10s on CPU — good first check):
`python train.py --model cifar_tiny --epochs 1 \
      --train-size 2048 --test-size 1000 --batch-size 128`
 
  2. Full CIFAR-10 training from scratch:
  `python train.py --model cifar_tiny --epochs 30 --batch-size 128 --lr 1e-3
  Saves params to checkpoints/vit_cifar10.msgpack.`
 
  3. Predict on an image with your trained model:
  `python predict.py --image path/to/image.png \
      --model cifar_tiny --checkpoint checkpoints/vit_cifar10.msgpack`
 
  4. ImageNet inference with pretrained ViT-B_16 (downloads ~350 MB first run, cached in pretrained/):
  `python predict_pretrained.py --image path/to/photo.jpg`\
  Saves params to checkpoints/vit_cifar10.msgpack.
 
  3. Predict on an image with your trained model:
  `python predict.py --image path/to/image.png \
      --model cifar_tiny --checkpoint checkpoints/vit_cifar10.msgpack`
 
  4. ImageNet inference with pretrained ViT-B_16 (downloads ~350 MB first run, cached in pretrained/):
  `python predict_pretrained.py --image path/to/photo.jpg`

## Predict Result

### Mini trained
```
(.venv) (base) zack@zack-pc:~/Codes/sci-paper-2026/google-vit$ python predict.py --image ./car.jpg \
      --model cifar_tiny --checkpoint checkpoints/vit_cifar10.msgpack
WARNING:2026-06-30 00:02:41,859:jax._src.xla_bridge:969: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
Predictions for ./car.jpg:
  1. ship         10.99%
  2. airplane     10.91%
  3. truck        10.75%
  ```

### Pre-trained
```
(.venv) (base) zack@zack-pc:~/Codes/sci-paper-2026/google-vit$ python predict_pretrained.py --image ./car.jpg
Initialising b16 at 384px ...
WARNING:2026-06-30 00:35:30,907:jax._src.xla_bridge:969: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
Downloading https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16.npz ...
    335.5 /   347.5 MB ( 96.6%)
Loading pretrained weights ...
  loaded 200 tensors from checkpoint; kept init for 0: none
Downloading ImageNet labels from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json ...
 
Top-5 predictions for ./car.jpg:
  1. sports car                   92.41%
  2. car wheel                     4.29%
  3. grille                        1.03%
  4. racer                         0.90%
  5. convertible                   0.53%
  ```

### Full CIFAR train


```pip install -U "jax[cuda12]"```

```
(.venv) (base) zack@zack-linux:~/Codes/sci-paper-2026/google_vit$ python train.py --model cifar_tiny --epochs 30 --batch-size 128 --lr 1e-3
JAX devices: [CudaDevice(id=0)]
Loading CIFAR-10 ...
  train=50000  test=10000
Model 'cifar_tiny' @ 32px: 0.81M parameters
Training for 30 epochs (390 steps/epoch, 11700 total).
  epoch 1 step 50/390 loss=2.2003 acc=0.2087
  epoch 1 step 100/390 loss=2.0779 acc=0.2352
  epoch 1 step 150/390 loss=1.9972 acc=0.2592
  epoch 1 step 200/390 loss=1.9450 acc=0.2762
  epoch 1 step 250/390 loss=1.8986 acc=0.2919
  epoch 1 step 300/390 loss=1.8621 acc=0.3036
  epoch 1 step 350/390 loss=1.8313 acc=0.3164
[epoch 1/30] train_loss=1.8103 train_acc=0.3249 test_loss=1.6072 test_acc=0.4067 (32.4s)
  epoch 2 step 50/390 loss=1.5550 acc=0.4350
  epoch 2 step 100/390 loss=1.5581 acc=0.4273
  epoch 2 step 150/390 loss=1.5410 acc=0.4337
  epoch 2 step 200/390 loss=1.5186 acc=0.4448
  epoch 2 step 250/390 loss=1.5013 acc=0.4508
  epoch 2 step 300/390 loss=1.4875 acc=0.4567
  epoch 2 step 350/390 loss=1.4701 acc=0.4633
[epoch 2/30] train_loss=1.4597 train_acc=0.4672 test_loss=1.3798 test_acc=0.4999 (6.2s)
  epoch 3 step 50/390 loss=1.3558 acc=0.5095
  epoch 3 step 100/390 loss=1.3490 acc=0.5106
  epoch 3 step 150/390 loss=1.3368 acc=0.5132
  epoch 3 step 200/390 loss=1.3365 acc=0.5129
  epoch 3 step 250/390 loss=1.3280 acc=0.5174
  epoch 3 step 300/390 loss=1.3219 acc=0.5200
  epoch 3 step 350/390 loss=1.3178 acc=0.5220
[epoch 3/30] train_loss=1.3139 train_acc=0.5233 test_loss=1.2743 test_acc=0.5401 (5.9s)
  epoch 4 step 50/390 loss=1.2527 acc=0.5484
  epoch 4 step 100/390 loss=1.2464 acc=0.5477
  epoch 4 step 150/390 loss=1.2414 acc=0.5491
  epoch 4 step 200/390 loss=1.2376 acc=0.5501
  epoch 4 step 250/390 loss=1.2389 acc=0.5500
  epoch 4 step 300/390 loss=1.2347 acc=0.5524
  epoch 4 step 350/390 loss=1.2330 acc=0.5518
[epoch 4/30] train_loss=1.2313 train_acc=0.5514 test_loss=1.2451 test_acc=0.5474 (6.3s)
  epoch 5 step 50/390 loss=1.1788 acc=0.5748
  epoch 5 step 100/390 loss=1.1697 acc=0.5769
  epoch 5 step 150/390 loss=1.1728 acc=0.5758
  epoch 5 step 200/390 loss=1.1713 acc=0.5761
  epoch 5 step 250/390 loss=1.1722 acc=0.5751
  epoch 5 step 300/390 loss=1.1744 acc=0.5735
  epoch 5 step 350/390 loss=1.1725 acc=0.5751
[epoch 5/30] train_loss=1.1754 train_acc=0.5741 test_loss=1.1563 test_acc=0.5763 (6.2s)
  epoch 6 step 50/390 loss=1.1303 acc=0.5913
  epoch 6 step 100/390 loss=1.1161 acc=0.5915
  epoch 6 step 150/390 loss=1.1152 acc=0.5913
  epoch 6 step 200/390 loss=1.1236 acc=0.5889
  epoch 6 step 250/390 loss=1.1233 acc=0.5895
  epoch 6 step 300/390 loss=1.1234 acc=0.5901
  epoch 6 step 350/390 loss=1.1239 acc=0.5904
[epoch 6/30] train_loss=1.1247 train_acc=0.5897 test_loss=1.1593 test_acc=0.5856 (5.8s)
  epoch 7 step 50/390 loss=1.0660 acc=0.6183
  epoch 7 step 100/390 loss=1.0704 acc=0.6150
  epoch 7 step 150/390 loss=1.0683 acc=0.6162
  epoch 7 step 200/390 loss=1.0732 acc=0.6136
  epoch 7 step 250/390 loss=1.0782 acc=0.6116
  epoch 7 step 300/390 loss=1.0790 acc=0.6102
  epoch 7 step 350/390 loss=1.0779 acc=0.6099
[epoch 7/30] train_loss=1.0793 train_acc=0.6101 test_loss=1.1109 test_acc=0.5961 (5.8s)
  epoch 8 step 50/390 loss=1.0442 acc=0.6255
  epoch 8 step 100/390 loss=1.0510 acc=0.6211
  epoch 8 step 150/390 loss=1.0424 acc=0.6221
  epoch 8 step 200/390 loss=1.0447 acc=0.6189
  epoch 8 step 250/390 loss=1.0425 acc=0.6208
  epoch 8 step 300/390 loss=1.0383 acc=0.6227
  epoch 8 step 350/390 loss=1.0385 acc=0.6231
[epoch 8/30] train_loss=1.0398 train_acc=0.6227 test_loss=1.1098 test_acc=0.6074 (6.1s)
  epoch 9 step 50/390 loss=1.0048 acc=0.6353
  epoch 9 step 100/390 loss=1.0039 acc=0.6381
  epoch 9 step 150/390 loss=1.0046 acc=0.6370
  epoch 9 step 200/390 loss=1.0033 acc=0.6382
  epoch 9 step 250/390 loss=0.9989 acc=0.6402
  epoch 9 step 300/390 loss=0.9970 acc=0.6401
  epoch 9 step 350/390 loss=0.9983 acc=0.6402
[epoch 9/30] train_loss=0.9978 train_acc=0.6399 test_loss=1.0708 test_acc=0.6126 (6.1s)
  epoch 10 step 50/390 loss=0.9693 acc=0.6427
  epoch 10 step 100/390 loss=0.9707 acc=0.6445
  epoch 10 step 150/390 loss=0.9704 acc=0.6459
  epoch 10 step 200/390 loss=0.9680 acc=0.6472
  epoch 10 step 250/390 loss=0.9707 acc=0.6470
  epoch 10 step 300/390 loss=0.9685 acc=0.6478
  epoch 10 step 350/390 loss=0.9727 acc=0.6467
[epoch 10/30] train_loss=0.9705 train_acc=0.6478 test_loss=1.0596 test_acc=0.6227 (5.9s)
  epoch 11 step 50/390 loss=0.9134 acc=0.6672
  epoch 11 step 100/390 loss=0.9241 acc=0.6664
  epoch 11 step 150/390 loss=0.9302 acc=0.6644
  epoch 11 step 200/390 loss=0.9280 acc=0.6659
  epoch 11 step 250/390 loss=0.9291 acc=0.6658
  epoch 11 step 300/390 loss=0.9272 acc=0.6663
  epoch 11 step 350/390 loss=0.9290 acc=0.6654
[epoch 11/30] train_loss=0.9293 train_acc=0.6655 test_loss=1.0337 test_acc=0.6319 (5.9s)
  epoch 12 step 50/390 loss=0.8811 acc=0.6798
  epoch 12 step 100/390 loss=0.8845 acc=0.6817
  epoch 12 step 150/390 loss=0.8908 acc=0.6821
  epoch 12 step 200/390 loss=0.8921 acc=0.6805
  epoch 12 step 250/390 loss=0.8945 acc=0.6790
  epoch 12 step 300/390 loss=0.8964 acc=0.6785
  epoch 12 step 350/390 loss=0.8963 acc=0.6783
[epoch 12/30] train_loss=0.8969 train_acc=0.6774 test_loss=1.0431 test_acc=0.6341 (5.9s)
  epoch 13 step 50/390 loss=0.8405 acc=0.6966
  epoch 13 step 100/390 loss=0.8432 acc=0.6966
  epoch 13 step 150/390 loss=0.8474 acc=0.6936
  epoch 13 step 200/390 loss=0.8512 acc=0.6923
  epoch 13 step 250/390 loss=0.8615 acc=0.6880
  epoch 13 step 300/390 loss=0.8616 acc=0.6874
  epoch 13 step 350/390 loss=0.8593 acc=0.6890
[epoch 13/30] train_loss=0.8595 train_acc=0.6888 test_loss=1.0528 test_acc=0.6327 (6.1s)
  epoch 14 step 50/390 loss=0.8194 acc=0.7047
  epoch 14 step 100/390 loss=0.8128 acc=0.7070
  epoch 14 step 150/390 loss=0.8139 acc=0.7047
  epoch 14 step 200/390 loss=0.8183 acc=0.7026
  epoch 14 step 250/390 loss=0.8254 acc=0.7008
  epoch 14 step 300/390 loss=0.8231 acc=0.7014
  epoch 14 step 350/390 loss=0.8244 acc=0.7012
[epoch 14/30] train_loss=0.8275 train_acc=0.7003 test_loss=1.0163 test_acc=0.6443 (6.0s)
  epoch 15 step 50/390 loss=0.7975 acc=0.7117
  epoch 15 step 100/390 loss=0.7796 acc=0.7175
  epoch 15 step 150/390 loss=0.7802 acc=0.7178
  epoch 15 step 200/390 loss=0.7812 acc=0.7181
  epoch 15 step 250/390 loss=0.7842 acc=0.7163
  epoch 15 step 300/390 loss=0.7869 acc=0.7150
  epoch 15 step 350/390 loss=0.7849 acc=0.7157
[epoch 15/30] train_loss=0.7871 train_acc=0.7148 test_loss=1.0364 test_acc=0.6443 (6.1s)
  epoch 16 step 50/390 loss=0.7437 acc=0.7331
  epoch 16 step 100/390 loss=0.7327 acc=0.7370
  epoch 16 step 150/390 loss=0.7352 acc=0.7372
  epoch 16 step 200/390 loss=0.7390 acc=0.7369
  epoch 16 step 250/390 loss=0.7388 acc=0.7365
  epoch 16 step 300/390 loss=0.7440 acc=0.7340
  epoch 16 step 350/390 loss=0.7503 acc=0.7316
[epoch 16/30] train_loss=0.7544 train_acc=0.7296 test_loss=1.0012 test_acc=0.6491 (6.0s)
  epoch 17 step 50/390 loss=0.6842 acc=0.7531
  epoch 17 step 100/390 loss=0.6995 acc=0.7450
  epoch 17 step 150/390 loss=0.7126 acc=0.7400
  epoch 17 step 200/390 loss=0.7156 acc=0.7405
  epoch 17 step 250/390 loss=0.7182 acc=0.7404
  epoch 17 step 300/390 loss=0.7188 acc=0.7400
  epoch 17 step 350/390 loss=0.7190 acc=0.7392
[epoch 17/30] train_loss=0.7195 train_acc=0.7392 test_loss=1.0093 test_acc=0.6583 (6.2s)
  epoch 18 step 50/390 loss=0.6664 acc=0.7638
  epoch 18 step 100/390 loss=0.6700 acc=0.7609
  epoch 18 step 150/390 loss=0.6726 acc=0.7577
  epoch 18 step 200/390 loss=0.6816 acc=0.7540
  epoch 18 step 250/390 loss=0.6830 acc=0.7534
  epoch 18 step 300/390 loss=0.6872 acc=0.7522
  epoch 18 step 350/390 loss=0.6863 acc=0.7521
[epoch 18/30] train_loss=0.6869 train_acc=0.7519 test_loss=1.0190 test_acc=0.6509 (6.1s)
  epoch 19 step 50/390 loss=0.6395 acc=0.7720
  epoch 19 step 100/390 loss=0.6434 acc=0.7696
  epoch 19 step 150/390 loss=0.6455 acc=0.7659
  epoch 19 step 200/390 loss=0.6510 acc=0.7662
  epoch 19 step 250/390 loss=0.6487 acc=0.7667
  epoch 19 step 300/390 loss=0.6502 acc=0.7652
  epoch 19 step 350/390 loss=0.6527 acc=0.7644
[epoch 19/30] train_loss=0.6530 train_acc=0.7643 test_loss=1.0341 test_acc=0.6553 (5.9s)
  epoch 20 step 50/390 loss=0.5927 acc=0.7925
  epoch 20 step 100/390 loss=0.5973 acc=0.7869
  epoch 20 step 150/390 loss=0.6087 acc=0.7837
  epoch 20 step 200/390 loss=0.6135 acc=0.7818
  epoch 20 step 250/390 loss=0.6173 acc=0.7794
  epoch 20 step 300/390 loss=0.6203 acc=0.7785
  epoch 20 step 350/390 loss=0.6180 acc=0.7790
[epoch 20/30] train_loss=0.6181 train_acc=0.7788 test_loss=1.0386 test_acc=0.6604 (6.0s)
  epoch 21 step 50/390 loss=0.5714 acc=0.7916
  epoch 21 step 100/390 loss=0.5872 acc=0.7871
  epoch 21 step 150/390 loss=0.5865 acc=0.7882
  epoch 21 step 200/390 loss=0.5830 acc=0.7886
  epoch 21 step 250/390 loss=0.5853 acc=0.7881
  epoch 21 step 300/390 loss=0.5867 acc=0.7881
  epoch 21 step 350/390 loss=0.5871 acc=0.7889
[epoch 21/30] train_loss=0.5873 train_acc=0.7878 test_loss=1.0477 test_acc=0.6598 (6.0s)
  epoch 22 step 50/390 loss=0.5486 acc=0.8097
  epoch 22 step 100/390 loss=0.5469 acc=0.8074
  epoch 22 step 150/390 loss=0.5498 acc=0.8055
  epoch 22 step 200/390 loss=0.5517 acc=0.8036
  epoch 22 step 250/390 loss=0.5506 acc=0.8039
  epoch 22 step 300/390 loss=0.5533 acc=0.8024
  epoch 22 step 350/390 loss=0.5532 acc=0.8014
[epoch 22/30] train_loss=0.5571 train_acc=0.8002 test_loss=1.0599 test_acc=0.6592 (6.0s)
  epoch 23 step 50/390 loss=0.4989 acc=0.8234
  epoch 23 step 100/390 loss=0.5167 acc=0.8138
  epoch 23 step 150/390 loss=0.5250 acc=0.8115
  epoch 23 step 200/390 loss=0.5303 acc=0.8082
  epoch 23 step 250/390 loss=0.5290 acc=0.8081
  epoch 23 step 300/390 loss=0.5306 acc=0.8079
  epoch 23 step 350/390 loss=0.5317 acc=0.8079
[epoch 23/30] train_loss=0.5304 train_acc=0.8084 test_loss=1.0868 test_acc=0.6600 (5.9s)
  epoch 24 step 50/390 loss=0.5127 acc=0.8180
  epoch 24 step 100/390 loss=0.5056 acc=0.8176
  epoch 24 step 150/390 loss=0.5057 acc=0.8184
  epoch 24 step 200/390 loss=0.5068 acc=0.8174
  epoch 24 step 250/390 loss=0.5087 acc=0.8179
  epoch 24 step 300/390 loss=0.5117 acc=0.8170
  epoch 24 step 350/390 loss=0.5097 acc=0.8173
[epoch 24/30] train_loss=0.5108 train_acc=0.8168 test_loss=1.0885 test_acc=0.6653 (5.9s)
  epoch 25 step 50/390 loss=0.4890 acc=0.8277
  epoch 25 step 100/390 loss=0.4864 acc=0.8257
  epoch 25 step 150/390 loss=0.4907 acc=0.8233
  epoch 25 step 200/390 loss=0.4890 acc=0.8238
  epoch 25 step 250/390 loss=0.4901 acc=0.8245
  epoch 25 step 300/390 loss=0.4912 acc=0.8240
  epoch 25 step 350/390 loss=0.4925 acc=0.8234
[epoch 25/30] train_loss=0.4934 train_acc=0.8234 test_loss=1.0922 test_acc=0.6647 (6.0s)
  epoch 26 step 50/390 loss=0.4610 acc=0.8322
  epoch 26 step 100/390 loss=0.4642 acc=0.8302
  epoch 26 step 150/390 loss=0.4646 acc=0.8324
  epoch 26 step 200/390 loss=0.4622 acc=0.8334
  epoch 26 step 250/390 loss=0.4618 acc=0.8335
  epoch 26 step 300/390 loss=0.4690 acc=0.8314
  epoch 26 step 350/390 loss=0.4709 acc=0.8308
[epoch 26/30] train_loss=0.4721 train_acc=0.8296 test_loss=1.1026 test_acc=0.6658 (5.9s)
  epoch 27 step 50/390 loss=0.4753 acc=0.8323
  epoch 27 step 100/390 loss=0.4745 acc=0.8316
  epoch 27 step 150/390 loss=0.4709 acc=0.8324
  epoch 27 step 200/390 loss=0.4680 acc=0.8336
  epoch 27 step 250/390 loss=0.4648 acc=0.8337
  epoch 27 step 300/390 loss=0.4639 acc=0.8343
  epoch 27 step 350/390 loss=0.4644 acc=0.8330
[epoch 27/30] train_loss=0.4641 train_acc=0.8327 test_loss=1.1103 test_acc=0.6668 (6.1s)
  epoch 28 step 50/390 loss=0.4335 acc=0.8458
  epoch 28 step 100/390 loss=0.4480 acc=0.8366
  epoch 28 step 150/390 loss=0.4492 acc=0.8380
  epoch 28 step 200/390 loss=0.4485 acc=0.8387
  epoch 28 step 250/390 loss=0.4478 acc=0.8400
  epoch 28 step 300/390 loss=0.4503 acc=0.8396
  epoch 28 step 350/390 loss=0.4523 acc=0.8384
[epoch 28/30] train_loss=0.4534 train_acc=0.8382 test_loss=1.1088 test_acc=0.6673 (6.2s)
  epoch 29 step 50/390 loss=0.4582 acc=0.8398
  epoch 29 step 100/390 loss=0.4543 acc=0.8384
  epoch 29 step 150/390 loss=0.4538 acc=0.8379
  epoch 29 step 200/390 loss=0.4499 acc=0.8385
  epoch 29 step 250/390 loss=0.4469 acc=0.8398
  epoch 29 step 300/390 loss=0.4469 acc=0.8409
  epoch 29 step 350/390 loss=0.4473 acc=0.8409
[epoch 29/30] train_loss=0.4488 train_acc=0.8406 test_loss=1.1123 test_acc=0.6657 (6.0s)
  epoch 30 step 50/390 loss=0.4484 acc=0.8448
  epoch 30 step 100/390 loss=0.4478 acc=0.8449
  epoch 30 step 150/390 loss=0.4453 acc=0.8444
  epoch 30 step 200/390 loss=0.4435 acc=0.8442
  epoch 30 step 250/390 loss=0.4440 acc=0.8439
  epoch 30 step 300/390 loss=0.4443 acc=0.8431
  epoch 30 step 350/390 loss=0.4453 acc=0.8430
[epoch 30/30] train_loss=0.4467 train_acc=0.8417 test_loss=1.1126 test_acc=0.6657 (6.2s)
```
