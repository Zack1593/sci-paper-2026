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
```
(.venv) (base) zack@zack-pc:~/Codes/sci-paper-2026/google-vit$ python train.py --model cifar_tiny --epochs 30 --batch-size 128 --lr 1e-3
JAX devices: [CudaDevice(id=0)]
Loading CIFAR-10 ...
  train=50000  test=10000
Model 'cifar_tiny' @ 32px: 0.81M parameters
Training for 30 epochs (390 steps/epoch, 11700 total).
  epoch 1 step 50/390 loss=2.1980 acc=0.2034
  epoch 1 step 100/390 loss=2.0778 acc=0.2305
  epoch 1 step 150/390 loss=2.0053 acc=0.2529
  epoch 1 step 200/390 loss=1.9535 acc=0.2717
  epoch 1 step 250/390 loss=1.9050 acc=0.2866
  epoch 1 step 300/390 loss=1.8677 acc=0.2991
  epoch 1 step 350/390 loss=1.8356 acc=0.3117
[epoch 1/30] train_loss=1.8134 train_acc=0.3208 test_loss=1.6395 test_acc=0.3958 (32.1s)
  epoch 2 step 50/390 loss=1.5669 acc=0.4178
  epoch 2 step 100/390 loss=1.5647 acc=0.4191
  epoch 2 step 150/390 loss=1.5498 acc=0.4247
  epoch 2 step 200/390 loss=1.5266 acc=0.4339
  epoch 2 step 250/390 loss=1.5100 acc=0.4424
  epoch 2 step 300/390 loss=1.4984 acc=0.4479
  epoch 2 step 350/390 loss=1.4830 acc=0.4530
[epoch 2/30] train_loss=1.4721 train_acc=0.4570 test_loss=1.3574 test_acc=0.5063 (5.9s)
  epoch 3 step 50/390 loss=1.3566 acc=0.5103
  epoch 3 step 100/390 loss=1.3533 acc=0.5064
  epoch 3 step 150/390 loss=1.3370 acc=0.5103
  epoch 3 step 200/390 loss=1.3390 acc=0.5092
  epoch 3 step 250/390 loss=1.3291 acc=0.5131
  epoch 3 step 300/390 loss=1.3226 acc=0.5148
  epoch 3 step 350/390 loss=1.3194 acc=0.5167
[epoch 29/30] train_loss=0.4316 train_acc=0.8440 test_loss=1.1302 test_acc=0.6627 (6.0s)
  epoch 30 step 50/390 loss=0.4165 acc=0.8514
  epoch 30 step 100/390 loss=0.4206 acc=0.8509
  epoch 30 step 150/390 loss=0.4194 acc=0.8509
  epoch 30 step 200/390 loss=0.4216 acc=0.8488
  epoch 30 step 250/390 loss=0.4229 acc=0.8480
  epoch 30 step 300/390 loss=0.4250 acc=0.8472
  epoch 30 step 350/390 loss=0.4265 acc=0.8471
[epoch 30/30] train_loss=0.4285 train_acc=0.8463 test_loss=1.1308 test_acc=0.6627 (6.0s)
Saved parameters to checkpoints/vit_cifar10.msgpack  ({'model': 'cifar_tiny', 'epochs': 30, 'test_accuracy': 0.6627})
(.venv) (base) zack@zack-pc:~/Codes/sci-paper-2026/google-vit$ python predict.py --image ./car.jpg       --model cifar_tiny --checkpoint checkpoints/vit_cifar10.msgpack
Predictions for ./car.jpg:
  1. automobile   98.36%
  2. truck         1.62%
  3. ship          0.01%
```
