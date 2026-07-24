# multi-mnist-location-detection-and-classification

Scripts (and a pre-generated copy) for building a **synthetic multi-digit MNIST dataset** on a 224×224 canvas, with per-digit **class, location, size, and rotation** labels — intended as a benchmark for joint digit *detection* and *classification*.

## Overview

This repository takes the standard MNIST digits and composes them into cluttered 224×224 scenes: each image contains 3–6 individual digits, randomly rotated, resized, and placed on a jittered 3×3 grid. For every generated image a CSV row records the number of digits and, for each digit, its class label, center coordinate, pixel size, and rotation angle. This turns single-digit MNIST into a small object-detection / multi-label classification problem.

Note: this repo contains only the **dataset and the generation scripts** — there is no model, training, or evaluation code here. It is a data-preparation utility, not a trained detector.

## Requirements

Python 3 with the following packages (inferred from the source imports; there is no `requirements.txt`):

```bash
pip install numpy opencv-python imutils tqdm wget
```

`pickle`, `gzip`, `os`, and `random` are from the standard library. `wget` is only needed if you run the optional download step below.

## Repository layout

```
mnist/
├── mnist.pkl.gz                 # bundled MNIST pickle (train/val/test)
├── download_and_save_mnist.py   # optional: export individual 28x28 digit JPGs
├── multi_digits.py              # main: build multi-digit 224x224 images + labels CSV
├── single_images/               # pre-exported single digits
│   ├── train/                   # 48,647 JPGs, named {label}_{index}.jpg
│   └── test/                    # 10,000 JPGs
└── multi_mnist/                 # pre-generated multi-digit dataset
    ├── train_data.csv  + train_multi_imgs/   # 11,103 images
    ├── val_data.csv    + val_multi_imgs/     #  2,230 images
    └── small_data.csv  + small_multi_imgs/   #      5 images (tiny sample)
```

## Usage

Both scripts read `mnist.pkl.gz` from the current directory, so run them from inside the `mnist/` folder.

**1. (Optional) Export single-digit images.**

```bash
python download_and_save_mnist.py
```

This writes each MNIST digit as a `{label}_{index}.jpg` into `train/`, `val/`, and `test/`. It first tries to download `mnist.pkl.gz` from `http://deeplearning.net/data/mnist/mnist.pkl.gz`. **That URL is no longer online**, but `mnist.pkl.gz` is already committed in this repo, so you can skip this step (and comment out the `wget.download(...)` line if you only want the export).

**2. Generate the multi-digit dataset.**

```bash
mkdir -p val_multi_imgs      # cv2.imwrite does not create the output dir
python multi_digits.py
```

This produces the composite 224×224 images in `val_multi_imgs/` and their labels in `val_data.csv`.

As committed, `multi_digits.py` iterates over the MNIST **validation** split and writes to `val_data.csv` / `val_multi_imgs/`. To regenerate the other splits, edit the script to read `training_data` (or `test_data`) and to point the output CSV / image directory at `train_multi_imgs/` etc., then rerun. (The `train_multi_imgs`, `val_multi_imgs`, and `small_multi_imgs` folders in the repo were produced this way.)

### How each image is built

For every canvas, the generator (`multi_digits.py`):

- picks a random digit count `num = randint(3, 6)`;
- chooses `num` of the 9 cells in a 3×3 grid and, per cell, sets a center at `74·col + jitter` / `74·row + jitter` where `jitter = 37 + randint(-25, 25)`;
- rotates each digit by `randint(-15, 15)` degrees and resizes it to a random square side in `[28, 56]` px;
- composites the digits onto a single 224×224 canvas (additive blend) and saves the result.

### Label format

Image filenames encode the ground truth: `{index}_{numDigits}_[{labels joined by -}].jpg`, e.g. `1_5_[5-0-4-1-9].jpg` is image 1 with 5 digits labeled 5, 0, 4, 1, 9.

Each row of the `*_data.csv` files is comma-separated, in this order:

1. `filename`
2. `count` — number of digits in the image (3–6)
3. `count` **class labels** (digit values, in placement order)
4. `count` **center coordinates**, each formatted `x-y` (pixel position on the 224×224 canvas)
5. `count` **sizes** — side length in pixels of each resized digit (28–56)
6. `count` **rotation angles** in degrees (−15 to 15)

Example row:

```
1_5_[5-0-4-1-9].jpg,5,5,0,4,1,9,26-97,198-162,198-34,199-89,33-37,30,36,29,43,36,0,-11,9,-3,-7
```

## Example generated samples

Composite 224×224 images from `mnist/multi_mnist/small_multi_imgs/`:

![Sample: 3 digits [2-1-3]](<mnist/multi_mnist/small_multi_imgs/2_3_[2-1-3].jpg>)
![Sample: 3 digits [5-3-6]](<mnist/multi_mnist/small_multi_imgs/4_3_[5-3-6].jpg>)
![Sample: 5 digits [5-0-4-1-9]](<mnist/multi_mnist/small_multi_imgs/1_5_[5-0-4-1-9].jpg>)

## Dataset

Built from the classic **MNIST** handwritten-digit dataset. The pickled source (`mnist/mnist.pkl.gz`) is bundled in the repository; the original `deeplearning.net` mirror referenced in the download script is no longer available.

## License

Released under the [MIT License](LICENSE).
