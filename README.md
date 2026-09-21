# SyDRA

Official PyTorch implementation of SyDRA: *Dual-Stream Synergistic Learning
for Interference-Robust Anomaly Localization on Structure-Dominant Industrial
Surfaces*.

Paper: [10.1109/TII.2026.3707644](https://doi.org/10.1109/TII.2026.3707644)

## Contents

- `sydra_model.py`: reconstructive and discriminative streams, including SMDR
  components MGFD and MCCC, and DISA.
- `data_loader_sydra.py`: image enumeration, mask matching, and synthetic
  anomaly generation.
- `train_sydra.py`: training and checkpoint export.
- `evaluate_sydra.py`: image-level, pixel-level, PRO, and boundary-ring metrics.
- `loss.py` and `perlin.py`: training utilities.
- `splits/absdd/`: fixed ABSDD train and test lists.

## Environment

The reference environment is Python 3.8 with PyTorch 1.8.0, torchvision 0.9.0,
and CUDA 10.2. Install the CUDA-enabled PyTorch pair first, then install the
remaining packages:

```bash
conda create -n sydra python=3.8
conda activate sydra
conda install pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Datasets

Download the public datasets from their source pages:

- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/), used for synthetic texture anomalies
- [Magnetic Tile Surface Defect Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets)

ABSDD is not redistributed with this repository due to data-use restrictions.
Users with authorized access can use the provided train/test split files.

Category layout, such as MVTec AD:

```text
DATASET/category/train/good/000.png
DATASET/category/test/good/001.png
DATASET/category/test/crack/002.png
DATASET/category/ground_truth/crack/002_mask.png
```

Direct layout, such as Magnetic Tile or ABSDD:

```text
DATASET/train/good/normal.png
DATASET/train/defect/example.png
DATASET/test/good/normal.png
DATASET/test/defect/example.png
```

The category-layout command expects `--data_path` to contain category
directories. The direct-layout command expects `--data_path/train` and
`--data_path/test` directly. Normal folders are named `good`, `free`, or
`MT_Free`. MVTec masks use `ground_truth/<defect>/<stem>_mask.png`.
Masks are excluded from image enumeration, and missing defect masks are
reported as errors.

The anomaly source must contain texture images in subdirectories:

```text
SOURCE/class_name/image.jpg
```

## Training configurations

Category-layout dataset:

```bash
python train_sydra.py --gpu_id 0 --dataset_layout category \
  --data_path ./datasets/mvtecd --classes tile \
  --anomaly_source_path ./datasets/dtd/images \
  --checkpoint_path ./checkpoints/tile \
  --lr 0.0001 --bs 8 --effective_bs 16 --epochs 400 \
  --seeds 2025 3407 --ssim_weight 2 --drop_last false
```

Magnetic Tile:

```bash
python train_sydra.py --gpu_id 0 --dataset_layout direct \
  --data_path ./datasets/Magnetic-Tile --classes magnetic_tile \
  --anomaly_source_path ./datasets/dtd/images \
  --checkpoint_path ./checkpoints/magnetic_tile \
  --lr 0.0001 --bs 8 --effective_bs 32 --epochs 600 \
  --seeds 3407 2026 --ssim_weight 4 --drop_last true
```

ABSDD with the included labeled split:

```bash
python train_sydra.py --gpu_id 0 --dataset_layout direct \
  --data_path ./datasets/ABSDD --classes absdd \
  --train_split ./splits/absdd/train_5pct.txt \
  --anomaly_source_path ./datasets/dtd/images \
  --checkpoint_path ./checkpoints/absdd \
  --lr 0.0001 --bs 8 --effective_bs 32 --epochs 600 \
  --seeds 2025 --ssim_weight 2 --drop_last true
```

For normal-only training, add `--normal_only`. Use the same run name and epoch
for the reconstruction and segmentation checkpoint pair.

## Evaluation

```bash
python evaluate_sydra.py --gpu_id 0 \
  --data_path ./datasets/mvtecd/tile \
  --recon_checkpoint ./checkpoints/tile/SyDRA_tile_effbs16_lr0.0001_seed2025_recon_ep400.pckl \
  --seg_checkpoint ./checkpoints/tile/SyDRA_tile_effbs16_lr0.0001_seed2025_seg_ep400.pckl \
  --sigma 2 --output ./outputs/tile.json
```

Use `--sigma 4` for Magnetic Tile and ABSDD. For ABSDD, add:

```bash
--test_split ./splits/absdd/test.txt --boundary_f1 --boundary_sigma 1
```

F1-max is computed over the precision-recall thresholds. PRO and Boundary-F1
use 100 thresholds, with PRO integrated up to a false-positive rate of 0.3.
Boundary-F1 is the maximum pixel F1 within a five-iteration 3 x 3
dilation/erosion ring around each nonempty ground-truth mask. Evaluation JSON
contains metrics and protocol settings without local checkpoint paths or
image-file listings.

## Citation

```bibtex
@article{zhang2026sydra,
  title   = {SyDRA: Dual-Stream Synergistic Learning for Interference-Robust Anomaly Localization on Structure-Dominant Industrial Surfaces},
  author  = {Zhang, Yiqiong and Wang, Xueping and Ma, Yunfeng and He, Yingmei and Jiang, Shuai and Wang, Yaonan and Liu, Min},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2026},
  doi     = {10.1109/TII.2026.3707644}
}
```

## Acknowledgements

The implementation uses components derived from the DRAEM codebase and includes
an adapted Focal Loss implementation. Provenance and license information for
these components is provided in `THIRD_PARTY_NOTICES.md`; the corresponding
license texts are included in `LICENSES/`. Please also cite DRAEM when using
the shared reconstruction and synthetic-anomaly components.

Unless otherwise noted, the original SyDRA code in this repository is released
under the MIT License. Third-party components retain their respective licenses.
