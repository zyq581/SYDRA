import glob
import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from PIL import Image
from perlin import rand_perlin_2d_np
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
NON_PNG_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ANNOTATION_DIR_NAMES = {
    "ground_truth",
    "groundtruth",
    "gt",
    "mask",
    "masks",
    "label",
    "labels",
    "annotation",
    "annotations",
}
NORMAL_DIR_TOKENS = ("good", "mt_free", "free")


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _path_parts(path):
    parts = []
    while True:
        path, tail = os.path.split(path)
        if tail:
            parts.append(tail.lower())
        else:
            if path:
                parts.append(path.lower())
            break
    return parts


def _is_annotation_dir(path):
    return any(part in ANNOTATION_DIR_NAMES for part in _path_parts(path))


def _is_image_file(path):
    return path.lower().endswith(IMAGE_EXTENSIONS)


def _has_non_png_sibling(path):
    stem, _ = os.path.splitext(path)
    return any(os.path.exists(stem + ext) for ext in NON_PNG_IMAGE_EXTENSIONS)


def _is_mask_file(path):
    lower = path.lower()
    if lower.endswith("_mask.png"):
        return True
    if _is_annotation_dir(os.path.dirname(path)):
        return True
    # ABSDD and Magnetic-Tile store masks as same-basename PNG files next to
    # BMP/JPG inputs. In MVTec, real inputs can be PNG, but they do not have a
    # non-PNG sibling in the same test folder.
    if lower.endswith(".png") and _has_non_png_sibling(path):
        return True
    return False


def _is_input_image(path):
    return _is_image_file(path) and not _is_mask_file(path)


def _is_normal_folder(path, root_dir):
    # The normal folders in the supported datasets include MVTec "good" and
    # Magnetic-Tile "MT_Free"; some prepared splits also use "free".
    relative = os.path.relpath(path, root_dir)
    category = relative.replace("\\", "/").split("/")[0].lower()
    return category in NORMAL_DIR_TOKENS


def _replace_path_part(path, old, new):
    parts = []
    current = path
    while True:
        current, tail = os.path.split(current)
        if tail:
            parts.append(tail)
        else:
            if current:
                parts.append(current)
            break
    parts = list(reversed(parts))
    replaced = False
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        if part.lower() == old.lower():
            parts[i] = new
            replaced = True
            break
    if not replaced:
        return None
    rebuilt = parts[0]
    for part in parts[1:]:
        rebuilt = os.path.join(rebuilt, part)
    return rebuilt


def _mask_candidates_for_image(img_path):
    dir_path, file_name = os.path.split(img_path)
    stem = os.path.splitext(file_name)[0]
    candidates = []

    # MVTec: category/test/defect/xxx.png -> category/ground_truth/defect/xxx_mask.png
    gt_dir = _replace_path_part(dir_path, "test", "ground_truth")
    if gt_dir is not None:
        candidates.append(os.path.join(gt_dir, stem + "_mask.png"))
        candidates.append(os.path.join(gt_dir, stem + ".png"))
    train_gt_dir = _replace_path_part(dir_path, "train", "ground_truth")
    if train_gt_dir is not None:
        candidates.append(os.path.join(train_gt_dir, stem + "_mask.png"))

    # Common local-mask layouts.
    candidates.append(os.path.join(dir_path, stem + "_mask.png"))
    candidates.append(os.path.join(dir_path, stem + ".png"))

    # Optional sibling mask directories.
    parent = os.path.dirname(dir_path)
    defect_dir = os.path.basename(dir_path)
    for folder in ("ground_truth", "masks", "mask", "labels", "label"):
        candidates.append(os.path.join(parent, folder, defect_dir, stem + "_mask.png"))
        candidates.append(os.path.join(parent, folder, defect_dir, stem + ".png"))
        candidates.append(os.path.join(parent, folder, stem + "_mask.png"))
        candidates.append(os.path.join(parent, folder, stem + ".png"))

    seen = set()
    unique = []
    for candidate in candidates:
        key = _norm(candidate)
        if key == _norm(img_path) or key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def find_mask_path(img_path):
    for candidate in _mask_candidates_for_image(img_path):
        if os.path.exists(candidate):
            return candidate
    return None


def _collect_input_images(root_dir):
    images = []
    for root, _, files in os.walk(root_dir):
        if _is_annotation_dir(root):
            continue
        for file_name in files:
            path = os.path.join(root, file_name)
            if _is_input_image(path):
                images.append(path)
    return sorted(images)


def _select_input_images(root_dir, split_file=None):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset split directory not found: {root_dir}")
    if split_file is None:
        return _collect_input_images(root_dir)
    images, seen = [], set()
    root = os.path.abspath(root_dir)
    with open(split_file, encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            relative = line.strip()
            if not relative or relative.startswith("#"):
                continue
            path = os.path.normpath(os.path.join(root, relative.replace("\\", "/")))
            key = _norm(path)
            if os.path.commonpath([_norm(root), key]) != _norm(root) or key in seen:
                raise ValueError(f"Invalid or duplicate split entry at {split_file}:{line_number}")
            if not os.path.isfile(path) or not _is_input_image(path):
                raise ValueError(f"Split entry is missing or is an annotation: {path}")
            seen.add(key)
            images.append(path)
    return sorted(images)


def _require_mask(img_path):
    mask_path = find_mask_path(img_path)
    if mask_path is None:
        raise FileNotFoundError(f"Missing defect mask for input image: {img_path}")
    return mask_path


class TestDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None, split_file=None):
        self.root_dir = root_dir
        self.resize_shape = resize_shape or [256, 256]
        self.images = _select_input_images(root_dir, split_file)
        for path in self.images:
            if not _is_normal_folder(os.path.dirname(path), root_dir):
                _require_mask(path)
        print(f"[DEBUG] Found {len(self.images)} input images in {root_dir}")

        self.transform_img = transforms.Compose([
            transforms.Resize((self.resize_shape[0], self.resize_shape[1])),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(
                (self.resize_shape[0], self.resize_shape[1]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, _ = os.path.split(img_path)
        is_normal = _is_normal_folder(dir_path, self.root_dir)

        image = Image.open(img_path).convert("RGB")

        if is_normal:
            has_anomaly = torch.tensor([0], dtype=torch.float32)
            mask = Image.new("L", image.size, color=0)
        else:
            has_anomaly = torch.tensor([1], dtype=torch.float32)
            mask = Image.open(_require_mask(img_path)).convert("L")

        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        return {"image": image, "has_anomaly": has_anomaly, "mask": mask, "idx": idx}


class TrainDataset(Dataset):
    def __init__(self, root_dir, anomaly_source_path, resize_shape=(256, 256), rotate_good=True,
                 split_file=None, normal_only=False):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.rotate_good = rotate_good
        self.image_paths = _select_input_images(root_dir, split_file)
        if normal_only:
            self.image_paths = [p for p in self.image_paths
                                if _is_normal_folder(os.path.dirname(p), root_dir)]
        for path in self.image_paths:
            if not _is_normal_folder(os.path.dirname(path), root_dir):
                _require_mask(path)
        if len(self.image_paths) == 0:
            print(f"[ERROR] No training images found in {root_dir}. Check path and extensions.")
        else:
            print(f"[DEBUG] Found {len(self.image_paths)} training input images in {root_dir}")

        self.anomaly_source_paths = sorted(glob.glob(os.path.join(anomaly_source_path, "*", "*.*")))
        self.anomaly_source_paths = [p for p in self.anomaly_source_paths if _is_image_file(p)]
        if not self.anomaly_source_paths:
            raise ValueError("No anomaly texture images found; expected SOURCE/class/image.*. "
                             "The hybrid generator requires its texture source.")
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90), mode="edge")])
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        return iaa.Sequential([
            self.augmenters[aug_ind[0]],
            self.augmenters[aug_ind[1]],
            self.augmenters[aug_ind[2]],
        ])

    def rotate_good_image(self, image):
        if self.rotate_good and np.random.rand() > 0.5:
            seq = self.rot
            image_uint8 = (image * 255).astype(np.uint8)
            image_rot = seq(image=image_uint8).astype(np.float32) / 255.0
            return image_rot
        return image

    def generate_cutpaste_anomaly(self, image):
        h, w = image.shape[:2]
        augmented_image = image.copy()
        mask = np.zeros((h, w, 1), dtype=np.float32)

        anomaly_type = "scar" if np.random.rand() > 0.6 else "block"
        if anomaly_type == "scar":
            patch_w = np.random.randint(2, max(3, w // 50))
            patch_h = np.random.randint(15, 60)
        else:
            patch_w = np.random.randint(max(1, w // 30), max(2, w // 10))
            patch_h = np.random.randint(max(1, h // 30), max(2, h // 10))

        patch_w = min(patch_w, w - 1)
        patch_h = min(patch_h, h - 1)
        x = np.random.randint(0, w - patch_w)
        y = np.random.randint(0, h - patch_h)
        patch = image[y:y + patch_h, x:x + patch_w].copy()

        seq = iaa.Sequential([
            iaa.Affine(rotate=(-45, 45), mode="edge"),
            iaa.GammaContrast((0.3, 2.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        ])
        patch_aug = seq(image=(patch * 255).astype(np.uint8)).astype(np.float32) / 255.0

        x_new = np.random.randint(0, w - patch_w)
        y_new = np.random.randint(0, h - patch_h)
        patch_aug_resized = cv2.resize(patch_aug, (patch_w, patch_h))
        patch_mask = np.ones((patch_h, patch_w), dtype=np.float32)

        k_size = min(patch_h, patch_w)
        if k_size % 2 == 0:
            k_size -= 1
        k_size = max(3, min(k_size, 9))
        patch_mask_blur = cv2.GaussianBlur(patch_mask, (k_size, k_size), 0)
        if len(image.shape) == 3:
            patch_mask_blur = np.expand_dims(patch_mask_blur, axis=2)

        roi = augmented_image[y_new:y_new + patch_h, x_new:x_new + patch_w]
        blended_patch = patch_aug_resized * patch_mask_blur + roi * (1.0 - patch_mask_blur)
        augmented_image[y_new:y_new + patch_h, x_new:x_new + patch_w] = blended_patch
        mask[y_new:y_new + patch_h, x_new:x_new + patch_w] = 1.0
        return augmented_image, mask

    def augment_image(self, image, mask_path=None):
        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {mask_path}")
            mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=2)

            if np.random.rand() > 0.5:
                seq_det = self.rot.to_deterministic()
                image_aug = seq_det.augment_image((image * 255).astype(np.uint8))
                mask_aug = seq_det.augment_image((mask * 255).astype(np.uint8))
                image = image_aug.astype(np.float32) / 255.0
                mask = (mask_aug > 127).astype(np.float32)

            augmented_image = image.copy()
            has_anomaly = 1.0 if mask.sum() > 0 else 0.0
            return image, augmented_image, mask, np.array([has_anomaly], dtype=np.float32)

        use_cutpaste = (np.random.rand() > 0.5) or (len(self.anomaly_source_paths) == 0)
        if use_cutpaste:
            augmented_image, mask = self.generate_cutpaste_anomaly(image)
            return image, augmented_image, mask, np.array([1.0], dtype=np.float32)

        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        src_path = np.random.choice(self.anomaly_source_paths)
        anomaly_img = cv2.imread(src_path)
        if anomaly_img is None:
            raise ValueError(f"Failed to read anomaly texture: {src_path}")

        anomaly_img = cv2.cvtColor(anomaly_img, cv2.COLOR_BGR2RGB)
        anomaly_img = cv2.resize(anomaly_img, (self.resize_shape[1], self.resize_shape[0]))
        anomaly_img_aug = aug(image=anomaly_img)
        perlin_scalex = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** np.random.randint(min_perlin_scale, perlin_scale)
        perlin_noise = rand_perlin_2d_np(
            (self.resize_shape[0], self.resize_shape[1]),
            (perlin_scalex, perlin_scaley),
        )
        perlin_thr = (perlin_noise > 0.5).astype(np.float32)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        img_thr = anomaly_img_aug.astype(np.float32) * perlin_thr / 255.0
        beta = np.random.rand() * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr

        if np.random.rand() > 0.5:
            return image, image.copy(), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)

        mask = perlin_thr.astype(np.float32)
        augmented_image = mask * augmented_image + (1 - mask) * image
        has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
        return image, augmented_image.astype(np.float32), mask, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, mask_path=None):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        image = image.astype(np.float32) / 255.0

        if mask_path is None:
            image = self.rotate_good_image(image)

        image, augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, mask_path)
        image = np.transpose(image, (2, 0, 1))
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        if anomaly_mask.ndim == 2:
            anomaly_mask = np.expand_dims(anomaly_mask, axis=2)
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        dir_path, _ = os.path.split(img_path)
        is_normal = _is_normal_folder(dir_path, self.root_dir)

        if is_normal:
            image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(img_path, None)
            is_labeled = np.array([0], dtype=np.float32)
        else:
            mask_path = _require_mask(img_path)
            image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(img_path, mask_path)
            is_labeled = np.array([1], dtype=np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "augmented_image": torch.tensor(augmented_image, dtype=torch.float32),
            "anomaly_mask": torch.tensor(anomaly_mask, dtype=torch.float32),
            "has_anomaly": torch.tensor(has_anomaly, dtype=torch.float32),
            "is_labeled": torch.tensor(is_labeled, dtype=torch.float32),
            "idx": idx,
        }
