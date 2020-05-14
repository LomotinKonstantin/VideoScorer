import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error, classification_report


def load_bmp(path: str, width=None, height=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_bmp(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if width is not None and height is not None:
        height = tf.cast(height, tf.int32)
        width = tf.cast(width, tf.int32)
        img = tf.image.resize(img, [height, width])
    return img


def count_lines(fpath):
    return sum(1 for _ in open(fpath))


def ds_len(ds):
    return sum(1 for _ in ds)


def oh_dist_from_fname(fname, depth):
    parts = tf.strings.split(fname, "_")
    int_dist = tf.strings.to_number(parts[1], tf.dtypes.int64)
    return tf.one_hot(int_dist - 1, depth)


def prepare_for_training(ds, batch_size=32, cache=False, shuffle_buffer_size=100, repeat=False):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    if repeat:
        ds = ds.repeat()
        
    if batch_size > 0:
        ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

    
def image_normalization(image: tf.Tensor, new_min=0, new_max=255) -> tf.Tensor:
    original_dtype = image.dtype
    new_min = tf.constant(new_min, dtype=tf.float32)
    new_max = tf.constant(new_max, dtype=tf.float32)
    image_min = tf.cast(tf.reduce_min(image), tf.float32)
    image_max = tf.cast(tf.reduce_max(image), tf.float32)
    image = tf.cast(image, tf.float32)

    normalized_image = (new_max - new_min) / (image_max - image_min) * (image - image_min) + new_min
    return tf.cast(normalized_image, original_dtype)


def image_shape(image: tf.Tensor, dtype=tf.int32) -> tf.Tensor:
    shape = tf.shape(image)
    shape = shape[:2] if image.get_shape().ndims == 3 else shape[1:3]
    return tf.cast(shape, dtype)


def gaussian_filter(image: tf.Tensor, kernel_size: int, sigma: float, dtype=tf.float32) -> tf.Tensor:
    kernel = gaussian_kernel2d(kernel_size, sigma)
    if image.get_shape().ndims == 3:
        image = image[tf.newaxis, :, :, :]
    image = tf.cast(image, tf.float32)
    image = tf.nn.conv2d(image, kernel[:, :, tf.newaxis, tf.newaxis], strides=1, padding='SAME')
    return tf.cast(image, dtype)


def low_pass_filter(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.rgb_to_grayscale(image)
    image_low = gaussian_filter(image, 16, 7 / 6)
    image_low = rescale(image_low, 1 / 4, method=tf.image.ResizeMethod.BICUBIC)
    image_low = tf.image.resize(image_low, size=image_shape(image), method=tf.image.ResizeMethod.BICUBIC)
    return image - tf.cast(image_low, image.dtype)
    

def gaussian_kernel2d(kernel_size: int, sigma: float, dtype=tf.float32) -> tf.Tensor:
    _range = tf.range(kernel_size)
    x, y = tf.meshgrid(_range, _range)
    constant = tf.cast(tf.round(kernel_size / 2), dtype=dtype)
    x = tf.cast(x, dtype=dtype) - constant
    y = tf.cast(y, dtype=dtype) - constant
    kernel = 1 / (2 * np.pi * sigma ** 2) * tf.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(kernel)


def normalize_kernel(kernel: tf.Tensor) -> tf.Tensor:
    return kernel / tf.reduce_sum(kernel)


def scale_shape(image: tf.Tensor, scale: float) -> tf.Tensor:
    shape = image_shape(image, tf.float32)
    shape = tf.math.ceil(shape * scale)
    return tf.cast(shape, tf.int32)


def rescale(image: tf.Tensor, scale: float, dtype=tf.float32, **kwargs) -> tf.Tensor:
    assert image.get_shape().ndims in (3, 4), 'The tensor must be of dimension 3 or 4'

    image = tf.cast(image, tf.float32)
    rescale_size = scale_shape(image, scale)
    rescaled_image = tf.image.resize(image, size=rescale_size, **kwargs)
    return tf.cast(rescaled_image, dtype)


def reference_from_fname(fname):
    return tf.strings.join([tf.strings.split(fname, "_")[0], ".bmp"])


def error_map(reference: tf.Tensor, distorted: tf.Tensor, p: float = 0.2) -> tf.Tensor:
    return tf.pow(tf.abs(reference - distorted), p)


def reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    return 2 / (1 + tf.exp(- alpha * tf.abs(distorted))) - 1


def average_reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    r = reliability_map(distorted, alpha)
    return r / tf.reduce_mean(r)


def test_model(model, test_ds):
    res = list(map(lambda line: (model.predict(low_pass_filter(line[1])).ravel(), line[2].numpy()), test_ds))
    y_true, y_pred = zip(*res)
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    spearman_corr = spearmanr(y_true, y_pred)
    if type(spearman_corr) != tuple:
        spearman_corr = (spearman_corr.correlation, spearman_corr.pvalue)
    metrics = {
        "spearman": spearman_corr,
        "pearson": pearsonr(y_true, y_pred),
        "kendall": kendalltau(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred), 
    }
    return metrics


def test_clf(model, test_ds):
    res = list(map(lambda line: (model.predict(low_pass_filter(line[1])).ravel().argmax(),
                                 line[3].numpy().argmax()), test_ds))
    y_true, y_pred = zip(*res)
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    return classification_report(y_true, y_pred, digits=4)


def test_model_on_patches(model, test_ds, mos_weights):
    y_true = []
    y_pred = []
    for ref, img_patches, true_mos, dist_type in test_ds:
        pred_mos_batch = model.predict_on_batch(low_pass_filter(img_patches))
        pred_mos_batch = tf.reshape(pred_mos_batch, [-1])   # Flatten in tf way
        assert len(mos_weights) == len(pred_mos_batch) == len(true_mos), \
            f"{mos_weights}, {pred_mos_batch}, {true_mos}"
        assert (tf.abs(tf.reduce_mean(true_mos) - true_mos[0])) <= 1e-6      # All are the same
        res_img_mos = np.average(pred_mos_batch.numpy(), weights=mos_weights)
        y_true.append(true_mos[0].numpy())
        y_pred.append(res_img_mos)
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    spearman_corr = spearmanr(y_true, y_pred)
    if type(spearman_corr) != tuple:
        spearman_corr = (spearman_corr.correlation, spearman_corr.pvalue)
    metrics = {
        "spearman": spearman_corr,
        "pearson": pearsonr(y_true, y_pred),
        "kendall": kendalltau(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
    }
    return metrics


def preview_images(dataset, n=6):
    _, imgs, mos, _ = next(iter(dataset))
    n_rows = int(np.sqrt(n))
    n_cols = n // n_rows
    fig = plt.figure(figsize=(12, 6))
    for i, (image, m) in enumerate(zip(imgs[:n], mos[:n])):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(image)
        ax.title.set_text(f"{m}")
        ax.axis('off')
    fig.tight_layout()


def demo_low_pass_filter(img):
    n_rows = 1
    n_cols = 2
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    ax1.imshow(img)
    ax2 = fig.add_subplot(n_rows, n_cols, 2)
    filtered = low_pass_filter(img)
    filtered = tf.image.grayscale_to_rgb(filtered)
    filtered = tf.cast(image_normalization(filtered), tf.dtypes.int64)
    ax2.imshow(filtered[0])
    ax1.axis('off')
    ax2.axis('off')
    fig.tight_layout()


def demo_err_map(ref, img):
    n_rows = 1
    n_cols = 3
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    ax1.imshow(img)
    ax2 = fig.add_subplot(n_rows, n_cols, 3)
    lp_img = low_pass_filter(img)
    err_map = error_map(ref, img)
    ax2.imshow(err_map)
    ax3 = fig.add_subplot(n_rows, n_cols, 2)
    lp_img = tf.image.grayscale_to_rgb(lp_img[0])
    lp_img = tf.cast(image_normalization(lp_img), tf.dtypes.int64)
    ax3.imshow(lp_img)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis("off")
    fig.tight_layout()

    
def demo_reliability_maps(img, alpha):
    n_rows, n_cols = 1, 3
    fig = plt.figure(figsize=(14, 6))
    lp_img = low_pass_filter(img)
    #
    ax_img = fig.add_subplot(n_rows, n_cols, 1)
    ax_img.imshow(img)
    ax_img.title.set_text("distorted")
    #
    ax_rel = fig.add_subplot(n_rows, n_cols, 2)
    rel_map = reliability_map(lp_img, alpha)
    rel_map = image_normalization(tf.image.grayscale_to_rgb(rel_map), 0, 1)[0]
    ax_rel.imshow(rel_map)
    ax_rel.title.set_text("reliability map")
    #
    ax_avg_rel = fig.add_subplot(n_rows, n_cols, 3)
    avg_rel_map = average_reliability_map(lp_img, alpha)
    avg_rel_map = image_normalization(tf.image.grayscale_to_rgb(avg_rel_map), 0, 1)[0]
    ax_avg_rel.imshow(avg_rel_map)
    ax_avg_rel.title.set_text("avg reliability map")
    fig.tight_layout()
    for ax in [ax_img, ax_rel, ax_avg_rel]:
        ax.axis("off")


def is_iterable(val) -> bool:
    try:
        iter(val)
        return True
    except TypeError:
        return False


def dump_metrics_to_str(metrics: dict) -> str:
    res_str = ""
    for k, v in metrics.items():
        if is_iterable(v):
            val_string = f"{round(v[0], 3)} (p = {round(v[1], 3)})"
        else:
            val_string = str(round(v, 3))
        res_str += f"{k}: {val_string}\n"
    return res_str


def extract_patches(img: tf.Tensor, patch_size: int, patches_per_side: int, shape=None) -> tf.Tensor:
    assert patch_size > 0
    if shape is None:
        img_shape = tf.shape(img)
        h, w = img_shape[0], img_shape[1]
    else:
        h, w = shape
    patches_space = patch_size * patches_per_side
    w_strides = (w - patches_space) // (patches_per_side-1) + patch_size
    h_strides = (h - patches_space) // (patches_per_side-1) + patch_size
    patches = tf.image.extract_patches(images=[img],
                                       sizes=[1, patch_size, patch_size, 1],
                                       strides=[1, h_strides, w_strides, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')[0]
    patches = tf.reshape(patches, [-1, patch_size, patch_size, img.shape[-1]])
    return patches


def crop_img(img, bb):
    off_w, off_h, t_w, t_h = bb
    return tf.image.crop_to_bounding_box(img, offset_width=int(off_w), offset_height=int(off_h),
                                         target_height=int(t_h), target_width=int(t_w))
