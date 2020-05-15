import tensorflow as tf

from utils.utils import image_normalization


class TFLiteModel:

    def __init__(self, path):
        interpreter = tf.lite.Interpreter(str(path))
        interpreter.allocate_tensors()
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        self.interpreter = interpreter
        self.input_shape = self.input_details[0]['shape']
        self._ntop = 3

    def predict(self, img):
        # if type(img) != tf.Tensor:
        #     img = tf.convert_to_tensor(img)
        input_data = tf.image.resize(img, (self.input_shape[2], self.input_shape[1]))
        input_data = tf.cast(image_normalization(input_data), tf.dtypes.uint8)
        input_data = tf.reshape(input_data, self.input_shape)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        scores = self.interpreter.get_tensor(self.output_details[1]['index'])
        return boxes[0], scores[0]

    def get_bounding_box(self, img: tf.Tensor, min_square_percent: float):
        if len(img.shape) == 3:
            h, w = img.shape[:-1]
        elif len(img.shape) == 4:
            h, w = img.shape[1:-1]
        else:
            raise ValueError(f"Invalid img shape: {img.shape}")
        img_square = h * w
        glob_top = h + 1
        glob_bot = -1
        glob_left = w + 1
        glob_right = -1
        glob_present = False
        boxes, _ = self.predict(img)
        boxes = sorted(boxes, key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]), reverse=True)
        for idx, coords in enumerate(boxes[:self._ntop]):
            top, left, bottom, right = coords
            top = h * top
            bottom = h * bottom
            left = w * left
            right = w * right
            glob_present = True
            glob_top = max(min(top, glob_top), 0)
            glob_bot = min(max(bottom, glob_bot), h)
            glob_left = max(min([0, left, glob_left]), 0)
            glob_right = min(max(right, glob_right), w)
        if glob_present:
            glob_x, glob_y = glob_left, glob_top
            glob_width = glob_right - glob_left
            glob_height = glob_bot - glob_top
            sq_percent = glob_width * glob_height / img_square
            if sq_percent >= min_square_percent:
                return glob_x, glob_y, glob_width, glob_height
        return None
