import os
import numpy as np
import tensorflow.compat.v1 as tf1

tf1.disable_eager_execution()


class TransNetParams:
    F = 16
    L = 3
    S = 2
    D = 256
    INPUT_WIDTH = 48
    INPUT_HEIGHT = 27
    CHECKPOINT_PATH = None


class TransNet:

    def __init__(self, params: TransNetParams, session=None):

        self.params = params
        self.session = session or tf1.Session()
        self._build()
        self._restore()

    def _build(self):
        def shape_text(tensor):
            return ", ".join(["?" if i is None else str(i) for i in tensor.get_shape().as_list()])

        with self.session.graph.as_default():
            print("[TransNet] Creating ops.")
            with tf1.variable_scope("TransNet"):
                def conv3d(inp, filters, dilation_rate):
                    return tf1.keras.layers.Conv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
                                                   padding="SAME", activation=tf1.nn.relu, use_bias=True,
                                                   name="Conv3D_{:d}".format(dilation_rate))(inp)

                self.inputs = tf1.placeholder(tf1.uint8,
                                              shape=[None, None, self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3])
                net = tf1.cast(self.inputs, dtype=tf1.float32) / 255.
                print(" " * 10, "Input ({})".format(shape_text(net)))

                for idx_l in range(self.params.L):
                    with tf1.variable_scope("SDDCNN_{:d}".format(idx_l + 1)):
                        filters = (2 ** idx_l) * self.params.F
                        print(" " * 10, "SDDCNN_{:d}".format(idx_l + 1))

                        for idx_s in range(self.params.S):
                            with tf1.variable_scope("DDCNN_{:d}".format(idx_s + 1)):
                                net = tf1.identity(net)  # improves look of the graph in TensorBoard
                                conv1 = conv3d(net, filters, 1)
                                conv2 = conv3d(net, filters, 2)
                                conv3 = conv3d(net, filters, 4)
                                conv4 = conv3d(net, filters, 8)
                                net = tf1.concat([conv1, conv2, conv3, conv4], axis=4)
                                print(" " * 10, "> DDCNN_{:d} ({})".format(idx_s + 1, shape_text(net)))

                        net = tf1.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(net)
                        print(" " * 10, "MaxPool ({})".format(shape_text(net)))

                shape = [tf1.shape(net)[0], tf1.shape(net)[1], np.prod(net.get_shape().as_list()[2:])]
                net = tf1.reshape(net, shape=shape, name="flatten_3d")
                print(" " * 10, "Flatten ({})".format(shape_text(net)))
                net = tf1.keras.layers.Dense(self.params.D, activation=tf1.nn.relu)(net)
                print(" " * 10, "Dense ({})".format(shape_text(net)))

                self.logits = tf1.keras.layers.Dense(2, activation=None)(net)
                print(" " * 10, "Logits ({})".format(shape_text(self.logits)))
                self.predictions = tf1.nn.softmax(self.logits, name="predictions")[:, :, 1]
                print(" " * 10, "Predictions ({})".format(shape_text(self.predictions)))
                self.net = net

            print("[TransNet] Network built.")
            no_params = np.sum([int(np.prod(v.get_shape().as_list())) for v in tf1.trainable_variables()])
            print("[TransNet] Found {:d} trainable parameters.".format(no_params))

    def _restore(self):
        if self.params.CHECKPOINT_PATH is not None:
            saver = tf1.train.Saver([tf1.trainable_variables("TransNet")])
            saver.restore(self.session, self.params.CHECKPOINT_PATH)
            print("[TransNet] Parameters restored from '{}'.".format(os.path.basename(self.params.CHECKPOINT_PATH)))

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and \
               list(frames.shape[2:]) == [self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3], \
               "[TransNet] Input shape must be [batch, frames, height, width, 3]."
        return self.session.run(self.predictions, feed_dict={self.inputs: frames})

    def predict_video(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and \
               list(frames.shape[1:]) == [self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3], \
               "[TransNet] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (frames.shape[0] % 50 if frames.shape[0] % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out

        res = []
        for inp in input_iterator():
            pred = self.predict_raw(np.expand_dims(inp, 0))[0, 25:75]
            res.append(pred)
            print("\r[TransNet] Processing video frames {}/{}".format(
                min(len(res) * 50, len(frames)), len(frames)
            ), end="")
        print("")
        return np.concatenate(res)[:len(frames)]  # remove extra padded frames
