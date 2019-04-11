import os
import tensorflow as tf
from preprocessing import tf_image
from tensorflow.python.ops import random_ops


def preprocessing_training(image, mask, out_shape, prob=0.5):
    with tf.name_scope('preprocessing_training'):
        with tf.name_scope('rotate'):
            rnd = tf.random_uniform((), minval=0, maxval=1, name='rotate')

            def rotate():
                k = random_ops.random_uniform([], 0, 10000)
                k = tf.cast(k, tf.int32)
                return tf.image.rot90(image, k=k), tf.image.rot90(mask, k=k)

            def no_rotate():
                return image, mask
            image, mask = tf.cond(tf.less(rnd, prob), rotate, no_rotate)
        with tf.name_scope('flip_left_right'):
            def flip_left_right():
                return tf.image.flip_left_right(image), tf.image.flip_left_right(mask)

            def no_flip_left_right():
                return image, mask
            rnd = tf.random_uniform((), minval=0, maxval=1, name='flip_left_right')
            image, mask = tf.cond(tf.less(rnd, prob), flip_left_right, no_flip_left_right)

        with tf.name_scope('flip_up_down'):
            def flip_up_down():
                return tf.image.flip_up_down(image), tf.image.flip_up_down(mask)

            def no_flip_up_down():
                return image, mask

            rnd = tf.random_uniform((), minval=0, maxval=1, name='flip_up_down')
            image, mask = tf.cond(tf.less(rnd, prob), flip_up_down, no_flip_up_down)
        image = tf_image.resize_image(image, out_shape,
                                      method=tf.image.ResizeMethod.BILINEAR,
                                      align_corners=False)
        mask = tf_image.resize_image(mask, out_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                     align_corners=False)
    return image, mask


def preprocessing_val(image, out_shape):
    image = tf_image.resize_image(image, out_shape,
                                  method=tf.image.ResizeMethod.BILINEAR,
                                  align_corners=False)
    return image


def segmentation_preprocessing(image, mask, out_shape, is_training):
    if is_training:
        return preprocessing_training(image, mask, out_shape)
    else:
        return preprocessing_val(image, out_shape)