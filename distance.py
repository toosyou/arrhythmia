import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class MultiHeadDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, head_dim, mode, output_dim=None, window_size=3, distance_norm=False, max_distance=np.Inf, smooth_embedding_ratio=4, **kwargs):
        self.num_head = num_head
        self.head_dim = head_dim
        self.mode = mode
        self.output_dim = output_dim
        self.window_size = window_size
        self.distance_norm = distance_norm
        self.max_distance = max_distance
        self.smooth_embedding_ratio= smooth_embedding_ratio

        assert self.mode in ('global', 'local'), 'mode must be either global or local'
        super().__init__(**kwargs)

    def get_config(self):
        config = {
            'num_head': self.num_head,
            'head_dim': self.head_dim,
            'mode': self.mode,
            'output_dim': self.output_dim,
            'window_size': self.window_size,
            'distance_norm': self.distance_norm,
            'max_distance': self.max_distance,
            'smooth_embedding_ratio': self.smooth_embedding_ratio,
        }
        base_config = super().get_config()
        config.update(base_config)
        return config

    def build(self, input_shape):
        data_length, input_dim = input_shape[-2:] # (n_batch, data_length, feature_dim)

        self.max_distance = np.clip(self.max_distance, 1, data_length-1).astype(int)

        self.query_embedding    = tf.keras.layers.Dense(self.num_head * self.head_dim, use_bias=True)
        self.key_embedding      = tf.keras.layers.Dense(self.num_head * self.head_dim, use_bias=True)
        self.value_embedding    = tf.keras.layers.Dense(self.num_head, use_bias=False)

        self.distance_pe = self.add_weight(shape=(self.num_head, self.head_dim, (2 * self.max_distance + 1) // self.smooth_embedding_ratio, 1),
                                            initializer='GlorotNormal',
                                            trainable=True, name='distance_pe')
        self.u_pe        = self.add_weight(shape=(self.num_head, 1, 1, self.head_dim),
                                            initializer='GlorotNormal',
                                            trainable=True, name='u_pe')
        self.v_pe        = self.add_weight(shape=(self.num_head, 1, 1, self.head_dim),
                                            initializer='GlorotNormal',
                                            trainable=True, name='v_pe')

        if self.mode == 'local': # local mode
            self.output_embedding = SmoothEmbedding(self.output_dim, self.smooth_embedding_ratio)

    def call(self, inputs, return_attention=False):
        query, key, value = inputs, inputs, inputs

        # get sizes
        data_length = query.shape[1]
        
        # embedding
        query   = self.query_embedding(query)               # (?, data_length, num_head * head_dim)
        key     = self.key_embedding(key)                   # (?, data_length, num_head * head_dim)
        value   = K.sigmoid(self.value_embedding(value))    # (?, data_length, num_head)

        multi_head_query    = tf.concat(tf.split(query[None, ...], self.num_head, axis=3), axis=0)      # (num_head, ?, data_length, head_dim)
        multi_head_key      = tf.concat(tf.split(key[None, ...], self.num_head, axis=3), axis=0)        # (num_head, ?, data_length, head_dim)
        multi_head_value    = K.permute_dimensions(value, (2, 0, 1))                                    # (num_head, ?, data_length)
        
        # calculate distance attention
        attention = tf.matmul(multi_head_query + self.u_pe, multi_head_key, transpose_b=True) # (num_head, ?, data_length, data_length)

        # distance padding
        attention = tf.linalg.diag_part(attention, k=(-self.max_distance, self.max_distance)) # (num_head, ?, data_length, 2 * max_d + 1)
        attention = K.permute_dimensions(attention, (0, 1, 3, 2)) # transpose
        attention = K.reverse(attention, axes=(-2, -1))

        # relative positional encoding
        smooth_distance_pe = tf.image.resize(self.distance_pe, [self.head_dim, 2 * self.max_distance + 1], 'bilinear') # (num_head, head_dim, 2 * max_d + 1, 1)
        smooth_distance_pe = K.squeeze(K.expand_dims(smooth_distance_pe, axis=1), axis=-1) # (num_head, 1, head_dim, 2 * max_d + 1)

        attention = attention + tf.matmul(multi_head_query + self.v_pe, smooth_distance_pe)   # (num_head, ?, data_length, 2 * max_d + 1)
        
        attention = attention * (float(self.head_dim) ** -0.5)
        attention = tf.keras.layers.Softmax()(attention)

        attention = attention * multi_head_value[..., None]

        if self.distance_norm:
            # (num_head, ?, data_length, 2 * max_d + 1) -> (?, num_head * data_length, 2 * max_d + 1)
            attention = K.permute_dimensions(attention, (1, 0, 2, 3))
            attention = K.reshape(attention, (-1, self.num_head * data_length, 2 * self.max_distance + 1))

            attention = DistanceNorm()(attention)

            # (?, num_head * data_length, 2 * max_d + 1) -> (num_head, ?, data_length, 2 * max_d + 1)
            attention = K.reshape(attention, (-1, self.num_head, data_length, 2 * self.max_distance + 1))
            attention = K.permute_dimensions(attention, (1, 0, 2, 3))
        
        if self.mode == 'global':
            output = K.sum(attention, axis=2)                   # (num_head, ?, 2 * max_d + 1)
            output = K.permute_dimensions(output, (1, 2, 0))    # (?, 2 * max_d + 1, num_head)
        else:
            output = self.output_embedding(attention)                                       # (num_head, ?, data_length, output_dim)
            output = K.permute_dimensions(output, (1, 2, 3, 0))                             # (?, data_length, output_dim, num_head)
            output = K.reshape(output, (-1, data_length, self.output_dim * self.num_head))  # (?, data_length, output_dim * num_head)

        if return_attention:
            return output, K.permute_dimensions(attention, (1, 2, 3, 0)) # (?, data_length, 2 * max_d + 1, num_head)
        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.num_head]

class SmoothEmbedding(tf.keras.layers.Layer):
    def __init__(self, output_dim, downsample_ratio=4, **kwargs):
        self.output_dim = output_dim
        self.downsample_ratio = downsample_ratio
        super().__init__(**kwargs)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'downsample_ratio': self.downsample_ratio
        }
        base_config = super().get_config()
        config.update(base_config)
        return config

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.weight = self.add_weight(shape=(input_dim // self.downsample_ratio, self.output_dim, 1),
                                        initializer='GlorotNormal',
                                        trainable=True, name='weight')
        self.bias = self.add_weight(shape=(self.output_dim),
                                        initializer='zeros',
                                        trainable=True, name='bias')

    def call(self, inputs):
        '''inputs: (..., input_dim)
        '''
        input_dim = tf.shape(inputs)[-1]

        smooth_weight = tf.image.resize(self.weight, [input_dim, self.output_dim], 'bilinear')
        smooth_weight = tf.squeeze(smooth_weight, axis=[-1]) # (input_dim, output_dim)

        output = tf.matmul(inputs, smooth_weight) + self.bias # (..., output_dim)
        return output

class DistanceNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        max_distance = input_shape[-1] # N-1 + 1 + N
        self.range_max_distance = tf.range(max_distance, dtype=tf.float32) - max_distance // 2 + 1. # (max_distance): [-(N-1), -(N-2), ..., 0, ..., N]

    @staticmethod
    def interpolated_gather_nd(source, indices):
        original_shape = tf.shape(source)
        data_length = original_shape[-2]
        max_distance = original_shape[-1]

        paddings = [[0, 0]] * (tf.shape(source).shape[0] - 1) + [[1, 1]]
        source = tf.pad(source, paddings)

        indices = K.clip(indices + 1, 0, K.cast(max_distance, tf.float32)+1)
        integer_indices = K.cast(indices, tf.int32)

        floor_indices, ceil_indices = integer_indices, integer_indices + 1
        weights = indices - K.cast(floor_indices, tf.float32)

        floor_indices = K.expand_dims(K.expand_dims(floor_indices, axis=1), axis=-1)            # (-1, 1, max_distance, 1)
        ceil_indices = K.expand_dims(K.expand_dims(ceil_indices, axis=1), axis=-1)              # (-1, 1, max_distance, 1)

        floor_indices = tf.repeat(floor_indices, data_length, axis=1)                           # (-1, data_length, max_distance, 1)
        ceil_indices = tf.repeat(ceil_indices, data_length, axis=1)                             # (-1, data_length, max_distance, 1)

        weights = K.expand_dims(weights, axis=1)
        weights = tf.repeat(weights, data_length, axis=1)
    
        # interpolation
        normed_distance_floor = tf.gather_nd(source, floor_indices, batch_dims=2)
        normed_distance_ceil = tf.gather_nd(source, ceil_indices, batch_dims=2)
        normed_distance = normed_distance_ceil * weights + normed_distance_floor * (1. - weights)

        return normed_distance

    def get_mean_std(self, distance):
        px    = K.sum(distance, axis=-2)  # (-1, max_distance)
        px    = px / K.sum(px, axis=-1)[:, None]   # (-1, max_distance), normed

        mean  = K.sum(px * self.range_max_distance, axis=-1) # (-1)
        std   = K.sqrt(K.sum(px * K.pow(self.range_max_distance[None, :] - mean[:, None], 2), axis=-1)) # (-1)

        return mean, std

    def call(self, distance):
        '''distance: (..., data_length, max_distance)
        '''
        original_shape = tf.shape(distance)
        data_length = original_shape[-2]
        max_distance = original_shape[-1]

        distance = K.reshape(distance, (-1, data_length, max_distance))

        max_distance = K.cast(max_distance, dtype=tf.float32)

        mean, std = self.get_mean_std(distance) # (-1), (-1)
        new_indices = (self.range_max_distance[None, :] * std[:, None] / (max_distance * 0.1) + mean[:, None]) + max_distance / 2. - 1 # (-1, max_distance)
        normed_distance = self.interpolated_gather_nd(distance, new_indices)

        return K.reshape(normed_distance, original_shape)