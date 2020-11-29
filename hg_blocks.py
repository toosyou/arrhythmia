import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Add, MaxPooling1D, UpSampling1D, concatenate
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import backend as K

from distance import MultiHeadDistanceLayer

class PaddingLike(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        source_shape = K.shape(source)
        target_shape = K.shape(target)
        length_diff = target_shape[1]-source_shape[1]
        return K.temporal_padding(source, padding=(length_diff//2, length_diff - length_diff//2))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def create_hourglass_network(num_classes, num_stacks, num_channels, input_shape, bottleneck, module_layers):
    input = Input(shape=input_shape)

    front_features = create_front_module(input, num_channels, bottleneck)

    head_next_stage = front_features

    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, module_layers, i)
        outputs.append(sigmoid(head_to_loss))

    model = Model(inputs=input, outputs=outputs)
    return model

def hourglass_module(bottom, num_classes, num_channels, bottleneck, module_layers, hgid):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels, module_layers)

    # distance1 = distance_module(left_features[1], num_channels, max_distance=60)
    distance2 = distance_module(left_features[2], num_channels, max_distance=30)

    # left_features[1] = concatenate([left_features[1], distance1])
    left_features[2] = concatenate([left_features[2], distance2])

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid, num_channels)

    return head_next_stage, head_parts

def distance_module(x, num_channels, num_heads=4, max_distance=30):
    distance_layer = MultiHeadDistanceLayer(num_heads, num_channels, 
                                            'local', num_channels//num_heads, 
                                            distance_norm=True, max_distance=max_distance,
                                            smooth_embedding_ratio=6)
    distance_layer = tf.recompute_grad(distance_layer)

    distance = distance_layer(x)
    # distance = BatchNormalization()(distance)
    distance = LayerNormalization()(distance)

    return distance

def bottleneck_block(bottom, num_out_channels, block_name):
    # skip layer
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = Conv1D(num_out_channels, kernel_size=1, activation='relu', padding='same',
                       name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = Conv1D(num_out_channels / 2, kernel_size=1, activation='relu', padding='same',
                name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    _x = Conv1D(num_out_channels / 2, kernel_size=3, activation='relu', padding='same',
                name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv1D(num_out_channels, kernel_size=1, activation='relu', padding='same',
                name=block_name + '_conv_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name + '_residual')([_skip, _x])

    return _x

def create_front_module(input, num_channels, bottleneck):
    # front module, input to 1/4 resolution
    # 1 7x7 conv + maxpooling
    # 3 residual block

    _x = Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu', name='front_conv_1x1_x1')(
        input)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
    _x = MaxPooling1D(pool_size=2, strides=2)(_x)

    _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
    _x = bottleneck(_x, num_channels, 'front_residual_x3')

    return _x


def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels, module_layers):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hgname = 'hg' + str(hglayer)

    features = list()

    _x = bottom
    for i in range(module_layers):
        f = bottleneck(_x, num_channels, hgname + '_l{}'.format(2**i))
        if i != module_layers - 1:
            _x = MaxPooling1D(pool_size=2, strides=2)(f)
        features.append(f)
    return features


def connect_left_to_right(left, right, bottleneck, name, num_channels):
    '''
    :param left: connect left feature to right feature
    :param name: layer name
    :return:
    '''
    # left -> 1 bottlenect
    # right -> upsampling
    # Add   -> left + right

    _xleft = bottleneck(left, num_channels, name + '_connect')
    _xright = UpSampling1D()(right)
    _xright = PaddingLike()([_xright, _xleft])
    add = Add()([_xleft, _xright])
    out = bottleneck(add, num_channels, name + '_connect_conv')
    return out


def bottom_layer(lf, bottleneck, hgid, num_channels, downsample):
    # blocks in lowest resolution
    # 3 bottlenect blocks + Add

    lf_connect = bottleneck(lf, num_channels, str(hgid) + "_lf{}".format(downsample))

    _x = bottleneck(lf, num_channels, str(hgid) + "_lf{}_x1".format(downsample))
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf{}_x2".format(downsample))
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf{}_x3".format(downsample))

    rf8 = Add()([_x, lf_connect])

    return rf8


def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels):

    num_leftfeatures = len(leftfeatures)

    rf = bottom_layer(leftfeatures[-1], bottleneck, hglayer, num_channels, 2**(num_leftfeatures-1))
    for i, lf in enumerate(leftfeatures[-2::-1]):
        rf = connect_left_to_right(lf, rf, bottleneck, 'hg' + str(hglayer) + '_rf{}'.format(2**(num_leftfeatures - 2 - i)), num_channels)
    return rf

def create_heads(prelayerfeatures, rf1, num_classes, hgid, num_channels):
    # two head, one head to next stage, one head to intermediate features
    head = Conv1D(num_channels, kernel_size=1, activation='relu', padding='same', name=str(hgid) + '_conv_1x1_x1')(
        rf1)
    head = BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = Conv1D(num_classes, kernel_size=1, activation='linear', padding='same',
                        name=str(hgid) + '_conv_1x1_parts')(head)

    # use linear activation
    head = Conv1D(num_channels, kernel_size=1, activation='linear', padding='same',
                  name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = Conv1D(num_channels, kernel_size=1, activation='linear', padding='same',
                    name=str(hgid) + '_conv_1x1_x3')(head_parts)

    head_next_stage = Add()([head, head_m, prelayerfeatures])
    return head_next_stage, head_parts