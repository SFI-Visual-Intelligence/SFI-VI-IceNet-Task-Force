import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Conv1D,
    Dense,
    Input,
    Lambda,
    ReLU,
    Softmax,
    Add,
)

import config
import numpy as np
import os
import sys


# import memory_saving_gradients

# # monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

M = 12  # no of symbolic entities in knowledge graph
K = 50  # fasttext embedding vector size
Dc = 64  # no of conv filters within SGR

masks = tf.convert_to_tensor(
    np.load(os.path.join(config.mask_data_folder, config.region_mask_filename)),
    dtype=tf.int32,
)
masks = tf.reshape(masks, [1] + masks.shape + [1])  # [1,432,432,1]


def norm_adjacency(a):
    # Adds identity connection to adjacency matrix, converts to tf constant
    # and normalize

    # row normalize the adjacency matrix

    tf_adj = a + tf.eye(M)  # add identity connections
    Q = tf.reduce_sum(tf_adj, axis=-1)  # take sum along rows # input mat 1,H,W
    Q = tf.cast(Q, tf.float32)  # conver tot float
    sQ = tf.math.rsqrt(Q)  # reciprocal square root
    sQ = tf.linalg.diag(sQ)  # make diagonal mat
    # do symmetric normalization
    norm_adj = tf.matmul(
        sQ, tf.matmul(tf_adj, sQ, name="right_side_norm"), name="left_side_norm"
    )
    return norm_adj


def _compute_compat_batches(inp, evolved, Dl):
    # batch_data = (inp, evolved)

    # def compute_compat_batch(batch_x, batch_feat):
    #     # batch_x has shape [H,W,Dl]
    #     # batch_feat has shape [M,Dc]
    #     sh = tf.shape(batch_x)
    #     batch_x = tf.reshape(batch_x, [sh[0] * sh[1], sh[2]])  # shape [H*W,Dl]
    #     batch_x = tf.tile(tf.expand_dims(batch_x, 1), [1, M, 1])  # has shape [H*W,M,Dl]

    #     batch_feat = tf.tile(
    #         tf.expand_dims(batch_feat, 0), [sh[0] * sh[1], 1, 1]
    #     )  # [H*W,M,Dc]

    #     compare_concat = tf.concat(
    #         [batch_feat, batch_x], axis=-1, name="concat_inp"
    #     )  # [H*W,M,(Dc+Dl)]

    #     # compare_concat = tf.expand_dims(compare_concat, 0)
    #     # compare_concat.set_shape([None, None, Dl + Dc])  # idk
    #     compat = tf.nn.conv2d(  # Ws in paper
    #         compare_concat[tf.newaxis, :, :, :],
    #         filters=[1, 1, Dl + Dc, 1],
    #         strides=1,
    #         padding="SAME",
    #     )  # has shape (H*W,M,1)

    #     return tf.reshape(compat, [sh[0] * sh[1], M])

    # res = Lambda(
    #     lambda x: tf.map_fn(
    #         lambda batch: compute_compat_batch(*batch), x, dtype=tf.float32
    #     )
    # )(batch_data)
    sh = tf.shape(inp)
    inp = tf.reshape(inp, (sh[0], sh[1] * sh[2], Dl))
    inp = tf.tile(tf.expand_dims(inp, 2), [1, 1, M, 1])
    evolved = tf.tile(tf.expand_dims(evolved, 1), [1, sh[1] * sh[2], 1, 1])
    compare_concat = tf.concat([evolved, inp], axis=-1, name="concat_inp")
    compat = Conv2D(
        filters=1,
        kernel_size=1,
        kernel_initializer="glorot_normal",
        padding="same",
    )(compare_concat)

    return tf.squeeze(compat)


def SGRLayer(Dl):
    input1 = Input(shape=(None, None, Dl))
    # input2 = Input(shape=(M, M))
    input2 = np.load(os.path.join("icenet", "average_graph.npz"))["graph"]

    INPUT_SHAPE = tf.shape(input1)

    #
    # Graph reasoning
    #

    votes = Conv2D(
        filters=M,
        kernel_size=1,
        padding="same",
        kernel_initializer="glorot_normal",
        activation="softmax",
    )(input1)
    votes = tf.reshape(
        votes,
        [INPUT_SHAPE[0], INPUT_SHAPE[1] * INPUT_SHAPE[2], M],
    )  # b,(H*W),M
    votes = tf.transpose(votes, [0, 2, 1])  # b,M,(H*W)

    in_feat = Conv2D(  # transform each local feature into length Dc
        filters=Dc,
        kernel_size=1,
        padding="same",
        kernel_initializer="glorot_normal",
    )(input1)
    # b,H,W,Dc

    in_feat = tf.reshape(
        in_feat, [INPUT_SHAPE[0], INPUT_SHAPE[1] * INPUT_SHAPE[2], Dc]
    )  # b,(H*W),Dc
    in_feat = tf.matmul(votes, in_feat, name="applied_votes")  # b,M,Dc
    visual_features = ReLU(name="visual_feat")(in_feat)  # b,M,Dc

    #
    # Graph reasoning
    #

    S = tf.tile(
        tf.random.normal((1, M, K), mean=0, stddev=0.1, dtype=tf.float32),
        [INPUT_SHAPE[0], 1, 1],
    )  # sh [b,M,K]

    concat_feat = tf.concat(
        [visual_features, S], axis=-1, name="concat_embed"
    )  # shape [b,M,Dc+K]
    concat_feat = tf.reshape(concat_feat, [-1, Dc + K])  # shape [b*M,Dc+K]

    transformed_concat_feat = Dense(Dc, kernel_initializer="glorot_normal")(concat_feat)
    transformed_concat_feat = tf.reshape(
        transformed_concat_feat, [INPUT_SHAPE[0], M, Dc]
    )

    norm_adj = norm_adjacency(input2)
    # norm_adj = tf.tile(norm_adj[tf.newaxis, :, :], [INPUT_SHAPE[0], 1, 1])
    evolved_feat = tf.matmul(
        norm_adj, transformed_concat_feat, name="matmul_adj_feat"
    )  # shape [b,M,M] @ [b,M,Dc] --> [b,M,Dc]

    evolved_feat = ReLU(name="evolved_feats")(evolved_feat)  # shape [b,M,Dc]

    #
    # Semantic-to-Local mapping
    #

    mapping = _compute_compat_batches(input1, evolved_feat, Dl)  # b,(H*W),M

    mapping = Softmax(name="mapping_softmax")(mapping)  # b,(H*W),M

    mapping = tf.linalg.normalize(mapping, ord=1, axis=-1)[0]  # b,(H*W),M

    evolved_feat = Conv1D(
        filters=Dl,
        kernel_size=1,
        padding="same",
        kernel_initializer="glorot_normal",
        name="Wsp",
    )(evolved_feat)
    # shape [b,M,Dl])

    applied_mapping = tf.matmul(
        mapping, evolved_feat, name="applied_mapping"
    )  # b,(H*W),Dl

    applied_mapping = ReLU(name="mapping_relu")(applied_mapping)  # b,(H*W),Dl
    applied_mapping = tf.reshape(applied_mapping, INPUT_SHAPE)  # b,H,W,Dl

    output = applied_mapping + input1

    return Model(inputs=input1, outputs=output)


def extract_graph(input1):
    sh = tf.shape(input1)  # [b,432,432,50]
    nodes = []
    for i in tf.range(1, M + 1):
        mask = masks == i
        num_pixels_mask = np.sum(mask)  # [1]
        mask = tf.broadcast_to(mask, sh)  # [b,432,432,50]
        masked = tf.where(mask, input1, tf.zeros_like(input1))  # [b,432,432,50]
        avg = (
            tf.reduce_sum(tf.reshape(masked, sh), axis=(1, 2)) / num_pixels_mask
        )  # [b,50]
        nodes.append(avg)  # list of length 12

    nodes = tf.stack(nodes, axis=1)  # [b,M,50]
    tile_1 = tf.tile(tf.expand_dims(nodes, 1), [1, M, 1, 1])  # [b,M,M,50]
    tile_2 = tf.tile(tf.expand_dims(nodes, 2), [1, 1, M, 1])  # [b,M,M,50]
    l2_dist = -tf.norm(tile_1 - tile_2, axis=-1)  # [b,M,M]
    minimum_dist = tf.math.top_k(l2_dist, 4).values[:, :, -1]
    knn = tf.cast(l2_dist >= tf.expand_dims(minimum_dist, -1), tf.float32)  # [b,M,M]

    return knn
