# -*- coding: utf-8 -*-


import core.network_zq1_hand as network_zq1_hand
import core.network_zq2_hand as network_zq2_hand
import core.network_wjz1_hand as network_wjz1_hand
import core.network_wjz2_hand as network_wjz2_hand
import core.network_wjz3_hand as network_wjz3_hand

def get_network(type, input, trainable=True):
    # if type == "zq1_hand":
    if type == "zq_hand_1":
        emotion = network_zq1_hand.build_network(input, trainable)
    elif type == "zq_hand_2":
        emotion = network_zq2_hand.build_network(input, trainable)
    elif type == "zq_hand_wjz1":
        emotion = network_wjz1_hand.build_network(input, trainable)
    elif type == "zq_hand_wjz2":
        emotion = network_wjz2_hand.build_network(input, trainable)
    elif type == "zq_wjz_hand":
        emotion = network_wjz3_hand.build_network(input, trainable)

    return emotion

