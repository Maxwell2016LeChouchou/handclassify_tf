# -*- coding: utf-8 -*-


import core.network_wjz1_hand as network_wjz1_hand
import core.network_wjz2_hand as network_wjz2_hand
import core.network_wjz3_hand as network_wjz3_hand
import core.network_wjz4_hand as network_wjz4_hand
import core.network_wjz5_hand as network_wjz5_hand

def get_network(type, input, trainable=True):
    # if type == "hand_net1":
    if type == "hand_net1":
        emotion = network_wjz1_hand.build_network(input, trainable)
    elif type == "hand_net2":
        emotion = network_wjz2_hand.build_network(input, trainable)
    elif type == "hand_net3":
        emotion = network_wjz3_hand.build_network(input, trainable)
    elif type == "hand_net4":
        emotion = network_wjz4_hand.build_network(input, trainable)
    elif type == "hand_net5":
        emotion = network_wjz5_hand.build_network(input, trainable)

    return emotion

