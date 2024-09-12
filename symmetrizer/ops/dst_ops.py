import numpy as np
import torch
import torch.nn.functional as F

from .ops import GroupRepresentations



def get_dst_state_group_representations():
    """
    Representation of the group symmetry on the state: a multiplication of all
    state variables by -1
    """

    # f = np.array(
    #     [
    #         [-1, 0, 0, 0],
    #         [0, -1, 0, 0],
    #         [0, 0, 1, 0],
    #         [0, 0, 0, 1],
    #     ]
    # )

    f = np.array(
        [
            [1, 0],
            [0, -1],
        ]
    )

    representations = [torch.FloatTensor(np.eye(2)),
                       torch.FloatTensor(f)
                       ]
    return GroupRepresentations(representations, "DstStateGroupRepr")


def get_dst_action_group_representations():
    """
    Representation of the group symmetry on the policy: a permutation of the
    actions
    """
    f = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )

    #print(np.dot(f,f))

    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(f)
                       ]
    return GroupRepresentations(representations, "DstActionGroupRepr")


def get_dst_invariants():
    """
    Function to enable easy construction of invariant layers (for value networks)
    """
    representations = [torch.FloatTensor(np.eye(1)),
                       torch.FloatTensor(np.eye(1))]
    return GroupRepresentations(representations, "DstInvariantGroupRepr")

