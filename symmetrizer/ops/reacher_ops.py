import numpy as np
import torch
import torch.nn.functional as F

from .ops import GroupRepresentations

def SO2(theta):

    f = np.array(
        [
            [np.cos(theta), 0, -np.sin(theta), 0, 0, 0],
            [0, np.cos(theta), 0, -np.sin(theta), 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0, 0, 0],
            [0, np.sin(theta), 0, np.cos(theta), 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    return f

def get_reacher_state_group_representations():
    """
    Representation of the group symmetry on the state: a multiplication of all
    state variables by -1
    """

    # print(SO2(np.pi/2))
    # print(SO2(np.pi))
    # print(SO2(np.pi*1.5))

    f = np.array(
        [
            [-1, 0, 0, 0, 0,0],
            [0,-1, 0, 0, 0, 0],
            [0, 0,-1, 0, 0, 0],
            [0, 0, 0,-1, 0, 0],
            [0, 0, 0, 0,-1, 0],
            [0, 0, 0, 0, 0,-1],
        ]
    )


    representations = [torch.FloatTensor(np.eye(6)),
                       torch.FloatTensor(f)
                       # torch.FloatTensor(SO2(np.pi/2)),
                       # torch.FloatTensor(SO2(np.pi)),
                       # torch.FloatTensor(SO2(1.5*np.pi)),
                       ]
    return GroupRepresentations(representations, "ReacherStateGroupRepr")


def get_reacher_action_group_representations():
    """
    Representation of the group symmetry on the policy: a permutation of the
    actions
    """
    f = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],

        ]
    )

    #print(np.dot(f,f))

    representations = [torch.FloatTensor(np.eye(9)),
                       torch.FloatTensor(f)
                       ]
    return GroupRepresentations(representations, "ReacherActionGroupRepr")


def get_reacher_invariants():
    """
    Function to enable easy construction of invariant layers (for value networks)
    """
    representations = [torch.FloatTensor(np.eye(1)),
                       torch.FloatTensor(np.eye(1))]
    return GroupRepresentations(representations, "ReacherInvariantGroupRepr")

