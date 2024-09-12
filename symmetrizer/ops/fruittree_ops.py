import numpy as np
import torch
import torch.nn.functional as F

from .ops import GroupRepresentations


## to be done

def get_fruittree_state_group_representations():
    """
    Representation of the group symmetry on the state: a multiplication of all
    state variables by -1
    """

    e = np.eye(8)

    f = np.array(
        [
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )

    representations = [torch.FloatTensor(e),
                       torch.FloatTensor(f)]
    return GroupRepresentations(representations, "FruitTreeStateGroupRepr")


def get_fruittree_action_group_representations():
    """
    Representation of the group symmetry on the policy: a permutation of the
    actions
    """
    f = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ]
    )

    print(np.dot(f,f))

    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(f)
                       ]
    return GroupRepresentations(representations, "FruitTreeActionGroupRepr")


def get_fruittree_invariants():
    """
    Function to enable easy construction of invariant layers (for value networks)
    """
    representations = [torch.FloatTensor(np.eye(1)),
                       torch.FloatTensor(np.eye(1))]
    return GroupRepresentations(representations, "FruitTreeInvariantGroupRepr")

