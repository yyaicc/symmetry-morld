import mo_gymnasium as mo_gym
import numpy as np
import time
import torch  # noqa: F401
from mo_gymnasium import MORecordEpisodeStatistics
from morl_baselines.common.evaluation import seed_everything
from morl_baselines.multi_policy.morld.morld import MORLD
from morl_baselines.common.env import get_ref_point_and_pfs, DSTS_MAP, DSTS_PARETO_FRONT
#torch.set_default_tensor_type(torch.DoubleTensor)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"



def main(seed, env_name, gamma=0.99, total_timesteps=int(1e5+1)):

    seed_everything(seed)

    if "mo-" not in env_name:
        env, eval_env, ref_point, _ = get_ref_point_and_pfs("deep-sea-treasure-concave-v0", gamma)

        env.unwrapped.sea_map = DSTS_MAP[env_name + "_S"]
        eval_env.unwrapped.sea_map = DSTS_MAP[env_name + "_S"]
        known_pareto_front = DSTS_PARETO_FRONT[env_name]

        env_name = "deep-sea-treasure-concave-" + env_name + "_S"

        # print(env.reset())

    else:
        env, eval_env, ref_point, known_pareto_front = get_ref_point_and_pfs(env_name, gamma)

    algo = MORLD(
        env=env,
        #known_pareto_front=known_pareto_front,
        env_name=env_name,
        exchange_every=int(1e3),
        pop_size=10,
        policy_name="EUPG",
        scalarization_method="tch",
        evaluation_mode="esr",
        alg_name="MORL-D",
        gamma=gamma,
        log=False,
        neighborhood_size=1,
        update_passes=10,
        shared_buffer=False,
        sharing_mechanism=[],
        weight_adaptation_method="PSA",
        seed=seed,
    )

    hv = algo.train(
        eval_env=eval_env,
        total_timesteps=total_timesteps,
        ref_point=ref_point,
        known_pareto_front=known_pareto_front,
    )

    return hv


if __name__ == "__main__":

    env_names = [
                #"deep-sea-treasure-concave-v0",
                # #"resource-gathering-v0",
                # "fruit-tree-v0",
                # # "minecart-v0",  # EUPG中有效的探索较少
                # "four-room-v0",   # EUPG中有效的探索较少
                # "mo-mountaincar-v0", # EUPG中有效的探索较少
                #"mo-lunar-lander-v2",
                #"mo-highway-fast-v0",   # training time may be a litter long
                #"v0",
                # "DST_2",
                # "DST_3",
                # "DST_4",
                "mo-reacher-v4",
                ]

    gamma = 0.99
    hv_list = []
    for env_name in env_names:

        if env_name != "mo-reacher-v4":
            total_timesteps = int(1e5 + 1)
        else:
            total_timesteps = int(2e5 + 1)

        for seed in range(3047, 3057):
            s_start = time.time()
            hv = main(seed, env_name, gamma, total_timesteps)
            print("Env:", env_name, "Seed:", seed, "HV:", hv, "Time:", time.time() - s_start)
