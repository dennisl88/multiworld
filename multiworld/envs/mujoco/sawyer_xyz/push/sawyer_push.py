from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerRandGoalEnv

class SawyerPushEnv(SawyerRandGoalEnv):
    def __init__(
            self,
            rew_mode='posPlace',
            obj_type='puck',
            **kwargs
    ):
        self.quick_init(locals())
        SawyerRandGoalEnv.__init__(
            self,
            obj_type='puck',
            **kwargs
        )
        self.rew_mode = rew_mode

    def step(self, action):
        ob, _, done, _ = super().step(action)
        reward, reachDist, placeDist  = self.compute_rewards(action, ob)
        return ob, reward, done, {'reachDist':reachDist, 'placeDist': placeDist, 'epRew': reward}

    def reset_model(self):
        super().reset_model()
        self.origPlacingDist = np.linalg.norm(self.obj_init_pos[:2] - self._state_goal[:2])
        return self._get_obs()

    def compute_rewards(self, actions, obs):   
        state_obs = obs['state_observation']
        endEffPos, objPos = state_obs[0:3], state_obs[3:6]
               
        placingGoal = self._state_goal
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        objPos = self.get_body_com("obj")
        fingerCOM = (rightFinger + leftFinger)/2

        c1 = 1 ; c2 = 1
        reachDist = np.linalg.norm(objPos - fingerCOM)    
        placeDist = np.linalg.norm(objPos - placingGoal)

        if self.rew_mode == 'normal':
            reward = -reachDist - placeDist

        elif self.rew_mode == 'posPlace':
            reward = -reachDist + 100* max(0, self.origPlacingDist - placeDist)

        return reward, reachDist, placeDist