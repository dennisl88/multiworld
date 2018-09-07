from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerRandGoalEnv

class SawyerPushSimpleEnv(SawyerRandGoalEnv):
    def __init__(
            self,
            obj_low=(-0.1, 0.5, 0.02),
            obj_high=(0.1, 0.5, 0.02),
            goal_low=(-0.1, 0.7, 0.02),
            goal_high=(0.1, 0.7, 0.02),
            hand_init_pos = (0, 0.4, 0.05),
            rewMode='angle',
            **kwargs
    ):
        self.quick_init(locals())
        
        SawyerRandGoalEnv.__init__(
            self,
            obj_low=obj_low,
            obj_high=obj_high,
            goal_low=goal_low,
            goal_high=goal_high,
            model_name=self.model_name,
            **kwargs
        )
        self.rewMode = rewMode

    def step(self, action):
        ob, _, done, _ = super().step(action)
        reward, reachDist, placeDist, cosDist  = self.compute_rewards(action, ob)
        return ob, reward, done, {'reachDist':reachDist, 'placeDist': placeDist, 'cosDist': cosDist, 'epRew': reward}

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

        reachDist = np.linalg.norm(objPos - fingerCOM)
        placeDist = np.linalg.norm(objPos - placingGoal)

        v1 = placingGoal - objPos
        v2 = objPos - fingerCOM
        cosDist = v1.dot(v2) / (reachDist * placeDist)

        if self.rewMode == 'normal':
            reward = -reachDist - placeDist

        elif self.rewMode == 'posPlace':
            reward = -reachDist + 100* max(0, self.origPlacingDist - placeDist)

        elif self.rewMode == 'angle':
            reward = -reachDist - placeDist + 0.1 * cosDist

        return reward, reachDist, placeDist, cosDist