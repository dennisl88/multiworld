from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerRandGoalEnv

class SawyerPickPlaceEnv(SawyerRandGoalEnv):
    def __init__(
            self,
            liftThresh=0.04,
            rewMode='orig',
            objType='block',
            **kwargs
    ):
        self.objType = objType
        
        self.quick_init(locals())
        SawyerRandGoalEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )

        self.liftThresh = liftThresh

        self.heightTarget = self.objHeight + self.liftThresh
        self.rewMode = rewMode

    @property
    def model_name(self):
        if self.objType == 'block':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')
        elif self.objType == 'fox':
            return get_asset_full_path('sawyer_xyz/pickPlace_fox.xml')
        else:
            raise AssertionError('Obj Type must be block or fox')

    def step(self, action):
        ob, _, done, _ = super().step(action)
        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_reward(action, ob)
        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist}

    def reset_model(self):
        super()._get_obs()
        self.pickCompleted = False
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

    def compute_rewards(self, actions, obsBatch):
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs):
        if isinstance(obs, dict):
            obs = obs['state_observation']

        objPos = obs[3:6]
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2
       
        heightTarget = self.heightTarget
        placingGoal = self._state_goal
      
        graspDist = np.linalg.norm(objPos - fingerCOM)
        placingDist = np.linalg.norm(objPos - placingGoal)
      
        def reachReward():
            graspRew = -graspDist
            # incentive to close fingers when graspDist is small
            if graspDist < 0.02:
                graspRew = -graspDist + max(actions[-1],0)/50
            return graspRew , graspDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def grasped():
            sensorData = self.data.sensordata
            return (sensorData[0]>0) and (sensorData[1]>0)

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (graspDist > 0.02) 

        def orig_pickReward():
            hScale = 50
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            elif (objPos[2]> (self.objHeight + 0.005)) and (graspDist < 0.1):
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def general_pickReward():
            hScale = 50
            if self.pickCompleted and grasped():
                return hScale*heightTarget
            elif (objPos[2]> (self.objHeight + 0.005)) and grasped():
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward(cond):
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if cond:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                placeRew = max(placeRew,0)
                return [placeRew , placingDist]
            else:
                return [0 , placingDist]

        reachRew, reachDist = reachReward()
        if self.rewMode == 'orig':
            pickRew = orig_pickReward()
            placeRew , placingDist = placeReward(cond = self.pickCompleted and (graspDist < 0.1) and not(objDropped()))
        else:
            assert(self.rewMode == 'general')
            pickRew = general_pickReward()
            placeRew , placingDist = placeReward(cond = self.pickCompleted and grasped())

        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        return reward, reachRew, reachDist, pickRew, placeRew, placingDist