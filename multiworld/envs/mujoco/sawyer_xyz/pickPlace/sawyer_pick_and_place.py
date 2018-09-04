from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerPickPlaceEnv(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=(-0.2, 0.5, 0.02),
            obj_high=(0.2, 0.7, 0.02),
            goal_low=(-0.2, 0.5, 0.02),
            goal_high=(0.2, 0.7, 0.02),
            hand_init_pos = (0, 0.4, 0.05),
            liftThresh = 0.04,
            rewMode = 'orig',
            objType = 'block',
            **kwargs
    ):
        self.objType = objType
        self.quick_init(locals())
        
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        obj_low = np.array(obj_low)
        obj_high = np.array(obj_high)
        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)

        self.liftThresh = liftThresh

        #Make sure that objectGeom is the last geom in the XML 
        self.objHeight = self.model.geom_pos[-1][2]
        self.heightTarget = self.objHeight + self.liftThresh
        self.rewMode = rewMode
        self.hand_init_pos = np.array(hand_init_pos)

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.obj_space = Box(obj_low, obj_high)

        self.goal_space = Box(goal_low, goal_high)

        self.observation_space = Dict([
            ('state_observation', self.hand_and_obj_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

    @property
    def model_name(self):
        if self.objType == 'block':
            return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')
        elif self.objType == 'fox':
            return get_asset_full_path('sawyer_xyz/pickPlace_fox.xml')
        else:
            raise AssertionError('Obj Type must be block or fox')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.6
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_reward(action, ob)
        done = False
        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist}
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('objGeom')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def sample_goals(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )
        obj_poss = np.random.uniform(
            self.obj_space.low,
            self.obj_space.high,
            size=(batch_size, self.obj_space.low.size),
        )
        return [{
            'state_desired_goal': goal,
            'obj_init_pos': obj_pos
        } for goal, obj_pos in zip(goals, obj_poss)]

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)
        self.obj_init_pos = self.adjust_initObjPos(goal['obj_init_pos'])

    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
            'obj_init_pos': self.obj_init_pos,
        }

    def adjust_initObjPos(self, orig_init_pos):
        # Dennis: What is this doing?
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],0]

    def reset_model(self):
        self._reset_hand()
        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        self.pickCompleted = False
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

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
        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist] 

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass