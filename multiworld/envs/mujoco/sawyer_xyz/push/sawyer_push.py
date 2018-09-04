from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

class SawyerPushEnv(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=(-0.2, 0.5, 0.02),
            obj_high=(0.2, 0.7, 0.02),
            goal_low=(-0.2, 0.5, 0.02),
            goal_high=(0.2, 0.7, 0.02),
            hand_init_pos = (0, 0.4, 0.05),
            rewMode = 'posPlace',
            **kwargs
    ):
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

        #Make sure that objectGeom is the last geom in the XML 
        self.objHeight = self.model.geom_pos[-1][2]
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
        return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

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
        self.do_simulation(None)
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()

        reward, reachDist, placeDist  = self.compute_rewards(action, ob)
        done = False
        return ob, reward, done, {'reachDist':reachDist, 'placeDist': placeDist, 'epRew': reward}

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

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
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
        self.origPlacingDist = np.linalg.norm( self.obj_init_pos[:2] - self._state_goal[:2])
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            #self.do_simulation([1,-1], self.frame_skip)
            self.do_simulation(None, self.frame_skip)

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

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

        if self.rewMode == 'normal':
            reward = -reachDist - placeDist

        elif self.rewMode == 'posPlace':
            reward = -reachDist + 100* max(0, self.origPlacingDist - placeDist)

        return [reward, reachDist, placeDist]    

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics