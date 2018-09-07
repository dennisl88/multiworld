import abc
import numpy as np

# import ipdb
# ipdb.set_trace()
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

from multiworld.envs.env_util import quat_to_zangle, zangle_to_quat

import copy


class SawyerMocapBase(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=5):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        # Resets the mocap welds that we use for actuation.
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.]
                    )

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.get_body_xpos('hand')]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.get_body_quat('hand')]),
        )

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            action_scale=1/100,
            action_zangle_scale = 1/10,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_scale = action_scale
        self.action_zangle_scale = action_zangle_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        self.mocap_low = np.hstack(hand_low)
        self.mocap_high = np.hstack(hand_high)

    def set_xyzRot_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        zangle_delta = action[3] * self.action_zangle_scale
        #zangle_delta = np.random.uniform(-0.1, 0.1)
        new_mocap_zangle = quat_to_zangle(self.data.mocap_quat[0]) + zangle_delta
        new_mocap_zangle = np.clip(
            new_mocap_zangle,
            -3.0,
            3.0,
        )

        if new_mocap_zangle < 0:
            new_mocap_zangle += 2 * np.pi

        self.data.set_mocap_quat('mocap', zangle_to_quat(new_mocap_zangle))

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

class SawyerRandGoalEnv(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=(-0.2, 0.5, 0.02),
            obj_high=(0.2, 0.7, 0.02),
            goal_low=(-0.2, 0.5, 0.02),
            goal_high=(0.2, 0.7, 0.02),
            hand_init_pos = (0, 0.4, 0.05),
            **kwargs
    ):        
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        obj_low = np.array(obj_low)
        obj_high = np.array(obj_high)
        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)

        # Make sure that objectGeom is the last geom in the XML 
        self.objHeight = self.model.geom_pos[-1][2]
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
        self.do_simulation(action[3:])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward = 0
        done = False

        return ob, reward, done, {}

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
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            #self.do_simulation([1,-1], self.frame_skip)
            self.do_simulation(None, self.frame_skip)

    def compute_rewards(self, actions, obs):   
        raise NotImplementedError

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics