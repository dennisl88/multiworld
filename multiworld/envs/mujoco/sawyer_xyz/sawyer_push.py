from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerPickPlaceEnv( SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            tasks = [{'goal': [0, 0.7, 0.02], 'obj_init_pos':[0, 0.6, 0.02]}] , 

            goal_low=None,
            goal_high=None,

            hand_init_pos = (0, 0.4, 0.05),
            #hand_init_pos = (0, 0.5, 0.35) ,
            blockSize = 0.02,

            **kwargs
    ):
        self.quick_init(locals())
        
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

       


        self.max_path_length = 150

        self.tasks = tasks
        self.num_tasks = len(tasks)

        # defaultTask = tasks[0]


        # self.obj_init_pos = np.array(tasksobj_init_pos)
        self.hand_init_pos = np.array(hand_init_pos)

        self.blockSize = blockSize

        self.action_space = Box(
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.goal_space = Box(goal_low, goal_high)

        self.observation_space = Dict([
           
            ('state_observation', self.hand_and_obj_space),

            ('desired_goal', self.goal_space)
        ])

        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

    def viewer_setup(self):
        pass
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.6
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1

    def step(self, action):

        #debug mode:

        # action[:3] = [0,0,-1]
        # self.do_simulation([1,-1])
     
        # print(action[-1])

    
        self.set_xyz_action(action[:3])

        #self.do_simulation([action[-1], -action[-1]])
        
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
       

        reward , reachRew, placeRew  = self.compute_rewards(action, ob)
        self.curr_path_length +=1

       
        #info = self._get_info()

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, { 'reachRew':reachRew,  'placeRew': placeRew, 'reward': reward}



   
    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
      
        flat_obs = np.concatenate((e, b))

        return dict(
            
            desired_goal=self._state_goal,
            
            state_observation=flat_obs,
            
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


    def sample_task(self):


        task_idx = np.random.randint(0, self.num_tasks)
    
        return self.tasks[task_idx]

    def reset_model(self):

        

        self._reset_hand()

        task = self.sample_task()
        
        self._state_goal = task['goal']
        self.obj_init_pos = task['obj_init_pos']
      
        self._set_goal_marker(self._state_goal)

        self._set_obj_xyz(self.obj_init_pos)

        self.curr_path_length = 0
        



        return self._get_obs()

    def _reset_hand(self):


        
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([1,-1], self.frame_skip)
            #self.do_simulation(None, self.frame_skip)



    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()





    def compute_rewards(self, actions, obs):
           
        state_obs = obs['state_observation']

        endEffPos , objPos = state_obs[0:3], state_obs[3:6]
        
        
       
       
        placingGoal = self._state_goal

        
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        objPos = self.get_body_com("obj")
        fingerCOM = (rightFinger + leftFinger)/2


        c1 = 1 ; c2 = 1

        reachRew = -np.linalg.norm(objPos - fingerCOM)
    
        placeRew = -np.linalg.norm(objPos - placingGoal)
        

        reward = c1*reachRew + c2*placeRew

      


        return [reward, reachRew, placeRew] 
     

   

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
       
        return statistics

   