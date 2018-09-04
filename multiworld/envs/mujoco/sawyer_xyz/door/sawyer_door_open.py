from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerDoorOpenEnv(SawyerXYZEnv):

    def __init__(
            self,
            doorGrasp_low=None,
            doorGrasp_high=None,
            goal_low= np.array([0]),
            goal_high=np.array([1.58825]),
            hand_init_pos = (0, 0.4, 0.05),
            doorHalfWidth = 0.2,
            **kwargs
    ):
        self.quick_init(locals()) 
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if doorGrasp_low is None:
            doorGrasp_low = self.hand_low
        if doorGrasp_high is None:
            doorGrasp_high = self.hand_high

        self.doorHalfWidth = doorHalfWidth
        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
            dtype = np.float32
        )
        self.hand_and_door_space = Box(
            np.hstack((self.hand_low, doorGrasp_low)),
            np.hstack((self.hand_high, doorGrasp_high)),
            dtype = np.float32
        )
        self.goal_space = Box(goal_low, goal_high)
        self.observation_space = Dict([
            ('state_observation', self.hand_and_door_space),
            ('desired_goal', self.goal_space),
            
        ])
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_door_open.xml')

    def viewer_setup(self):
        pass
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.3
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1

    def step(self, action):


        self.set_xyz_action(action[:3])


       
      
        self.do_simulation([action[-1], -action[-1]])

        self._set_goal_marker()
        # The marker seems to get reset every time you do a simulation
       
        ob = self._get_obs()
       

        reward , doorOpenRew , doorAngle, doorAngleTarget= self.compute_rewards(action, ob)
        self.curr_path_length +=1

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, {'doorOpenRew':doorOpenRew, 'reward': reward, 'doorAngle': doorAngle, 'doorAngleTarget': doorAngleTarget}

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_site_pos('doorGraspPoint')
        flat_obs = np.concatenate((e, b))

        return dict(
            state_observation=flat_obs,
            desired_goal=self._state_goal,
           
        )


    def get_endeff_pos(self):


        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')

        return (rightFinger + leftFinger)/2




    def _set_door_xyz(self, doorPos):


        self.model.body_pos[-1] = doorPos

        
        


    def sample_task(self):


        task_idx = np.random.randint(0, self.num_tasks)
    
        return self.tasks[task_idx]



    def _set_goal_marker(self):
    

        angle = self._state_goal

        door_pos = self.door_init_pos

        # import ipdb
        # ipdb.set_trace()

        goal_x = door_pos[0] + self.doorHalfWidth *(1 - np.cos(angle))

        goal_y = door_pos[1] - self.doorHalfWidth*np.sin(angle)

        goalSitePos = np.array([goal_x, goal_y,  door_pos[2]])

        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goalSitePos
        )



    def reset_model(self):

        

        self._reset_hand()
        
        task = self.sample_task()


        self._state_goal = task['goalAngle'][0]
        self.door_init_pos = task['door_init_pos']



        self._set_goal_marker()

        self._set_door_xyz(self.door_init_pos)

        self.curr_path_length = 0
      
       
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

   
    

    def get_site_pos(self, siteName):

       
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()





    def compute_rewards(self, actions, obs):


        if isinstance(obs, Dict):

            obs = obs['state_observation']
           
        fingerCOM , doorGraspPoint = obs[:3], obs[3:6]


       

        doorAngleTarget = self._state_goal
       
       
        graspDist = np.linalg.norm(doorGraspPoint - fingerCOM)
        
        

        graspRew = -graspDist

        def doorOpenReward(doorAngle):


            

            #angleDiff = np.linalg.norm(doorAngle - doorAngleTarget)

            doorRew = 0
            if graspDist < 0.1:

                if doorAngle <= doorAngleTarget:

                    doorRew =  max(10 * doorAngle,0)

                elif doorAngle> doorAngleTarget:
                    doorRew =  max(10*(doorAngleTarget -  (doorAngle - doorAngleTarget)),0)


                
            return doorRew

        
        doorAngle = self.data.get_joint_qpos('doorjoint')

        doorOpenRew = doorOpenReward(doorAngle)

        reward = graspRew + doorOpenRew
     
       
        return [reward, doorOpenRew, doorAngle, doorAngleTarget] 
      
   

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
       
        return statistics


  
  