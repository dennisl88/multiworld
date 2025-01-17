import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld mujoco gym environments")

    """
    Reaching tasks
    """
    register(
        id='SawyerReachXYEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'vitchyr'
        },
        kwargs={
            'hide_goal_markers': False,
        },
    )
    register(
        id='Image48SawyerReachXYEnv-v0',
        entry_point=create_image_48_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'vitchyr'
        },
    )
    register(
        id='Image84SawyerReachXYEnv-v0',
        entry_point=create_image_84_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'vitchyr'
        },
    )

    """
    Pushing tasks, XY, With Reset
    """
    register(
        id='SawyerPushAndReacherXYEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '3503e9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            hide_goal_markers=True,
            action_scale=.02,
            puck_low=[-0.25, .4],
            puck_high=[0.25, .8],
            mocap_low=[-0.2, 0.45, 0.],
            mocap_high=[0.2, 0.75, 0.5],
            goal_low=[-0.2, 0.45, 0.02, -0.25, 0.4],
            goal_high=[0.2, 0.75, 0.02, 0.25, 0.8],
        )
    )
    register(
        id='Image48SawyerPushAndReacherXYEnv-v0',
        entry_point=create_Image48SawyerPushAndReacherXYEnv_v0,
        tags={
            'git-commit-hash': '3503e9f',
            'author': 'vitchyr'
        },
    )
    register(
        id='SawyerPushAndReachXYEasyEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'fec148f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.05, 0.4, 0.02, -.1, .5),
            goal_high=(0.05, 0.7, 0.02, .1, .7),
        )
    )
    register(
        id='Image48SawyerPushAndReachXYEasyEnv-v0',
        entry_point=create_image_48_sawyer_reach_and_reach_xy_easy_env_v0,
        tags={
            'git-commit-hash': 'fec148f',
            'author': 'vitchyr'
        },
    )

    """
    Pushing tasks, XY, Reset Free
    """
    register(
        id='SawyerPushAndReacherXYEnv-ResetFree-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '3d4adbe',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )
    register(
        id='SawyerPushXYEnv-ResetFree-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )
    register(
        id='SawyerPushAndReachXYEnv-ResetFree-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=1,
        )
    )
    register(
        id='SawyerPushAndReachXYEnv-ResetFree-Every1B-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=int(1e9),
        )
    )
    register(
        id='SawyerPushAndReachXYEnv-ResetFree-Every2-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=2,
        )
    )
    register(
        id='SawyerPushAndReachXYEnv-ResetFree-Every3-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=3,
        )
    )

    """
    Push XYZ
    """
    register(
        id='SawyerPushXyzEasyEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.05, 0.4, 0.02, -.1, .5),
            goal_high=(0.05, 0.7, 0.02, .1, .7),
        )
    )
    register(
        id='SawyerPushAndReachXyzEasyEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.05, 0.4, 0.02, -.1, .5),
            goal_high=(0.05, 0.7, 0.02, .1, .7),
        )
    )
    register(
        id='SawyerPushXyzFullArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )
    register(
        id='SawyerPushAndReachXyzFullArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )


def create_image_48_sawyer_reach_xy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera

    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera

    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera,
        transpose=True,
        normalize=True,
    )


def create_image_48_sawyer_reach_and_reach_xy_easy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachXYEasyEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )


def create_Image48SawyerPushAndReacherXYEnv_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_top_down

    wrapped_env = gym.make('SawyerPushAndReacherXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_top_down,
        transpose=True,
        normalize=True,
    )


register_custom_envs()
