from setuptools import setup, find_packages

setup(name="multiworld",
      version='0.1',
      description='Multitask Environments for RL',
      url='https://github.com/vitchyr/multiworld',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('multiworld')],
      install_requires=[
        'mujoco_py>=1.50',
        'numpy',
        'gym>=0.10',
        'matplotlib',
        'Pillow',
      ],
      package_data={'multiworld': [
        'envs/assets/classic_mujoco/*.xml',
        'envs/assets/meshes/sawyer/sawyer_ft/*.stl',
        'envs/assets/meshes/sawyer/sawyer_ft/*.DAE',
        'envs/assets/meshes/sawyer/sawyer_mp1/*.stl',
        'envs/assets/meshes/sawyer/sawyer_mp1/*.DAE',
        'envs/assets/meshes/sawyer/sawyer_mp3/*.stl',
        'envs/assets/meshes/sawyer/sawyer_mp3/*.DAE',
        'envs/assets/meshes/sawyer/sawyer_pv/*.stl',
        'envs/assets/meshes/sawyer/sawyer_pv/*.DAE',
        'envs/assets/meshes/sawyer/*.stl',
        'envs/assets/sawyer_xyz/*.xml',]
      },
      zip_safe=False)