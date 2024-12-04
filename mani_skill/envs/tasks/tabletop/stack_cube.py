from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StackCube-v1", max_episode_steps=50)
class StackCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA"
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    
from mani_skill.envs.tasks.tabletop.table_env import TableEnv

@register_env("StackCube-v2", max_episode_steps=200)
class StackCubeEnv2(TableEnv):

    SUPPORTED_ROBOTS = ["panda",  "panda_wristcam", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.1, 0.5, 0.5], [0., 0.0, -0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[0.7, 0, 0, 1], name="cubeA"
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0.7, 0, 1], name="cubeB"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.16 - 0.08
            region = [[-0.08, -0.16], [0.08, 0.16]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            
    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        obs.update(
            cubeA_pose=self.cubeA.pose.raw_pose,
            cubeB_pose=self.cubeB.pose.raw_pose,
            tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
            tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
            cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
        )
        return obs
    
    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=1.0)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~info["is_cubeA_on_cubeB"]] = 0.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
    
    
import mplib 
@register_env("StackCubeHard-v2", max_episode_steps=150)
class StackCubeHardEnv(StackCubeEnv2):
    def __init__(self, *args, num_qpos = 1, **kwargs):
        self.num_qpos = num_qpos        
        super().__init__(*args, **kwargs)
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        link_names = [link.get_name() for link in self.agent.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.agent.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf=self.unwrapped.agent.urdf_path,
            srdf=self.unwrapped.agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * 2.5,
            joint_acc_limits=np.ones(7) * 2.5,
        )
        self.planner.set_base_pose(mplib.Pose(p = self.agent.robot.pose.p[0].cpu().numpy(), q = self.agent.robot.pose.q[0].cpu().numpy()))

        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Set up the placement region
            obj_xy_range = 0.3  # Using the same range as in the old environment
            object_size = 0.04  # Using the same object size as in the old environment
            
            obj_range_low = torch.tensor([-obj_xy_range / 2, -obj_xy_range / 2, 0], device=self.device)
            obj_range_high = torch.tensor([obj_xy_range / 2, obj_xy_range / 2, 4 * object_size], device=self.device)

            # Initialize positions and orientations            
            cubeA_position = torch.zeros((b, 3), device=self.device)
            cubeB_position = torch.zeros((b, 3), device=self.device)

            valid_count = 0
            while valid_count < b:
                # Sample positions for remaining environments
                remaining = b - valid_count
                noise1 = torch.rand((remaining, 3), device=self.device) * (obj_range_high - obj_range_low) + obj_range_low
                noise2 = torch.rand((remaining, 3), device=self.device) * (obj_range_high - obj_range_low) + obj_range_low

                temp_cubeA = torch.zeros((remaining, 3), device=self.device)
                temp_cubeB = torch.zeros((remaining, 3), device=self.device)

                temp_cubeA[:, :2] = noise1[:, :2]
                temp_cubeA[:, 2] = object_size / 2

                temp_cubeB[:, :2] = noise2[:, :2]
                temp_cubeB[:, 2] = object_size / 2

                # Check distance between cubes
                distance = torch.norm(temp_cubeA - temp_cubeB, dim=1)
                valid_samples = (distance > 0.0)

                # Store valid samples
                num_valid = valid_samples.sum().item()
                cubeA_position[valid_count:valid_count+num_valid] = temp_cubeA[valid_samples]
                cubeB_position[valid_count:valid_count+num_valid] = temp_cubeB[valid_samples]

                valid_count += num_valid

            # Sample orientations (only around z-axis)
            rand_angles = torch.rand((b, 2), device=self.device) * np.pi - np.pi/2
            
            # Set poses for cubeA and cubeB
            qs = randomization.random_quaternions(
                b,
                lock_x=False,
                lock_y=False,
                lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_position, q=qs))
            qs = randomization.random_quaternions(
                b,
                lock_x=False,
                lock_y=False,
                lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_position, q=qs))
        
            qposs = []
            
            divider = max(self.num_envs // self.num_qpos, 1)
            for i in range(b):
                if i % divider < 1:
                    while True:
                        target_pose = Pose(torch.Tensor([torch.empty(1).uniform_(-obj_xy_range / 2, obj_xy_range / 2),  
                                                         torch.empty(1).uniform_(-obj_xy_range / 2, obj_xy_range / 2), 
                                                         torch.empty(1).uniform_(0.00, 0.2), 
                                                         0, 1, 0, 0]))
                        # result = self.planner.planner.plan_qpos_to_pose(
                        #     np.concatenate([target_pose.p, target_pose.q]),
                        #     self.agent.robot.get_qpos().cpu().numpy()[i],
                        #     time_step=self.control_timestep,
                        #     rrt_range=0.1,
                        #     planning_time=1.0,
                        #     use_point_cloud=False,
                        #     fix_joint_limits = True,
                        #     wrt_world = True)
                        result = self.planner.plan_pose(
                            mplib.Pose(p = target_pose.p.cpu().numpy(), q = target_pose.q.cpu().numpy()),
                            self.agent.robot.get_qpos().cpu().numpy()[0],
                            time_step=self.unwrapped.control_timestep,
                            fix_joint_limits=True,
                            rrt_range = 0.2,
                            planning_time = 1.5,
                        )
                        if result["status"] == "Success": 
                            qpos = result["position"][-1].tolist() + [0.03, 0.03]
                            break

                noise_qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (len(qpos))
                    )
                    + qpos
                )
                qposs.append(noise_qpos)
                
            qposs = np.stack(qposs)
            self.agent.reset(qposs)   

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.norm(offset[..., :2], dim=1)
            <= torch.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        is_cubeB_static = self.cubeB.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        dist_to_cubeA = torch.norm(self.cubeA.pose.p - self.agent.tcp.pose.p, dim=1) 
        success = is_cubeA_on_cubeB * (~is_cubeA_grasped) * is_cubeA_static * is_cubeB_static * (dist_to_cubeA > 0.02)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "is_cubeB_static": is_cubeB_static,
            "success": success.bool(),
        }