import os
import math
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from scipy.interpolate import PchipInterpolator
import cv2
from PIL import Image
from diffusers import DDIMScheduler, DDPMScheduler

from CrowdES.emitter.emitter_pre_config import CrowdESEmitterPreConfig
from CrowdES.emitter.emitter_pre_model import CrowdESEmitterPreModel
from CrowdES.emitter.emitter_model import CrowdESEmitterModel
from CrowdES.emitter.emitter_pipeline import CrowdESEmitterPipeline
from CrowdES.simulator.simulator_config import CrowdESSimulatorConfig
from CrowdES.simulator.simulator_model import CrowdESSimulatorModel

from utils.utils import reproducibility_settings, ProgressParallel
from utils.image import sampling_xy_pos
from utils.homography import image2world, world2image
from utils.navmesh import get_control_point, filter_collinear_polygons, PathFinderNew
from utils.trajectory import preprocess_kdtree, batched_nearest_nonzero_idx_kdtree, KalmanModel


SAVE_IMAGE_FOR_DEBUG = False  # save image for debug
VERBOSE = False               # print debug information

# Controllability parameters
CONTROL_POPULATION_MULTIPLIER = None  # Population multiplier to control, None: no control, float: multiplier
CONTROL_POPULATION_SIZE = None        # Population size to control, None: no control, int: population size
CONTROL_AGENT_TYPE = None             # Agent type to control, None: no control, 1: pedestrian, 2: bicycle, 3: vehicle
CONTROL_WALKING_PACE = None           # Walking pace to control, None: no control, float: walking pace in m/s


class CrowdESFramework:
    """CrowdES framework for inference."""

    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset.dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained model
        CrowdES_emitter_pre_checkpoint_dir = config.crowd_emitter.emitter_pre.checkpoint_dir.format(self.dataset_name)
        self.CrowdES_emitter_pre = CrowdESEmitterPreModel.from_pretrained(CrowdES_emitter_pre_checkpoint_dir)
        self.CrowdES_emitter_pre.to(self.device)
        self.CrowdES_emitter_pre.eval()

        CrowdES_emitter_checkpoint_dir = config.crowd_emitter.emitter.checkpoint_dir.format(self.dataset_name)
        self.CrowdES_emitter = CrowdESEmitterModel.from_pretrained(CrowdES_emitter_checkpoint_dir, use_safetensors=False)
        self.CrowdES_emitter.to(self.device)
        self.CrowdES_emitter.eval()
        CrowdES_emitter_diffusion_steps = config.crowd_emitter.emitter.diffusion_steps
        CrowdES_emitter_noise_scheduler = DDIMScheduler(num_train_timesteps=CrowdES_emitter_diffusion_steps, beta_schedule='linear')
        self.CrowdES_emitter_pipeline = CrowdESEmitterPipeline(unet=self.CrowdES_emitter, scheduler=CrowdES_emitter_noise_scheduler)

        CrowdES_simulator_checkpoint_dir = config.crowd_simulator.simulator.checkpoint_dir.format(self.dataset_name)
        self.CrowdES_simulator = CrowdESSimulatorModel.from_pretrained(CrowdES_simulator_checkpoint_dir)
        self.CrowdES_simulator.to(self.device)
        self.CrowdES_simulator.eval()

        # Set hyperparameters
        self.dataset_fps = self.config.dataset.dataset_fps
        self.simulator_fps = self.config.crowd_simulator.simulator.simulator_fps
        self.window_frame = self.config.crowd_emitter.emitter.window_size * self.simulator_fps
        self.traj_fut_frame = self.config.crowd_simulator.simulator.future_length

        # Prepare ORCA simulator if specified
        if self.config.crowd_simulator.type == 'ORCA':
            navmesh_vertices = [[x, 0, y] for x, y in image2world(np.array(self.navmesh['vertices'])[..., [0, 1]], self.H)]  # (w, h) -> (x, y) -> (x, 0, y)
            navmesh_polygons = self.navmesh['polygons']
            self.pf = PathFinderNew(navmesh_vertices, navmesh_polygons)
            self.sim_agent_id_to_agent_id = {}

    def initialize_scene(self, img, seg, walkable, navmesh, H, appearance_gt=None, population_gt=None):
        """Initialize the scene for crowd emitter.

        Params:
            img (np.array): An image.
            seg (np.array): A segmentation map.
            walkable (np.array): A walkable map.
            navmesh (list): A navigation mesh.
            H (np.array): A homography matrix.
            appearance_gt (np.array, optional): A ground truth appearance density map. Defaults to None.
            population_gt (np.array, optional): A ground truth population density map. Defaults to None.
        """

        self.img = img
        self.seg = seg
        self.walkable = walkable
        self.navmesh = navmesh
        self.H = H
        self.scene_size = self.img.shape[:2]  # (H, W)

        self.appearance_gt = appearance_gt
        self.population_gt = population_gt

        # Preparation
        self.navmesh['polygons'] = filter_collinear_polygons(navmesh['vertices'], navmesh['polygons'])
        self.env_pixel_meter = self.config.crowd_simulator.simulator.environment_pixel_meter
        self.env_size = tuple(self.config.crowd_simulator.simulator.environment_size)
        env_base_pointer_h = np.linspace(-self.env_size[0] / 2, self.env_size[0] / 2, self.env_size[0])
        env_base_pointer_w = np.linspace(-self.env_size[1] / 2, self.env_size[1] / 2, self.env_size[1])
        self.env_base_pointer = np.stack(np.meshgrid(env_base_pointer_h, env_base_pointer_w), axis=-1) * self.env_pixel_meter  # shape=(64, 64, 2)
        self.kdtree = preprocess_kdtree(self.walkable)

    def process_emitter_pre(self):
        # Extract the scene layouts from the image
        process_image_size = tuple(self.config.crowd_emitter.emitter_pre.image_size)
        input_types = self.config.crowd_emitter.emitter_pre.input_types
        input = []
        input.append(self.img) if 'scene_img' in input_types else None
        input.append(self.seg) if 'scene_seg' in input_types else None
        input = [(i[:, :, None] if i.ndim == 2 else i) for i in input]
        input = np.concatenate(input, axis=2)
        input = torch.FloatTensor(input).permute(2, 0, 1)
        input = torch.nn.functional.interpolate(input.unsqueeze(0), size=process_image_size, mode='bilinear', align_corners=False)

        input = input.to(self.device)
        outputs = self.CrowdES_emitter_pre(input)

        pred_batch = outputs.logits
        pred_batch = torch.nn.functional.interpolate(pred_batch, size=self.scene_size, mode='bilinear', align_corners=False)
        temperature = self.config.crowd_emitter.emitter_pre.population_density_temperature
        pred_batch = torch.sigmoid(pred_batch / temperature).squeeze(0)  # Sigmoid with steep slope using temperature

        appearance_density_map = pred_batch[0].detach().cpu().numpy()
        population_density_map = pred_batch[1].detach().cpu().numpy()
        appearance_density_map = appearance_density_map / appearance_density_map.max()  # normalize
        population_density_map = population_density_map / population_density_map.max()  # normalize
        self.appearance_density_map = appearance_density_map  # We will limit the appearance points to the walkable area in the emitter.
        self.population_density_map = population_density_map * self.walkable

        # Put the ground truth if it is not None
        if self.appearance_gt is not None:
            assert self.appearance_gt.shape == self.appearance_density_map.shape
            self.appearance_density_map = self.appearance_gt
        if self.population_gt is not None:
            assert self.population_gt.shape == self.population_density_map.shape
            self.population_density_map = self.population_gt

        # Make the population probability to 0 if the value is too small.
        pred_logit_batch = outputs.logits_unit.squeeze(0).sigmoid().detach().cpu().numpy()
        population_probability_threshold = self.config.crowd_emitter.emitter_pre.population_probability_threshold
        pred_logit_batch[pred_logit_batch < population_probability_threshold] = 0
        self.population_probability = pred_logit_batch / pred_logit_batch.sum()  # normalize

        # Prepare conditions for Emitter
        input_types_e = self.config.crowd_emitter.emitter.input_types
        environment_e = []  # TODO: move condition to process_emitter_pre
        environment_e.append(self.img) if 'scene_img' in input_types_e else None
        environment_e.append(self.seg) if 'scene_seg' in input_types_e else None
        environment_e.append(self.population_density_map) if 'scene_population' in input_types_e else None
        environment_e.append(self.appearance_density_map) if 'scene_appearance' in input_types_e else None
        environment_e = [(c[:, :, None] if c.ndim == 2 else c) for c in environment_e]
        self.environment_e = np.concatenate(environment_e, axis=2)

        # Prepare conditions for Simulator
        environment_types_s = self.config.crowd_simulator.simulator.environment_types
        environment_s = []
        environment_s.append(self.img) if 'scene_img' in environment_types_s else None
        environment_s.append(self.seg) if 'scene_seg' in environment_types_s else None
        environment_s.append(self.population_density_map) if 'scene_population' in environment_types_s else None
        environment_s.append(self.appearance_density_map) if 'scene_appearance' in environment_types_s else None
        environment_s = [(c[:, :, None] if c.ndim == 2 else c) for c in environment_s]
        self.environment_s = np.concatenate(environment_s, axis=2)
        self.environment_s = torch.FloatTensor(self.environment_s).permute(2, 0, 1)

        return self.appearance_density_map, self.population_density_map, self.population_probability

    def process_emitter(self):
        # Step 1: Plan the number of agents to emit
        max_num_agents_pad = self.config.crowd_emitter.emitter.max_num_agents_pad
        num_agents = np.random.choice(len(self.population_probability), p=self.population_probability)
        num_agents = int(num_agents * self.config.crowd_emitter.emitter_pre.population_multiplier)
        
        # Controllability - Population size
        num_agents = num_agents if CONTROL_POPULATION_MULTIPLIER is None else int(num_agents * CONTROL_POPULATION_MULTIPLIER)
        num_agents = num_agents if CONTROL_POPULATION_SIZE is None else CONTROL_POPULATION_SIZE

        num_agents_to_emit = num_agents - len(self.agent_ids_in_current_scene)
        num_agents_to_emit = min(num_agents_to_emit, max_num_agents_pad)

        # For initialization, cut down the number of agents in half
        if self.current_frame < 0:
            num_agents_to_emit = math.ceil(num_agents_to_emit / 2)

        if num_agents_to_emit <= 0:
            print(f'Skip frame {self.current_frame} because there are enough agents in the scene.') if VERBOSE else None
            self.statistics_added = 0
            return self.agent_ids_in_current_scene
        
        # Step 2: Prepare conditions
        # Make binary map of individual positions in the previous frame.
        process_image_size = tuple(self.config.crowd_emitter.emitter.image_size)
        agent_dist = np.zeros(self.scene_size)
        if len(self.agent_ids_in_current_scene) > 0:
            agent_coord = [self.agent_trajectory[agent_id][-1] if len(self.agent_trajectory[agent_id]) > 0 else np.empty((2,)) for agent_id in self.agent_ids_in_current_scene]
            agent_coord = np.stack(agent_coord, axis=0)
            agent_coord = world2image(agent_coord, self.H)
            agent_coord = agent_coord.astype(int)
            agent_coord = np.clip(agent_coord, 0, np.array(self.scene_size)[[1, 0]] - 1)
            agent_dist[agent_coord[:, 1], agent_coord[:, 0]] = 1
        
        # Make the condition for the crowd emitter
        condition = []
        condition.append(self.environment_e)
        condition.append(agent_dist)
        condition = [(c[:, :, None] if c.ndim == 2 else c) for c in condition]
        condition = np.concatenate(condition, axis=2)
        condition = torch.FloatTensor(condition).permute(2, 0, 1)
        condition = torch.nn.functional.interpolate(condition.unsqueeze(0), size=process_image_size, mode='bilinear', align_corners=False)
        condition = condition.to(self.device)

        # Step 3: Populate the scene with crowd emitter diffusion model
        # Populate the scene with agents
        diffusion_steps = self.config.crowd_emitter.emitter.diffusion_steps
        leapfrog_steps = self.config.crowd_emitter.emitter.leapfrog_steps
        seed = np.random.randint(0, 100000)  # For reproducibility. we already set the global seed
        gen_crowd = self.CrowdES_emitter_pipeline(batch_size=1, condition=condition, num_crowd=num_agents_to_emit, device=self.device, seed=seed, num_inference_steps=diffusion_steps, leapfrog_steps=leapfrog_steps)
        
        # Generated agent parameters: [agent_type, frame_origin, preferred_speed, origin_x, origin_y, goal_x, goal_y]
        gen_crowd = gen_crowd['crowd_emission'].detach().cpu().numpy()[0]
        gen_crowd[:, [3, 5]] *= self.scene_size[1]  # x*w
        gen_crowd[:, [4, 6]] *= self.scene_size[0]  # y*h

        # Move origin_xy and goal_xy to the nearest non-zero points on the walkable map
        origin_xy = gen_crowd[:, [3, 4]]
        goal_xy = gen_crowd[:, [5, 6]]
        origin_xy = batched_nearest_nonzero_idx_kdtree(self.kdtree, origin_xy)
        goal_xy = batched_nearest_nonzero_idx_kdtree(self.kdtree, goal_xy)
        gen_crowd[:, [3, 4]] = origin_xy
        gen_crowd[:, [5, 6]] = goal_xy
        gen_crowd[:, 1] = np.round(gen_crowd[:, 1] * self.window_frame)

        # Update the scenario
        for j in range(num_agents_to_emit):
            self.last_agent_id += 1
            agent_id = self.last_agent_id
            self.new_agent_ids.append(agent_id)
            data = gen_crowd[j]
            agent_params = {'agent_type': int(data[0]), 'frame_origin': int(data[1]), 'preferred_speed': data[2], 'origin_xy': data[3:5], 'goal_xy': data[5:7]}
            agent_params['frame_global'] = self.current_frame + agent_params['frame_origin']
            self.agent_parameter[agent_id] = agent_params
            self.agent_trajectory[agent_id] = np.empty((0, 2))
            self.agent_previous_behavior_states[agent_id] = np.zeros(self.CrowdES_simulator.config.latent_dim)

            # Controllability - Agent type
            if CONTROL_AGENT_TYPE is not None:
                self.agent_parameter[agent_id]['agent_type'] = CONTROL_AGENT_TYPE

            # Controllability - Walking pace
            if CONTROL_WALKING_PACE is not None:
                self.agent_parameter[agent_id]['preferred_speed'] = CONTROL_WALKING_PACE
        
        self.statistics_added = num_agents_to_emit

        # Visualize the crowd emitter output
        if SAVE_IMAGE_FOR_DEBUG:
            crowd_emitter_result = self.appearance_density_map.copy()
            crowd_emitter_result = (crowd_emitter_result * 255).astype(np.uint8)
            crowd_emitter_result = np.stack([crowd_emitter_result] * 3, axis=-1)

            pred_crowd_origin = (gen_crowd[:, [3, 4]]).astype(np.int32)  # (W, H)
            pred_crowd_goal = (gen_crowd[:, [5, 6]]).astype(np.int32)  # (W, H)
            
            pred_crowd_color = np.random.rand(1000, 3) * 255
            for j in range(num_agents_to_emit):
                cv2.circle(crowd_emitter_result, tuple(pred_crowd_origin[j]), 8, pred_crowd_color[j], -1)
                cv2.circle(crowd_emitter_result, tuple(pred_crowd_goal[j]), 7, pred_crowd_color[j], 2)
            cv2.putText(crowd_emitter_result, str(num_agents_to_emit), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            os.makedirs('temp', exist_ok=True)
            crowd_emitter_result = Image.fromarray(crowd_emitter_result)
            crowd_emitter_result.save(f'temp/crowd_emitter_result_{self.current_frame}.png')

    def process_surface_emitter(self):
        # Step 1: Plan the number of agents to emit
        max_num_agents_pad = self.config.crowd_emitter.emitter.max_num_agents_pad
        num_agents = np.random.choice(len(self.population_probability), p=self.population_probability)
        num_agents = int(num_agents * self.config.crowd_emitter.emitter_pre.population_multiplier)
        num_agents_to_emit = num_agents - len(self.agent_ids_in_current_scene)
        num_agents_to_emit = min(num_agents_to_emit, max_num_agents_pad)

        # For initialization, cut down the number of agents in half
        if self.current_frame < 0:
            num_agents_to_emit = math.ceil(num_agents_to_emit / 2)

        if num_agents_to_emit <= 0:
            print(f'Skip frame {self.current_frame} because there are enough agents in the scene.') if VERBOSE else None
            self.statistics_added = 0
            return self.agent_ids_in_current_scene
        
        # Step 2: Populate the scene with surface emitter
        gen_crowd = np.random.randn(num_agents_to_emit, 7)
        norm_mean = self.CrowdES_emitter.norm_mean.detach().cpu().numpy()
        norm_std = self.CrowdES_emitter.norm_std.detach().cpu().numpy()
        gen_crowd = gen_crowd * norm_std + norm_mean
        
        # Generated agent parameters: [agent_type, frame_origin, preferred_speed, origin_x, origin_y, goal_x, goal_y]       
        gen_crowd[:, 0] = np.round(np.clip(gen_crowd[:, 0], 0, 2))  # round agent_type to 0, 1 or 2
        gen_crowd[:, 1] = np.random.randint(0, self.window_frame, size=(num_agents_to_emit,))
        gen_crowd[:, 2] = np.clip(gen_crowd[:, 2], 0, float('inf'))
        gen_crowd[:, [3, 4]] = sampling_xy_pos(torch.FloatTensor(self.walkable).unsqueeze(dim=0), num_agents_to_emit).squeeze(dim=0).numpy()
        gen_crowd[:, [5, 6]] = sampling_xy_pos(torch.FloatTensor(self.walkable).unsqueeze(dim=0), num_agents_to_emit).squeeze(dim=0).numpy()

        # Update the scenario
        for j in range(num_agents_to_emit):
            self.last_agent_id += 1
            agent_id = self.last_agent_id
            self.new_agent_ids.append(agent_id)
            data = gen_crowd[j]
            agent_params = {'agent_type': int(data[0]), 'frame_origin': int(data[1]), 'preferred_speed': data[2], 'origin_xy': data[3:5], 'goal_xy': data[5:7]}
            agent_params['frame_global'] = self.current_frame + agent_params['frame_origin']
            self.agent_parameter[agent_id] = agent_params
            self.agent_trajectory[agent_id] = np.empty((0, 2))
            self.agent_previous_behavior_states[agent_id] = np.zeros(self.CrowdES_simulator.config.latent_dim)

        self.statistics_added = num_agents_to_emit

    def process_simulator(self):
        # Step 1: Repeat simulation for (self.window_frame / future_length) recurrent times
        T_hist = self.config.crowd_simulator.simulator.history_length
        T_fut = self.traj_fut_frame
        interaction_range = self.config.crowd_simulator.simulator.interaction_range
        interaction_max_num_agents = self.config.crowd_simulator.simulator.interaction_max_num_agents
        control_time_offset = self.config.crowd_simulator.simulator.control_time_offset
        
        before_num_agents = len(self.agent_ids_in_current_scene)

        for recurrent in range(self.window_frame // self.traj_fut_frame):
            # Step 2. Add new agents into current scene
            for agent_id in self.new_agent_ids:
                frame_origin = self.agent_parameter[agent_id]['frame_origin']
                if T_fut * recurrent <= frame_origin < T_fut * (recurrent + 1):
                    self.agent_ids_in_current_scene.append(agent_id)
                    
            if len(self.agent_ids_in_current_scene) == 0:
                continue
            
            # 3. Generate the future trajectory of length future_length
            B = len(self.agent_ids_in_current_scene)
            traj_hist = np.zeros((B, T_hist, 2))
            goal = np.zeros((B, 2))
            attr = np.zeros((B, 2))
            
            for idx, agent_id in enumerate(self.agent_ids_in_current_scene):
                agent_trajectory = self.agent_trajectory[agent_id]
                if len(agent_trajectory) < T_hist:
                    origin_xy = self.agent_parameter[agent_id]['origin_xy']
                    traj_hist[idx] = np.array(origin_xy)[None, :]
                    if len(agent_trajectory) > 0:
                        traj_hist[idx, -len(agent_trajectory):] = np.array(agent_trajectory)
                else:
                    traj_hist[idx] = np.array(agent_trajectory[-T_hist:])

                goal[idx] = np.array(self.agent_parameter[agent_id]['goal_xy'])
                attr[idx] = np.array([self.agent_parameter[agent_id]['agent_type'], self.agent_parameter[agent_id]['preferred_speed']])

            # Prepare the control points
            start_points = traj_hist[:, -1]
            finish_points = goal
            distances = attr[:, 1] * control_time_offset
            control_points_meter = get_control_point(self.navmesh['vertices'], self.navmesh['polygons'], start_points, finish_points, self.H, distances)
            control_points_meter = np.array(control_points_meter) # Note: control points are already in meter coordinates

            traj_hist_meter = image2world(traj_hist, self.H)
            goal_meter = image2world(goal, self.H)

            # Aggregate neighborhood trajectories
            startpoint_meter = traj_hist_meter[:, -1]
            distance_matrix = np.linalg.norm(startpoint_meter[None, :, :] - startpoint_meter[:, None, :], axis=2)
            neighbor_index = np.argsort(distance_matrix, axis=1)[:, 1:interaction_max_num_agents+1]
            neighbor_mask = distance_matrix[np.arange(B)[:, None], neighbor_index] < interaction_range
            neighbor_index = [neighbor_index[i][neighbor_mask[i]] for i in range(B)]
            # Fill the neighbor matrix with traj_hist_meter
            neighbor_meter = np.zeros((B, interaction_max_num_agents, T_hist, 2))
            for idx, neighbor_idx in enumerate(neighbor_index):
                neighbor_meter[idx, :len(neighbor_idx)] = traj_hist_meter[neighbor_idx]

            # Normalize with startpoint_meter
            traj_hist_meter_norm = traj_hist_meter - startpoint_meter[:, None, :]
            goal_meter_norm = goal_meter - startpoint_meter
            control_points_meter_norm = control_points_meter - startpoint_meter
            neighbor_meter_norm = neighbor_meter - startpoint_meter[:, None, None, :]

            # Build environmental data
            _, h, w = self.environment_s.shape
            env_pointer_meter = self.env_base_pointer[None, :] + startpoint_meter[:, None, None, [1, 0]]  # (B, 64, 64, 2) + (B, None, None, (2=xy -> hw in meter))
            env_pointer_pixel = world2image(env_pointer_meter.transpose(0, 2, 1, 3), self.H).transpose(0, 2, 1, 3)  # hw -> wh
            env_pointer_pixel = env_pointer_pixel.round().astype(int)
            env_pointer_pixel = np.clip(env_pointer_pixel, 0, [h-1, w-1])
            env_data_crop = self.environment_s[:, env_pointer_pixel[..., 0], env_pointer_pixel[..., 1]]
            env_data_crop = env_data_crop.permute(1, 0, 2, 3) if len(env_data_crop.shape) == 4 else env_data_crop
            
            # Aggregate previous behavior states
            previous_states = np.zeros((B, self.CrowdES_simulator.config.latent_dim))
            for idx, agent_id in enumerate(self.agent_ids_in_current_scene):
                previous_states[idx] = self.agent_previous_behavior_states[agent_id]

            # Convert to torch tensors
            traj_hist_meter_norm = torch.FloatTensor(traj_hist_meter_norm).to(self.device)
            goal_meter_norm = torch.FloatTensor(goal_meter_norm).to(self.device)
            attr = torch.FloatTensor(attr).to(self.device)
            control_points_meter_norm = torch.FloatTensor(control_points_meter_norm).to(self.device)
            neighbor_meter_norm = torch.FloatTensor(neighbor_meter_norm).to(self.device)
            env_data_crop = env_data_crop.to(self.device)
            previous_states = torch.FloatTensor(previous_states).to(self.device)

            # Predict the future trajectories of the crowds
            outputs = self.CrowdES_simulator(traj_hist_meter_norm, None, goal_meter_norm, attr, control_points_meter_norm, neighbor_meter_norm, env_data_crop, sampling=True, previous_state=previous_states, tmp=self.config.crowd_simulator.simulator.latent_temperature, alpha=self.config.crowd_simulator.simulator.latent_mixup_alpha)
            behavior_states = outputs.states.detach().cpu().numpy()
            traj_fut_meter_norm = outputs.preds.detach().cpu().numpy()
            traj_fut_meter_norm = traj_fut_meter_norm[:, :T_fut]  # Make sure
            assert traj_fut_meter_norm.shape[1] == T_fut  # Make sure
            
            traj_fut_meter = traj_fut_meter_norm + startpoint_meter[:, None, :]
            traj_fut = world2image(traj_fut_meter, self.H)  # B, T_fut, 2

            # Shift the trajectory coordinates into the walkable area
            traj_fut = batched_nearest_nonzero_idx_kdtree(self.kdtree, traj_fut)

            # Step 4: Update the scene
            for idx, agent_id in list(enumerate(self.agent_ids_in_current_scene)):
                # Update the behavior states
                self.agent_previous_behavior_states[agent_id] = behavior_states[idx]

                if agent_id in self.new_agent_ids:
                    # If the agent is new in the scene, trim the trajectory with frame_origin
                    frame_origin = self.agent_parameter[agent_id]['frame_origin']
                    frame_origin_norm = min(max(0, frame_origin - T_fut * recurrent), T_fut-1)  # Make sure
                    traj_agent = traj_fut[idx, :T_fut-frame_origin_norm]  # from start
                    # traj_agent = traj_fut[idx, frame_origin_norm:]  # from end
                    self.new_agent_ids.remove(agent_id)
                else:
                    traj_agent = np.concatenate([self.agent_trajectory[agent_id], traj_fut[idx]], axis=0)

                # Step 5: Pop the agent if the agent out of the scene or the agent reach the goal
                traj_agent_sceneoutmask = ((traj_fut[idx] <= 0) | (traj_fut[idx] >= np.array(self.scene_size)[[1, 0]] - 1)).any(axis=1)
                traj_agent_goal = np.linalg.norm(traj_fut_meter[idx] - goal_meter[idx][None, :], axis=1)
                traj_agent_goalmask = traj_agent_goal < 0.5  # 0.5 meter threshold to consider the agent reached the goal

                if traj_agent_sceneoutmask.any():
                    cut_pivot = np.where(traj_agent_sceneoutmask)[0][0]
                    cut_pivot_from_end = T_fut - cut_pivot  # It should not be 0 or negative
                    traj_agent = traj_agent[:-cut_pivot_from_end]
                    self.agent_ids_in_current_scene.remove(agent_id)
                    
                elif traj_agent_goalmask.any():
                    cut_pivot = np.argmin(traj_agent_goal)
                    cut_pivot_from_end = T_fut - cut_pivot + 1  # It should not be 0 or negative
                    traj_agent = traj_agent[:-cut_pivot_from_end] if cut_pivot_from_end > 0 else traj_agent
                    self.agent_ids_in_current_scene.remove(agent_id)

                # Update the trajectory
                self.agent_trajectory[agent_id] = traj_agent

        after_num_agents = len(self.agent_ids_in_current_scene)
        self.statistics_dropped = self.statistics_added + before_num_agents - after_num_agents
        print(f'Frame {self.current_frame}: {before_num_agents} -> {after_num_agents}') if VERBOSE else None

    def process_orca_simulator(self):
        # Step 1: Repeat simulation for window_frame recurrent times
        self.traj_fut_frame = 1  # for ORCA, we will simulate each frame one by one
        before_num_agents = len(self.agent_ids_in_current_scene)

        for recurrent in range(self.window_frame):
            # Step 2. Add new agents into current scene
            for agent_id in self.new_agent_ids.copy():
                frame_origin = self.agent_parameter[agent_id]['frame_origin']
                if recurrent <= frame_origin < recurrent + 1:
                    self.agent_ids_in_current_scene.append(agent_id)
                    self.new_agent_ids.remove(agent_id)

                    # Set agent radius based on agent type (0: pedestrian, 1: bicycle, 2: vehicle)
                    agent_type = self.agent_parameter[agent_id]['agent_type']
                    agent_radius = 0.2 * 2
                    agent_radius = 0.5 * 2 if agent_type == 1 else agent_radius
                    agent_radius = 1.0 * 2 if agent_type == 2 else agent_radius

                    # Add agent to the ORCA simulator
                    origin_xy = image2world(self.agent_parameter[agent_id]['origin_xy'], self.H)
                    goal_xy = image2world(self.agent_parameter[agent_id]['goal_xy'], self.H)
                    preferred_speed = self.agent_parameter[agent_id]['preferred_speed']      
                    sim_agent_id = self.pf.add_agent((origin_xy[0], 0, origin_xy[1]), agent_radius, preferred_speed)

                    if sim_agent_id == -1:
                        self.agent_ids_in_current_scene.remove(agent_id)
                        continue
                    
                    try:
                        self.pf.set_agent_destination(sim_agent_id, (goal_xy[0], 0, goal_xy[1]))
                    except:
                        print(f'ORCA error at frame {self.current_frame}')  # ORCA bug that cannot be fixed
                        break
                    self.sim_agent_id_to_agent_id[sim_agent_id] = agent_id

                    # add agent trajectories
                    origin_xy_image = self.agent_parameter[agent_id]['origin_xy']
                    self.agent_trajectory[agent_id] = np.array(origin_xy_image)[None, :]  # Add start position
            
            # Step 3: Simulate the future coordinates of the crowds
            if len(self.agent_ids_in_current_scene) == 0:
                continue
            
            try:
                self.pf.update(delta_time=1/self.simulator_fps)
            except:
                print(f'ORCA error at frame {self.current_frame}')  # ORCA bug that cannot be fixed
                break
            
            # Step 4: Update the scene
            for sim_agent_id in self.pf._agents_id:
                agent_id = self.sim_agent_id_to_agent_id[sim_agent_id]
                new_pos = self.pf.get_agent_position(sim_agent_id)
                new_pos_image = world2image(np.array(new_pos), self.H)
                before_trajectory = self.agent_trajectory[agent_id]
                self.agent_trajectory[agent_id] = np.concatenate([before_trajectory, new_pos_image[None, :]], axis=0)
            
            # Step 5: Pop the agent if the agent out of the scene or the agent reach the goal
            for sim_agent_id in self.pf._agents_id:
                agent_id = self.sim_agent_id_to_agent_id[sim_agent_id]
                goal_xy = image2world(self.agent_parameter[agent_id]['goal_xy'], self.H)
                agent_pos = self.pf.get_agent_position(sim_agent_id)
                agent_pos_image = world2image(np.array(agent_pos), self.H)
                static_threshold = 10

                # Check if the agent arrived
                if np.linalg.norm(agent_pos - goal_xy) < 0.1:
                    self.agent_ids_in_current_scene.remove(agent_id)
                    self.pf.delete_agent(sim_agent_id)

                # Check if the agent out of the scene
                elif (agent_pos_image < 0).any() or (agent_pos_image >= np.array(self.scene_size)[[1, 0]]).any():
                    self.agent_ids_in_current_scene.remove(agent_id)
                    self.pf.delete_agent(sim_agent_id)

                # Compensate RVO simulation bug by checking if the agent stays more than 10 frames
                elif len(self.agent_trajectory[agent_id]) > static_threshold + 1:
                    traj = self.agent_trajectory[agent_id][-(static_threshold+1):]
                    traj_diff = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
                    if traj_diff.mean() < 0.01:
                        self.agent_ids_in_current_scene.remove(agent_id)
                        self.pf.delete_agent(sim_agent_id)

        after_num_agents = len(self.agent_ids_in_current_scene)
        self.statistics_dropped = self.statistics_added + before_num_agents - after_num_agents
        print(f'Frame {self.current_frame}: {before_num_agents} -> {after_num_agents}') if VERBOSE else None

    def postprocess_trajectory(self):
        """Post-process the trajectory by interpolating and smoothing the trajectory."""

        parallel = len(self.agent_trajectory.keys()) > 1e10

        if not parallel:
            for agent_id in tqdm(self.agent_trajectory.keys(), desc='Post-processing the trajectory'):
                agent_id, agent_trajectory = self.postprocess_trajectory_single(agent_id)
                self.agent_trajectory[agent_id] = agent_trajectory

        else:
            agent_ids = list(self.agent_trajectory.keys())
            output = ProgressParallel(total=len(agent_ids), n_jobs=-1)(delayed(self.postprocess_trajectory_single)(agent_id) for agent_id in agent_ids)
            agent_trajectories = dict(output)
            self.agent_trajectory = agent_trajectories

    def postprocess_trajectory_single(self, agent_id):
        """Smooth the trajectory of a single agent."""
        agent_trajectory = self.agent_trajectory[agent_id]
        if len(agent_trajectory) < 2:
            return agent_id, agent_trajectory

        agent_trajectory_meter = image2world(agent_trajectory, self.H)
        agent_origin_meter = agent_trajectory_meter[0]
        agent_trajectory_meter_norm = agent_trajectory_meter - agent_origin_meter
        
        kf = KalmanModel(dt=1, n_dim=2, n_iter=1)
        smoothed_pos, smoothed_vel = kf.smooth(agent_trajectory_meter_norm)
        agent_trajectory_meter_norm = smoothed_pos

        agent_trajectory_meter = agent_trajectory_meter_norm + agent_origin_meter
        agent_trajectory = world2image(agent_trajectory_meter, self.H)

        return agent_id, agent_trajectory
    
    def densify_scenario(self, interpolation_method='linear'):
        """Densify the scenario by interpolating the trajectory."""

        upsample = self.dataset_fps // self.simulator_fps
        for agent_id in self.agent_parameter.keys():
            agent_trajectory = self.agent_trajectory[agent_id]
            agent_type = self.agent_parameter[agent_id]['agent_type']
            frame_origin = self.agent_parameter[agent_id]['frame_global']

            if len(agent_trajectory) < 2:
                continue

            # Interpolate the trajectory
            x = agent_trajectory[:, 0]
            y = agent_trajectory[:, 1]
            frame = np.arange(frame_origin, len(agent_trajectory) + frame_origin) * upsample

            if interpolation_method == 'pchip':
                pchip_x = PchipInterpolator(frame, x)
                pchip_y = PchipInterpolator(frame, y)
                frame_interp = np.arange(frame_origin * upsample, (len(agent_trajectory) + frame_origin) * upsample)
                x_interp = pchip_x(frame_interp)
                y_interp = pchip_y(frame_interp)
            elif interpolation_method == 'linear':
                frame_interp = np.arange(frame_origin * upsample, (len(agent_trajectory) + frame_origin) * upsample)
                x_interp = np.interp(frame_interp, frame, x)
                y_interp = np.interp(frame_interp, frame, y)
            else:
                raise NotImplementedError(f'Interpolation method {interpolation_method} is not implemented.')

            # Trim the trajectory to fit within the scenario length
            for i in range(len(frame_interp)):
                if 0 <= frame_interp[i] < self.scenario_len:
                    self.scenario.append([agent_id, agent_type, frame_interp[i], x_interp[i], y_interp[i]])

    def generate(self, scenario_len, seed=None):
        """Generate crowd trajectories from an image and a segmentation map.

        Params:
            img (np.array): An image.
            seg (np.array): A segmentation map.

        Returns:
            list: A list of trajectories.
        """
        
        # Reproducibility
        reproducibility_settings(seed=seed) if seed is not None else None

        self.scenario = []
        self.scenario_len = scenario_len
        self.last_agent_id = -1
        self.agent_parameter = {}
        self.agent_trajectory = {}
        self.agent_ids_in_current_scene = []
        self.new_agent_ids = []
        self.agent_previous_behavior_states = {}

        max_frame = int(np.ceil(scenario_len / self.dataset_fps * self.simulator_fps))

        # Preprocess the scene layouts for crowd emitter
        scene_layout = self.process_emitter_pre()
        appearance_density_map, population_density_map, population_probability = scene_layout
        
        # Generate crowd trajectories
        start_frames = list(range(-self.window_frame, max_frame, self.window_frame))
        pbar = tqdm(start_frames, desc='Generating crowd trajectories')
        for frame_idx in start_frames:
            self.current_frame = frame_idx

            # Emitter phase
            pbar.set_description('Crowd Emitter  ')
            if self.config.crowd_emitter.type == 'CrowdES':
                self.process_emitter()
            elif self.config.crowd_emitter.type == 'surface':
                self.process_surface_emitter()
            else:
                raise NotImplementedError(f'Emitter type {self.config.crowd_emitter.type} is not implemented.')

            # Simulator phase
            pbar.set_description('Crowd Simulator')
            if self.config.crowd_simulator.type == 'CrowdES':
                self.process_simulator()
            elif self.config.crowd_simulator.type == 'ORCA':
                self.process_orca_simulator()
            else:
                raise NotImplementedError(f'Simulator type {self.config.crowd_simulator.type} is not implemented.')

            pbar.set_postfix(population=len(self.agent_ids_in_current_scene), added=self.statistics_added, dropped=self.statistics_dropped)
            pbar.update()

            print() if VERBOSE else None

        pbar.close()

        # Post-process trajectory
        self.postprocess_trajectory()

        # Densify scenario
        self.densify_scenario()

        # Convert to pandas dataframe
        column_names = ['agent_id', 'agent_type', 'frame', 'x', 'y']
        self.scenario = pd.DataFrame(self.scenario, columns=column_names)
        self.scenario = self.scenario.astype({'agent_id': int, 'agent_type': int, 'frame': int, 'x': float, 'y': float})

        # Reorganize agent_id starting from 0
        agent_id2new_id = {agent_id: idx for idx, agent_id in enumerate(self.scenario['agent_id'].unique())}
        self.scenario['agent_id'] = self.scenario['agent_id'].map(agent_id2new_id)
        
        if SAVE_IMAGE_FOR_DEBUG:
            # Save scene layouts into temp folder
            os.makedirs(f'./output/generated/{self.dataset_name}', exist_ok=True)
            appearance_density_map = (appearance_density_map * 255).astype(np.uint8)
            population_density_map = (population_density_map * 255).astype(np.uint8)
            appearance_density_map = Image.fromarray(appearance_density_map)
            population_density_map = Image.fromarray(population_density_map)
            appearance_density_map.save(f'./output/generated/{self.dataset_name}/appearance_density_map.png')
            population_density_map.save(f'./output/generated/{self.dataset_name}/population_density_map.png')
            np.savetxt(f'./output/generated/{self.dataset_name}/population_probability.txt', population_probability, fmt='%.6f')

        return self.scenario
