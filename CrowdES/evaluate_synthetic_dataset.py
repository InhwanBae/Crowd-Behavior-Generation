import os

from CrowdES.inference_model import CrowdESFramework
from utils.dataloader.synthetic_dataloader import SyntheticDataset
from utils.utils import reproducibility_settings
from utils.visualization import produce_video_from_data


# Global settings
TRIALS = 1                      # Number of trials for each scene
SCENE_LIST = ['synth_scurve',]  # List of scenes to use for inference
SCENARIO_LENGTH = 30 * 60 * 10  # Scenario length in frames (30fps * 60s * 10min)
POSTFIX = 'crowdes-synthetic'   # Postfix for generated files
EXPORT_VIDEO = True             # Export video for visualization


def main(config, seed=0):
    # Reproducibility
    reproducibility_settings(seed=seed)

    # Load dataset and framework
    dataset_test = SyntheticDataset(config, SCENE_LIST)  # ['synth_scurve', 'synth_maze', 'synth_bottleneck', 'cathedral', 'fernsehturm', 'park', 'manhattan']
    framework = CrowdESFramework(config)

    # Start inference
    scene_list = dataset_test.scene_list

    for scene_idx, scene in enumerate(scene_list):
        for trial in range(TRIALS):
            print(f'Scene {scene_idx + 1}/{len(scene_list)}: {scene}, Trial {trial + 1}/{TRIALS}')

            # Load data
            data = dataset_test[scene_idx]
            scene_img = data['img']
            scene_seg = data['seg']
            scene_H = data['H']
            scene_appearance = data['appearance']
            scene_population = data['population']
            scene_walkable = data['walkable']
            scene_navmesh = data['navmesh']
            # Inference
            scenario_length = SCENARIO_LENGTH
            framework.initialize_scene(scene_img, scene_seg, scene_walkable, scene_navmesh, scene_H, appearance_gt=scene_appearance, population_gt=scene_population)
            generated_scenario = framework.generate(scenario_length, seed=trial)
            generated_scenario['scene'] = scene
            print(f"Generated scenario: {len(generated_scenario['agent_id'].unique())} total agents")

            # Save generated scenario
            columns=['scene', 'agent_id', 'agent_type', 'frame', 'x', 'y']
            generated_scenario = generated_scenario[columns]
            os.makedirs("output/generated/synthetic", exist_ok=True)

            generated_scenario.to_csv(f'./output/generated/synthetic/{scene}-{POSTFIX}-{trial}.csv', index=False)

            # Export video for visualization
            if EXPORT_VIDEO:
                scene_bg = data['bg']
                video_path = f'./output/generated/synthetic/{scene}-{POSTFIX}-{trial}.mp4'
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                produce_video_from_data(video_path, scene_img, scene_bg, generated_scenario, scenario_length, config)
                print(f'Video saved at {video_path}')

    return 
