import argparse
from utils.config import get_config, print_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='./configs/model/CrowdES_gcs.yaml', help='Path to a model config file')
    parser.add_argument('--dataset_config', type=str, default=None, help='Path to a trainer config file (optional). If not provided, the default config will be used.')
    parser.add_argument('--trainer_config', type=str, default=None, help='Path to a trainer config file (optional). If not provided, the default config will be used.')
    parser.add_argument('--model_train', type=str, default='emitter', help='Stage of the experiment', choices=['emitter_pre', 'emitter', 'simulator'])
    parser.add_argument('--test', default=False, action='store_true', help='Evaluation mode.')
    parser.add_argument('--export', default=False, action='store_true', help='Visualization mode.')
    parser.add_argument('--synthetic', default=False, action='store_true', help='Use synthetic dataset for inference.')
    args = parser.parse_args()

    config = get_config(args.model_config, args.dataset_config, args.trainer_config)
    # Print the arguments and configs
    print('===== Arguments =====')
    print_arguments(vars(args))

    print('===== Configs =====')
    print_arguments(config)

    # Import the appropriate pipeline
    if args.test:
        from CrowdES.evaluate import *
    elif args.export:
        from CrowdES.evaluate_export_generated_traj import *
    elif args.synthetic:
        from CrowdES.evaluate_synthetic_dataset import *
    elif args.model_train == 'emitter_pre':
        from CrowdES.emitter.emitter_pre_trainer import *
    elif args.model_train == 'emitter':
        from CrowdES.emitter.emitter_trainer import *
    elif args.model_train == 'simulator':
        from CrowdES.simulator.simulator_trainer import *
        
    main(config)
