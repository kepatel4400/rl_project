#!/usr/bin/env python3
"""
Train a robotic pushing agent using Stable Baselines 3 SAC algorithm.
This script handles training, evaluation, logging, and visualization.
Uses YAML configuration files for systematic experiment management.
"""

import os
import time
import numpy as np
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import shutil
import logging

# Stable Baselines 3 imports
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, 
    EvalCallback, CallbackList
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

# Import agent environment
from stage_5_agent_environment_v0_obj_rand_pos import RoboticPushingEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_available = True
except ImportError:
    tensorboard_available = False
    print("TensorBoard not available. Install with: pip install tensorboard")


class RobotPushTrainer:
    """
    Trainer class for robotic pushing using Stable Baselines 3.
    
    This class encapsulates the environment setup, model configuration,
    training, evaluation, and visualization functionality.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_dir: str = None,
        task_name: str = None,
        *,
        trust_checkpoint: bool = False,
    ):
        """
        Initialize the trainer with configuration parameters from YAML.
        
        Args:
            config: Dictionary containing all configuration parameters
            experiment_dir: Root directory for the current experiment
            task_name: Name of the current task/experiment
        """
        # Store experiment info
        self.experiment_dir = experiment_dir
        self.task_name = task_name
        
        # Extract configuration parameters with defaults
        self.total_timesteps = config.get('total_timesteps', 1_000_000)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.buffer_size = config.get('buffer_size', 1_000_000)
        self.batch_size = config.get('batch_size', 256)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.gradient_steps = config.get('gradient_steps', 1)
        self.train_freq = config.get('train_freq', 1)
        self.learning_starts = config.get('learning_starts', 10000)
        self.use_sde = config.get('use_sde', True)
        self.sde_sample_freq = config.get('sde_sample_freq', -1)
        self.ent_coef = config.get('ent_coef', "auto")
        self.checkpoint_freq = config.get('checkpoint_freq', 10000)
        self.eval_freq = config.get('eval_freq', 10000)
        self.random_seed = config.get('seed', None)
        self.n_envs = config.get('n_envs', 1)
        self.parallel_envs = config.get('parallel', False)
        self.max_steps = config.get('max_steps', 1000)
        self.load_model = config.get('load_model', None)
        self.trust_checkpoint = trust_checkpoint
        if self.load_model and self.trust_checkpoint is not True:
            raise ValueError("Loading a checkpoint requires trust_checkpoint=True.")
        self.render_mode = config.get('render_mode', None)
        
        # Setup logging and model directories based on experiment structure
        if self.experiment_dir:
            # Inside experiment directory
            self.model_dir = os.path.join(self.experiment_dir, "models")
            # Shared logs in the output root folder
            self.log_dir = os.path.join(os.path.dirname(self.experiment_dir), "logs")
            self.tensorboard_log = os.path.join(self.log_dir, "tensorboard") if config.get('tensorboard', False) else None
        else:
            # Use defaults from config if no experiment directory provided
            self.log_dir = config.get('log_dir', "logs")
            self.model_dir = config.get('model_dir', "models")
            self.tensorboard_log = os.path.join(self.log_dir, "tensorboard") if config.get('tensorboard', False) else None
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        if self.tensorboard_log:
            os.makedirs(self.tensorboard_log, exist_ok=True)
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        # Initialize environment and model
        self.env = None
        self.eval_env = None
        self.model = None
        
        # Setup additional variables
        self.run_id = f"{self.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if self.task_name else f"sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure logger
        self.logger_path = os.path.join(self.log_dir, self.run_id)
        os.makedirs(self.logger_path, exist_ok=True)
        
        # Save config in the experiment directory for reproducibility
        if self.experiment_dir:
            config_path = os.path.join(self.experiment_dir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logging.info(f"Configuration saved to {config_path}")
        
    def setup_environments(self):
        """Create and configure training and evaluation environments."""
        print("Setting up environments...")
        
        # Configure environment creation function
        def make_env(rank):
            def _init():
                env = RoboticPushingEnv(
                    render_mode=self.render_mode if rank == 0 else None,
                    max_steps=self.max_steps
                )
                # Add Monitor wrapper for logging episode statistics
                env = Monitor(env, os.path.join(self.logger_path, f"env_{rank}"))
                return env
            return _init
        
        # Create vectorized environment for training
        if self.n_envs > 1:
            if self.parallel_envs:
                self.env = SubprocVecEnv([make_env(i) for i in range(self.n_envs)])
            else:
                self.env = DummyVecEnv([make_env(i) for i in range(self.n_envs)])
        else:
            self.env = DummyVecEnv([make_env(0)])
        
        # Create separate environment for evaluation
        self.eval_env = DummyVecEnv([make_env(0)])
        
        print(f"Created training environment with {self.n_envs} {'parallel' if self.parallel_envs else 'sequential'} environments")
        
    def setup_model(self):
        """Set up the SAC model with specified hyperparameters."""
        print("Setting up SAC model...")
        
        # Load pre-trained model if specified
        if self.load_model:
            print(f"Loading pre-trained model from {self.load_model}")
            self.model = self._load_checkpoint(
                self.load_model,
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                gradient_steps=self.gradient_steps,
                train_freq=self.train_freq,
                learning_starts=self.learning_starts,
                use_sde=self.use_sde,
                sde_sample_freq=self.sde_sample_freq,
                ent_coef=self.ent_coef,
                verbose=1,
                seed=self.random_seed,
            )
            # Reset replay buffer if you want to start fresh for fine-tuning
            # self.model.replay_buffer = None
        else:
            # Create a new model
            self.model = SAC(
                "MultiInputPolicy",  # Required for dict observation spaces
                self.env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                tau=self.tau,
                gradient_steps=self.gradient_steps,
                train_freq=self.train_freq,
                learning_starts=self.learning_starts,
                use_sde=self.use_sde,
                sde_sample_freq=self.sde_sample_freq,
                tensorboard_log=self.tensorboard_log,
                ent_coef=self.ent_coef,
                verbose=1,
                seed=self.random_seed,
            )
            
        print("Model setup complete")
        
    def setup_callbacks(self):
        """Set up training callbacks for checkpointing, evaluation, etc."""
        print("Setting up callbacks...")
        
        callbacks = []
        
        # Checkpoint callback saves the model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=self.checkpoint_freq,
            save_path=self.model_dir,
            name_prefix=f"sac_pusher_{self.run_id}",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback periodically evaluates the model
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best_model"),
            log_path=self.logger_path,
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        return CallbackList(callbacks)
    
    def train(self):
        """Train the SAC agent."""
        print(f"Starting training for {self.total_timesteps} timesteps...")
        
        # Setup environments if not done already
        if self.env is None:
            self.setup_environments()
            
        # Setup model if not done already
        if self.model is None:
            self.setup_model()
            
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        try:
            # Start training
            start_time = time.time()
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                log_interval=10,
                tb_log_name=self.run_id,
                reset_num_timesteps=True,
            )
            training_time = time.time() - start_time
            
            # Save the final model
            final_model_path = os.path.join(self.model_dir, f"final_model_{self.run_id}")
            self.model.save(final_model_path)
            print(f"Training completed in {training_time:.2f} seconds.")
            print(f"Final model saved to {final_model_path}")
            
            return final_model_path
            
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            # Save the model on interrupt
            interrupted_model_path = os.path.join(self.model_dir, f"interrupted_model_{self.run_id}")
            self.model.save(interrupted_model_path)
            print(f"Interrupted model saved to {interrupted_model_path}")
            return interrupted_model_path
    
    def _load_checkpoint(self, path, **kwargs):
        if self.trust_checkpoint is not True:
            raise ValueError("Loading a checkpoint requires trust_checkpoint=True.")
        return SAC.load(path, **kwargs)

    def evaluate(self, model_path=None, n_eval_episodes=10, deterministic=True):
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to the model to evaluate (if None, uses the current model)
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            mean_reward, std_reward: Mean and standard deviation of episode rewards
        """
        print(f"Evaluating model over {n_eval_episodes} episodes...")
        
        # Setup environments if not done already
        if self.eval_env is None:
            self.setup_environments()
        
        # Load model if path is provided
        if model_path:
            model = self._load_checkpoint(model_path, env=self.eval_env)
        else:
            # Use existing model
            model = self.model
            
        if model is None:
            raise ValueError("No model available for evaluation. Either provide a model_path or train a model first.")
        
        # Run evaluation
        mean_reward, std_reward = evaluate_policy(
            model, 
            self.eval_env, 
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=True if self.render_mode else False,
            return_episode_rewards=False,
        )
        
        print(f"Evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        return mean_reward, std_reward
    
    def run_episode(self, model_path=None, deterministic=True, render=True):
        """
        Run a single episode with the trained model (useful for visualization).
        
        Args:
            model_path: Path to the model to run (if None, uses the current model)
            deterministic: Whether to use deterministic actions
            render: Whether to render the episode
        """
        # Create a dedicated environment for running episodes
        env = RoboticPushingEnv(
            render_mode="human" if render else None,
            max_steps=self.max_steps
        )
        
        # Load model if path is provided
        if model_path:
            model = self._load_checkpoint(model_path)
        else:
            # Use existing model
            model = self.model
            
        if model is None:
            raise ValueError("No model available. Either provide a model_path or train a model first.")
        
        # Run an episode
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        print("Running episode...")
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if render:
                time.sleep(0.01)  # Slow down for visibility
        
        print(f"Episode finished after {step_count} steps with total reward: {total_reward:.2f}")
        print(f"Success: {info.get('is_success', False)}")
        print(f"Termination reason: {info.get('termination_reason', 'unknown')}")
        
        env.close()
        return total_reward, info


def load_config_from_yaml(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert any numerical string values with underscores (e.g., "1_000_000")
    for key, value in config.items():
        if isinstance(value, str) and "_" in value and any(c.isdigit() for c in value):
            try:
                # Replace underscores and try to convert to int
                clean_value = value.replace("_", "")
                config[key] = int(clean_value)
            except ValueError:
                # If conversion fails, keep the original value
                pass
    
    return config


def setup_experiment_dir(task_name, mode="train"):
    """
    Create an experiment directory structure.
    
    Args:
        task_name: Name of the task/experiment
        mode: Either "train" or "eval" to separate training from evaluation runs
        
    Returns:
        str: Path to the created experiment directory
    """
    # Create base output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mode-specific directory (train or eval)
    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    
    # Create shared logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create experiment directory with timestamp and task name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{task_name}"
    experiment_dir = os.path.join(mode_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create models directory inside experiment directory (for training mode)
    if mode == "train":
        models_dir = os.path.join(experiment_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
    
    return experiment_dir


def parse_arguments():
    """Parse command line arguments for the trainer."""
    parser = argparse.ArgumentParser(description="Train a robotic pushing agent with SAC using YAML configuration")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    
    # Task name
    parser.add_argument("--task-name", type=str, default=None, help="Name of the task/experiment")
    
    # Optional override parameters
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=None, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument("--tau", type=float, default=None, help="Target network update rate")
    parser.add_argument("--gradient-steps", type=int, default=None, help="Gradient steps per update")
    parser.add_argument("--train-freq", type=int, default=None, help="Update frequency")
    parser.add_argument("--learning-starts", type=int, default=None, help="Steps before learning starts")
    parser.add_argument("--ent-coef", type=str, default=None, help="Entropy coefficient ('auto' or float)")
    
    # Environment parameters
    parser.add_argument("--n-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--parallel", action="store_true", help="Use parallel environments (SubprocVecEnv)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    
    # Logging and saving
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory for model saves")
    parser.add_argument("--tensorboard", action="store_true", help="Use TensorBoard logging")
    parser.add_argument("--checkpoint-freq", type=int, default=None, help="Checkpoint frequency (in timesteps)")
    parser.add_argument("--eval-freq", type=int, default=None, help="Evaluation frequency (in timesteps)")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--load-model", type=str, default=None, help="Path to model to load")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow loading a checkpoint from a trusted source (can execute code)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on a trained model")
    parser.add_argument("--run-episode", action="store_true", help="Run a single episode with a trained model")
    parser.add_argument("--n-eval-episodes", type=int, default=None, help="Number of episodes for evaluation")
    
    return parser.parse_args()


def merge_config_with_args(config, args):
    """
    Merge YAML configuration with command-line arguments.
    Command-line arguments take precedence over YAML config.
    
    Args:
        config: Configuration dictionary from YAML
        args: Parsed command-line arguments
        
    Returns:
        dict: Merged configuration
    """
    # Create a copy of the config to avoid modifying the original
    merged_config = config.copy()
    merged_config.pop('trust_checkpoint', None)
    
    # Update the config with any non-None command-line arguments
    for key, value in vars(args).items():
        if value is not None and key not in ('config', 'task_name', 'trust_checkpoint'):
            # Handle boolean flags specially
            if key == 'parallel' and value:
                merged_config['parallel'] = True
            elif key == 'render' and value:
                merged_config['render'] = True
            elif key == 'tensorboard' and value:
                merged_config['tensorboard'] = True
            elif key == 'eval_only' and value:
                merged_config['eval_only'] = True
            elif key == 'run_episode' and value:
                merged_config['run_episode'] = True
            elif value is not None:
                # For all other arguments, replace hyphens with underscores in the key
                config_key = key.replace('-', '_')
                merged_config[config_key] = value
    
    return merged_config


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration from YAML if provided
    if args.config:
        try:
            config = load_config_from_yaml(args.config)
            logging.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            exit(1)
    else:
        # If no config file, use empty dict (command-line args will apply)
        config = {}
        logging.warning("No YAML configuration provided, using command-line arguments only")
    
    # Merge command-line arguments with YAML config
    merged_config = merge_config_with_args(config, args)
    if merged_config.get("load_model") and not args.trust_checkpoint:
        raise SystemExit("Loading a checkpoint requires --trust-checkpoint (trusted sources only).")
    
    # Get task name
    task_name = args.task_name or "unnamed_task"
    
    # Determine operation mode (train or eval)
    mode = "eval" if merged_config.get('eval_only') else "train"
    
    # Set up experiment directory with appropriate mode
    experiment_dir = setup_experiment_dir(task_name, mode)
    logging.info(f"Created {mode} experiment directory: {experiment_dir}")
    
    # Initialize trainer with merged configuration
    trainer = RobotPushTrainer(
        config=merged_config,
        trust_checkpoint=args.trust_checkpoint,
        experiment_dir=experiment_dir,
        task_name=task_name
    )
    
    # Mode selection based on configuration
    if merged_config.get('eval_only') and merged_config.get('load_model'):
        # Evaluation mode
        trainer.evaluate(
            model_path=merged_config.get('load_model'),
            n_eval_episodes=merged_config.get('n_eval_episodes', 10)
        )
    elif merged_config.get('run_episode') and merged_config.get('load_model'):
        # Run a single episode
        trainer.run_episode(
            model_path=merged_config.get('load_model'),
            deterministic=True,
            render=True
        )
    else:
        # Training mode
        trainer.train()
        
        # Optionally evaluate the trained model
        trainer.evaluate(
            n_eval_episodes=merged_config.get('n_eval_episodes', 10)
        )