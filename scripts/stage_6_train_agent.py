"""
Train a robotic pushing agent using Stable Baselines 3 SAC algorithm.
This script handles training, evaluation, logging, and visualization.
"""

import os
import time
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union

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
from stage_5_agent_environment import RoboticPushingEnv

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
        total_timesteps: int = 1_000_000,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        gradient_steps: int = 1,
        train_freq: int = 1,
        learning_starts: int = 10000,
        use_sde: bool = True,
        sde_sample_freq: int = -1,
        tensorboard_log: Optional[str] = None,
        ent_coef: Union[str, float] = "auto",
        checkpoint_freq: int = 10000,
        eval_freq: int = 10000,
        random_seed: Optional[int] = None,
        n_envs: int = 1,
        parallel_envs: bool = False,
        max_steps: int = 1000,
        load_model: Optional[str] = None,
        log_dir: str = "logs",
        model_dir: str = "models",
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the trainer with configuration parameters.
        
        Args:
            total_timesteps: Total timesteps for training
            learning_rate: Learning rate for the optimizer
            buffer_size: Size of the replay buffer
            batch_size: Batch size for each gradient update
            gamma: Discount factor
            tau: Target network update rate
            gradient_steps: How many gradient steps to take per update
            train_freq: Update the model every train_freq steps
            learning_starts: How many steps to collect before learning starts
            use_sde: Whether to use State Dependent Exploration
            sde_sample_freq: Sample frequency for SDE
            tensorboard_log: TensorBoard log directory
            ent_coef: Entropy coefficient or 'auto'
            checkpoint_freq: Frequency of checkpoints
            eval_freq: Frequency of evaluations
            random_seed: Random seed for reproducibility
            n_envs: Number of parallel environments
            parallel_envs: Whether to use SubprocVecEnv (parallel) or DummyVecEnv
            max_steps: Maximum number of steps per episode
            load_model: Path to a pre-trained model to load
            log_dir: Directory for logs
            model_dir: Directory for model saves
            render_mode: Rendering mode for the environment
        """
        # Store configuration parameters
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.train_freq = train_freq
        self.learning_starts = learning_starts
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.tensorboard_log = tensorboard_log
        self.ent_coef = ent_coef
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.random_seed = random_seed
        self.n_envs = n_envs
        self.parallel_envs = parallel_envs
        self.max_steps = max_steps
        self.load_model = load_model
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.render_mode = render_mode
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            # TODO: add torch also?
            
        # Initialize environment and model
        self.env = None
        self.eval_env = None
        self.model = None
        
        # Setup additional variables
        self.run_id = f"sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure logger
        self.logger_path = os.path.join(self.log_dir, self.run_id)
        os.makedirs(self.logger_path, exist_ok=True)
        
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
        if self.load_model and os.path.exists(self.load_model):
            print(f"Loading pre-trained model from {self.load_model}")
            self.model = SAC.load(
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
        if model_path and os.path.exists(model_path):
            model = SAC.load(model_path, env=self.eval_env)
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
        if model_path and os.path.exists(model_path):
            model = SAC.load(model_path)
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
        

def parse_arguments():
    """Parse command line arguments for the trainer."""
    parser = argparse.ArgumentParser(description="Train a robotic pushing agent with SAC")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient steps per update")
    parser.add_argument("--train-freq", type=int, default=1, help="Update frequency")
    parser.add_argument("--learning-starts", type=int, default=10000, help="Steps before learning starts")
    parser.add_argument("--ent-coef", type=str, default="auto", help="Entropy coefficient ('auto' or float)")
    
    # Environment parameters
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--parallel", action="store_true", help="Use parallel environments (SubprocVecEnv)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    
    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for model saves")
    parser.add_argument("--tensorboard", action="store_true", help="Use TensorBoard logging")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Checkpoint frequency (in timesteps)")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency (in timesteps)")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--load-model", type=str, default=None, help="Path to model to load")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on a trained model")
    parser.add_argument("--run-episode", action="store_true", help="Run a single episode with a trained model")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="Number of episodes for evaluation")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure TensorBoard logging
    tensorboard_log = None
    if args.tensorboard and tensorboard_available:
        tensorboard_log = os.path.join(args.log_dir, "tensorboard")
        os.makedirs(tensorboard_log, exist_ok=True)
    
    # Initialize trainer
    trainer = RobotPushTrainer(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        gradient_steps=args.gradient_steps,
        train_freq=args.train_freq,
        learning_starts=args.learning_starts,
        tensorboard_log=tensorboard_log,
        ent_coef=args.ent_coef,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        random_seed=args.seed,
        n_envs=args.n_envs,
        parallel_envs=args.parallel,
        max_steps=args.max_steps,
        load_model=args.load_model,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        render_mode="human" if args.render else None,
    )
    
    # Mode selection based on command line arguments
    if args.eval_only and args.load_model:
        # Evaluation mode
        trainer.evaluate(
            model_path=args.load_model,
            n_eval_episodes=args.n_eval_episodes
        )
    elif args.run_episode and args.load_model:
        # Run a single episode
        trainer.run_episode(
            model_path=args.load_model,
            deterministic=True,
            render=True
        )
    else:
        # Training mode
        trained_model_path = trainer.train()
        
        # Optionally evaluate the trained model
        trainer.evaluate(
            model_path=trained_model_path,
            n_eval_episodes=args.n_eval_episodes
        )