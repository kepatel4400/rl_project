"""
Advanced testing script for robotic pushing environment.
This script provides detailed analysis, visualization, and statistics.
"""

import os
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import datetime

# Import the environment
from stage_5_agent_environment import RoboticPushingEnv

def test_environment(
    num_episodes=10,
    max_steps=1000,
    render_mode="human",
    log_level="INFO",
    sleep_time=0.01,
    save_stats=True,
    plot_results=True,
    save_dir="test_results"
):
    """
    Comprehensive test of the robotic pushing environment.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        sleep_time: Time to sleep between steps for visualization
        save_stats: Whether to save statistics to a file
        plot_results: Whether to plot results
        save_dir: Directory to save results
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Create environment
    logging.info(f"Creating environment with render_mode={render_mode}")
    env = RoboticPushingEnv(render_mode=render_mode, max_steps=max_steps)
    
    # Statistics to track
    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "successes": 0,
        "termination_reasons": defaultdict(int),
        "reward_components": defaultdict(list),  # For tracking different reward components if available
        "observations": {
            "vision": [],      # Object bounding box
            "target": [],      # Target position
            "proprioception": []  # End-effector state
        },
        "object_target_distances": [],  # Distance between object and target
        "actions": [],         # Actions taken
        "timestamps": []       # Timing information
    }
    
    try:
        # Run episodes
        for episode in range(num_episodes):
            episode_start_time = time.time()
            logging.info(f"Starting episode {episode+1}/{num_episodes}")
            
            # Reset environment
            obs, info = env.reset()
            
            # Track data for this episode
            episode_reward = 0
            episode_rewards = []
            episode_actions = []
            episode_observations = {
                "vision": [],
                "target": [],
                "proprioception": []
            }
            episode_obj_target_distances = []
            
            # Run episode
            for step in range(max_steps):
                # Sample random action
                action = env.action_space.sample()
                episode_actions.append(action)
                
                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)
                
                # Update statistics
                episode_reward += reward
                episode_rewards.append(reward)
                
                # Track observations
                episode_observations["vision"].append(obs["vision"])
                episode_observations["target"].append(obs["target"])
                episode_observations["proprioception"].append(obs["proprioception"])
                
                # Calculate approximate distance between object and target
                # Using the bounding box center as an approximation of object position
                obj_bbox = obs["vision"]  # [x, y, width, height] normalized
                obj_center_x, obj_center_y = obj_bbox[0] + obj_bbox[2]/2, obj_bbox[1] + obj_bbox[3]/2
                target_x, target_y = obs["target"]  # Target position
                
                # Convert from normalized coordinates to same scale
                # This is approximate since we don't know the exact scaling
                # but it gives us a relative measure
                obj_target_dist = np.sqrt((obj_center_x - (target_x+1)/2)**2 + 
                                         (obj_center_y - (target_y+1)/2)**2)
                episode_obj_target_distances.append(obj_target_dist)
                
                # Log step information
                if step % 100 == 0 or done or truncated:
                    ee_pos = obs["proprioception"][:3]  # End-effector position
                    logging.info(f"Episode {episode+1}, Step {step}, Reward: {reward:.4f}")
                    logging.info(f"Object bbox: {obs['vision']}, Target: {obs['target']}, EE pos: {ee_pos}")
                    logging.info(f"Approx obj-target distance: {obj_target_dist:.4f}")
                
                # Add small delay for visualization if rendering
                if render_mode and sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Check for episode end
                if done or truncated:
                    break
            
            # Calculate episode duration
            episode_duration = time.time() - episode_start_time
            
            # Update episode statistics
            stats["episode_rewards"].append(episode_reward)
            stats["episode_lengths"].append(step + 1)
            stats["timestamps"].append(episode_duration)
            
            # Store observations (just the last state for memory efficiency)
            stats["observations"]["vision"].append(episode_observations["vision"][-1])
            stats["observations"]["target"].append(episode_observations["target"][-1])
            stats["observations"]["proprioception"].append(episode_observations["proprioception"][-1])
            
            # Store average object-target distance
            stats["object_target_distances"].append(np.mean(episode_obj_target_distances))
            
            # Store actions (just the last few for memory efficiency)
            stats["actions"].append(episode_actions[-5:] if len(episode_actions) > 5 else episode_actions)
            
            if info.get("is_success", False):
                stats["successes"] += 1
            
            term_reason = info.get("termination_reason", "unknown")
            stats["termination_reasons"][term_reason] += 1
            
            # Split reward components if available in info
            if "reward_info" in info:
                for component, value in info["reward_info"].items():
                    stats["reward_components"][component].append(value)
            
            logging.info(f"Episode {episode+1} finished after {step+1} steps with reward: {episode_reward:.4f}")
            logging.info(f"Success: {info.get('is_success', False)}")
            logging.info(f"Termination reason: {term_reason}")
            logging.info(f"Episode duration: {episode_duration:.2f} seconds")
            
            # Print separator for clarity
            logging.info("-" * 50)
    
    except KeyboardInterrupt:
        logging.info("Testing interrupted by user")
    
    except Exception as e:
        logging.error(f"Error during testing: {e}", exc_info=True)
    
    finally:
        # Close environment
        env.close()
        
        # Print overall statistics
        if len(stats["episode_rewards"]) > 0:
            print_statistics(stats, num_episodes)
            
            # Save statistics if requested
            if save_stats:
                save_statistics(stats, save_dir)
            
            # Plot results if requested
            if plot_results:
                plot_statistics(stats, save_dir)
        else:
            logging.warning("No episodes completed successfully. Cannot generate statistics.")
    
    return stats

def print_statistics(stats, num_episodes):
    """Print overall statistics."""
    print("\n" + "=" * 50)
    print("Environment Test Results:")
    print("=" * 50)
    
    print(f"Completed {len(stats['episode_rewards'])}/{num_episodes} episodes")
    print(f"Success rate: {stats['successes'] / len(stats['episode_rewards']) * 100:.2f}%")
    print(f"Average episode reward: {np.mean(stats['episode_rewards']):.4f} ± {np.std(stats['episode_rewards']):.4f}")
    print(f"Average episode length: {np.mean(stats['episode_lengths']):.2f} ± {np.std(stats['episode_lengths']):.2f}")
    print(f"Average episode duration: {np.mean(stats['timestamps']):.2f} seconds")
    print(f"Average object-target distance: {np.mean(stats['object_target_distances']):.4f}")
    
    print("\nTermination reasons:")
    for reason, count in stats["termination_reasons"].items():
        print(f"  {reason}: {count} episodes ({count / len(stats['episode_rewards']) * 100:.2f}%)")
    
    if stats["reward_components"]:
        print("\nReward components (averaged across episodes):")
        for component, values in stats["reward_components"].items():
            print(f"  {component}: {np.mean(values):.4f} ± {np.std(values):.4f}")

def save_statistics(stats, save_dir):
    """Save test statistics to files."""
    # Create directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Save summary stats
    with open(f"{save_dir}/env_test_summary_{timestamp}.txt", "w") as f:
        f.write("Environment Test Results:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Episodes: {len(stats['episode_rewards'])}\n")
        f.write(f"Success rate: {stats['successes'] / len(stats['episode_rewards']) * 100:.2f}%\n")
        f.write(f"Average episode reward: {np.mean(stats['episode_rewards']):.4f} ± {np.std(stats['episode_rewards']):.4f}\n")
        f.write(f"Average episode length: {np.mean(stats['episode_lengths']):.2f} ± {np.std(stats['episode_lengths']):.2f}\n")
        f.write(f"Average episode duration: {np.mean(stats['timestamps']):.2f} seconds\n")
        f.write(f"Average object-target distance: {np.mean(stats['object_target_distances']):.4f}\n")
        
        f.write("\nTermination reasons:\n")
        for reason, count in stats["termination_reasons"].items():
            f.write(f"  {reason}: {count} episodes ({count / len(stats['episode_rewards']) * 100:.2f}%)\n")
        
        if stats["reward_components"]:
            f.write("\nReward components (averaged across episodes):\n")
            for component, values in stats["reward_components"].items():
                f.write(f"  {component}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
    
    # Save detailed stats as numpy arrays
    np.savez(
        f"{save_dir}/env_test_data_{timestamp}.npz",
        episode_rewards=np.array(stats["episode_rewards"]),
        episode_lengths=np.array(stats["episode_lengths"]),
        object_target_distances=np.array(stats["object_target_distances"]),
        timestamps=np.array(stats["timestamps"]),
        termination_reasons=list(stats["termination_reasons"].items())
    )
    
    print(f"Statistics saved to {save_dir}/env_test_summary_{timestamp}.txt")

def plot_statistics(stats, save_dir):
    """Create and save visualizations of test results."""
    # Create directories
    plot_dir = f"{save_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Figure 1: Episode metrics
    plt.figure(figsize=(12, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(stats["episode_rewards"], 'b-o', alpha=0.7)
    plt.axhline(y=np.mean(stats["episode_rewards"]), color='r', linestyle='--', alpha=0.7)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    
    # Plot episode lengths
    plt.subplot(2, 2, 2)
    plt.plot(stats["episode_lengths"], 'g-o', alpha=0.7)
    plt.axhline(y=np.mean(stats["episode_lengths"]), color='r', linestyle='--', alpha=0.7)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, alpha=0.3)
    
    # Plot object-target distances
    plt.subplot(2, 2, 3)
    plt.plot(stats["object_target_distances"], 'm-o', alpha=0.7)
    plt.axhline(y=np.mean(stats["object_target_distances"]), color='r', linestyle='--', alpha=0.7)
    plt.title("Avg Object-Target Distance")
    plt.xlabel("Episode")
    plt.ylabel("Distance (normalized)")
    plt.grid(True, alpha=0.3)
    
    # Plot episode durations
    plt.subplot(2, 2, 4)
    plt.plot(stats["timestamps"], 'c-o', alpha=0.7)
    plt.axhline(y=np.mean(stats["timestamps"]), color='r', linestyle='--', alpha=0.7)
    plt.title("Episode Duration")
    plt.xlabel("Episode")
    plt.ylabel("Seconds")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/episode_metrics_{timestamp}.png")
    
    # Figure 2: Success and termination
    plt.figure(figsize=(12, 6))
    
    # Plot success vs failure
    plt.subplot(1, 2, 1)
    success_data = [stats["successes"], len(stats["episode_rewards"]) - stats["successes"]]
    plt.pie(success_data, labels=['Success', 'Failure'], autopct='%1.1f%%', colors=['green', 'red'], explode=[0.1, 0])
    plt.title(f"Success Rate: {stats['successes'] / len(stats['episode_rewards']) * 100:.1f}%")
    
    # Plot termination reasons
    plt.subplot(1, 2, 2)
    reasons = list(stats["termination_reasons"].keys())
    counts = list(stats["termination_reasons"].values())
    
    # Sort by count
    sorted_idx = np.argsort(counts)
    reasons = [reasons[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]
    
    y_pos = np.arange(len(reasons))
    plt.barh(y_pos, counts, align='center')
    plt.yticks(y_pos, reasons)
    plt.xlabel('Count')
    plt.title('Termination Reasons')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/success_termination_{timestamp}.png")
    
    # Figure 3: Reward components (if available)
    if stats["reward_components"]:
        plt.figure(figsize=(12, 6))
        components = list(stats["reward_components"].keys())
        avg_values = [np.mean(stats["reward_components"][c]) for c in components]
        
        # Sort by absolute magnitude
        sorted_idx = np.argsort(np.abs(avg_values))
        components = [components[i] for i in sorted_idx]
        avg_values = [avg_values[i] for i in sorted_idx]
        
        colors = ['green' if v >= 0 else 'red' for v in avg_values]
        plt.barh(components, avg_values, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        plt.title('Average Reward Components')
        plt.xlabel('Value')
        plt.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/reward_components_{timestamp}.png")
    
    # Additional plots if needed...
    
    print(f"Plots saved to {plot_dir}/")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced testing of robotic pushing environment")
    
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--render", choices=["human", "rgb_array", "none"], default="human", 
                        help="Rendering mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Logging level")
    parser.add_argument("--sleep", type=float, default=0.01, help="Sleep time between steps (for visualization)")
    parser.add_argument("--no-save", action="store_true", help="Don't save statistics")
    parser.add_argument("--no-plot", action="store_true", help="Don't plot results")
    parser.add_argument("--save-dir", type=str, default="test_results", help="Directory to save results")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set render mode
    render_mode = None if args.render == "none" else args.render
    
    # Run test
    test_environment(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode=render_mode,
        log_level=args.log_level,
        sleep_time=args.sleep,
        save_stats=not args.no_save,
        plot_results=not args.no_plot,
        save_dir=args.save_dir
    )