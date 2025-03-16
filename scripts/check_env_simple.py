"""
Simple test for the robotic pushing environment.
"""

import numpy as np
import time
import pybullet as p  # Import PyBullet directly

# Import the environment
from stage_5_agent_environment import RoboticPushingEnv

def test_environment(episodes=3, steps_per_episode=200, render="human", sleep_time=0.02):
    """Run a simple test of the environment."""
    print(f"Testing environment with {episodes} episodes of {steps_per_episode} steps each...")
    
    # Create environment
    env = RoboticPushingEnv(render_mode=render, max_steps=steps_per_episode)
    
    # Dictionary to track episode statistics
    stats = {
        "total_rewards": [],
        "episode_lengths": [],
        "successes": 0
    }
    
    try:
        for episode in range(episodes):
            print(f"\nEpisode {episode+1}/{episodes}")
            print("-" * 30)
            
            # Reset environment
            obs, info = env.reset()
            print(f"Initial observation keys: {obs.keys()}")
            
            # Run episode
            total_reward = 0
            for step in range(steps_per_episode):
                # Take random action
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                # Print info occasionally
                if step % 50 == 0 or done or truncated:
                    print(f"Step {step}: Reward={reward:.4f}, Total={total_reward:.4f}")
                    print(f"Object bbox (normalized): {obs['vision']}")
                    print(f"Target position: {obs['target']}")
                    print(f"End-effector state: {obs['proprioception'][:3]}") # Just position part
                
                # Wait a bit for visualization
                if render and sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Check for episode end
                if done or truncated:
                    break
            
            # Record episode statistics
            stats["total_rewards"].append(total_reward)
            stats["episode_lengths"].append(step + 1)
            if info.get("is_success", False):
                stats["successes"] += 1
            
            print(f"Episode finished after {step+1} steps")
            print(f"Total reward: {total_reward:.4f}")
            print(f"Success: {info.get('is_success', False)}")
            print(f"Termination reason: {info.get('termination_reason', 'unknown')}")
        
        # Print overall statistics
        if stats["total_rewards"]:
            success_rate = stats["successes"] / len(stats["total_rewards"]) * 100
            print("\nOverall Statistics:")
            print(f"Success rate: {success_rate:.2f}%")
            print(f"Average episode reward: {np.mean(stats['total_rewards']):.4f}")
            print(f"Average episode length: {np.mean(stats['episode_lengths']):.2f}")
    
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        # Close environment
        print("\nClosing environment...")
        env.close()

if __name__ == "__main__":
    test_environment()