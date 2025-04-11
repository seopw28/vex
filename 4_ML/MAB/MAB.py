#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import seaborn as sns
from tqdm import tqdm

#%%
class BanditAlgorithm:
    """Base class for Multi-Armed Bandit algorithms"""
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.name = "Base Bandit"
    
    def select_arm(self) -> int:
        """Select which arm to pull"""
        raise NotImplementedError
    
    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the algorithm's parameters based on the reward received"""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Incremental update formula
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

#%%
class EpsilonGreedy(BanditAlgorithm):
    """Epsilon-Greedy algorithm for Multi-Armed Bandit problems"""
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.name = f"ε-Greedy (ε={epsilon})"
    
    def select_arm(self) -> int:
        """
        Select an arm using epsilon-greedy strategy:
        - With probability epsilon, select a random arm (exploration)
        - With probability 1-epsilon, select the arm with highest estimated value (exploitation)
        """
        if np.random.random() < self.epsilon:
            # Exploration: choose a random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploitation: choose the best arm
            return np.argmax(self.values)

#%%
class ThompsonSampling(BanditAlgorithm):
    """Thompson Sampling algorithm for Multi-Armed Bandit problems with Bernoulli rewards"""
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        # For each arm, track alpha and beta parameters of the Beta distribution
        self.alpha = np.ones(n_arms)  # Prior successes + 1
        self.beta = np.ones(n_arms)   # Prior failures + 1
        self.name = "Thompson Sampling"
    
    def select_arm(self) -> int:
        """Select an arm by sampling from the posterior distribution of each arm"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, chosen_arm: int, reward: float) -> None:
        """Update the Beta distribution parameters based on the observed reward"""
        super().update(chosen_arm, reward)
        # For Bernoulli bandits, reward is either 0 or 1
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += (1 - reward)

#%%
class UCB(BanditAlgorithm):
    """Upper Confidence Bound algorithm for Multi-Armed Bandit problems"""
    def __init__(self, n_arms: int, c: float = 2.0):
        super().__init__(n_arms)
        self.c = c  # Exploration parameter
        self.t = 0  # Total number of trials
        self.name = f"UCB (c={c})"
    
    def select_arm(self) -> int:
        """
        Select an arm using UCB strategy:
        - For arms that haven't been pulled, prioritize them
        - Otherwise, select the arm with the highest upper confidence bound
        """
        self.t += 1
        
        # If an arm hasn't been tried yet, try it
        if np.any(self.counts == 0):
            return np.argmin(self.counts)
        
        # Calculate UCB for each arm
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

#%%
def simulate_bandit(algorithms: List[BanditAlgorithm], true_probs: List[float], 
                   n_trials: int = 1000, n_experiments: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate multiple bandit algorithms over multiple experiments
    
    Args:
        algorithms: List of bandit algorithms to compare
        true_probs: True probabilities of success for each arm
        n_trials: Number of trials per experiment
        n_experiments: Number of experiments to run
        
    Returns:
        Tuple of DataFrames containing cumulative rewards and optimal arm selection rates
    """
    n_algorithms = len(algorithms)
    n_arms = len(true_probs)
    
    # Store results
    all_rewards = np.zeros((n_algorithms, n_experiments, n_trials))
    optimal_arm = np.argmax(true_probs)
    optimal_selections = np.zeros((n_algorithms, n_experiments, n_trials))
    
    for exp in tqdm(range(n_experiments), desc="Experiments"):
        # Reset algorithms for each experiment
        for i, algorithm in enumerate(algorithms):
            algorithm.__init__(n_arms)
            
        for t in range(n_trials):
            for i, algorithm in enumerate(algorithms):
                # Select arm
                chosen_arm = algorithm.select_arm()
                
                # Generate reward (Bernoulli)
                reward = 1 if np.random.random() < true_probs[chosen_arm] else 0
                
                # Update algorithm
                algorithm.update(chosen_arm, reward)
                
                # Store results
                if t == 0:
                    all_rewards[i, exp, t] = reward
                else:
                    all_rewards[i, exp, t] = all_rewards[i, exp, t-1] + reward
                
                optimal_selections[i, exp, t] = 1 if chosen_arm == optimal_arm else 0
    
    # Calculate average results across experiments
    avg_rewards = np.mean(all_rewards, axis=1)
    avg_optimal = np.mean(optimal_selections, axis=1)
    
    # Create DataFrames for easier plotting
    rewards_df = pd.DataFrame({
        algorithm.name: avg_rewards[i] for i, algorithm in enumerate(algorithms)
    })
    rewards_df['Trial'] = np.arange(1, n_trials + 1)
    
    optimal_df = pd.DataFrame({
        algorithm.name: avg_optimal[i] for i, algorithm in enumerate(algorithms)
    })
    optimal_df['Trial'] = np.arange(1, n_trials + 1)
    
    return rewards_df, optimal_df

#%%
def plot_results(rewards_df: pd.DataFrame, optimal_df: pd.DataFrame, save_path: str = None):
    """
    Plot the results of the bandit simulation
    
    Args:
        rewards_df: DataFrame containing cumulative rewards
        optimal_df: DataFrame containing optimal arm selection rates
        save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative rewards
    plt.subplot(2, 1, 1)
    for col in rewards_df.columns:
        if col != 'Trial':
            plt.plot(rewards_df['Trial'], rewards_df[col], label=col)
    
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot optimal arm selection rate
    plt.subplot(2, 1, 2)
    for col in optimal_df.columns:
        if col != 'Trial':
            # Calculate moving average for smoother curves
            window_size = 50
            smoothed = optimal_df[col].rolling(window=window_size, min_periods=1).mean()
            plt.plot(optimal_df['Trial'], smoothed, label=col)
    
    plt.xlabel('Trials')
    plt.ylabel('Optimal Arm Selection Rate')
    plt.title('Probability of Selecting the Optimal Arm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

#%%
# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define true probabilities for each arm (e.g., CTR for different versions)
    # In this example, we have 4 arms with different success probabilities
    true_probs = [0.05, 0.1, 0.15, 0.2]  # Arm 3 is optimal
    n_arms = len(true_probs)
    
    # Create bandit algorithms
    algorithms = [
        EpsilonGreedy(n_arms, epsilon=0.1),
        EpsilonGreedy(n_arms, epsilon=0.01),
        ThompsonSampling(n_arms),
        UCB(n_arms, c=2.0)
    ]
    
    # Run simulation
    print("Running MAB simulation for A/B/C/D testing...")
    rewards_df, optimal_df = simulate_bandit(
        algorithms=algorithms,
        true_probs=true_probs,
        n_trials=1000,
        n_experiments=50
    )
    
    # Plot and save results
    plot_results(rewards_df, optimal_df, save_path="mab_comparison.png")
    
    # Print final results
    print("\nFinal Cumulative Rewards:")
    final_rewards = rewards_df.iloc[-1].drop('Trial')
    for algorithm, reward in final_rewards.items():
        print(f"{algorithm}: {reward:.2f}")
    
    print("\nFinal Optimal Arm Selection Rates:")
    final_optimal = optimal_df.iloc[-100:].drop('Trial', axis=1).mean()
    for algorithm, rate in final_optimal.items():
        print(f"{algorithm}: {rate*100:.2f}%")

# %%
