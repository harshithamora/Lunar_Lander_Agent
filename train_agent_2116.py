import gymnasium as gym
import numpy as np
import argparse
import os

def policy_action(params, observation):
    # The policy is a linear mapping from the 8-dimensional observation to 4 action scores.
    W = params[:8 * 4].reshape(8, 4)
    b = params[8 * 4:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)


def evaluate_policy(params, episodes=3, render=False):
    total_reward = 0.0

    for _ in range(episodes):
        if render:
            env = gym.make('LunarLander-v3', render_mode='human')
        else:
            env = gym.make('LunarLander-v3')

        observation, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        env.close()
        total_reward += episode_reward

    return total_reward / episodes


def gaussian_crossover(p1, p2):
    alpha = np.random.uniform(0, 1, size=p1.shape)
    child = alpha * p1 + (1 - alpha) * p2
    return child

def gaussian_mutation(child, sigma, mutation_rate=0.2):
    mask = np.random.rand(*child.shape) < mutation_rate
    noise = np.random.normal(0, sigma, size=child.shape)
    child[mask] += noise[mask]
    return child

def tournament_selection(population, fitness, k=3):
    idx = np.random.choice(len(population), k, replace=False)
    return population[idx[np.argmax(fitness[idx])]]


def local_search(child, steps=10, sigma=0.1, T=1.0, alpha=0.9):
    current = child.copy()
    current_score = evaluate_policy(current, episodes=2)

    best = current.copy()
    best_score = current_score

    for _ in range(steps):
        candidate = current + np.random.normal(0, sigma, size=current.shape)
        candidate_score = evaluate_policy(candidate, episodes=2)

        delta = candidate_score - current_score

        # Accept if better OR with probability (SA)
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current = candidate
            current_score = candidate_score

            if current_score > best_score:
                best = current
                best_score = current_score

        T *= alpha

    return best

def genetic_algorithm(population_size=200, num_generations=200, elite_frac=0.2,
                      mutation_rate=0.1, lower_bound=-5, upper_bound=5):
    gene_size = 8 * 4 + 4  # 8 inputs x 4 outputs + 4 biases = 36 parameters
    population = np.random.randn(population_size, gene_size)
    
    num_elites = int(population_size * elite_frac)
    best_reward = -np.inf
    best_params = None

    for generation in range(num_generations):
        fitness = np.array([evaluate_policy(individual, episodes=4) for individual in population])
        elite_indices = fitness.argsort()[::-1][:num_elites]
        elites = population[elite_indices]
        
        if fitness[elite_indices[0]] > best_reward:
            best_reward = fitness[elite_indices[0]]
            best_params = population[elite_indices[0]].copy()
        
        print(f"Generation {generation+1}: Best Average Reward = {best_reward:.2f}")
        
        # Create new population using elitism, tournament selection,
        # arithmetic crossover, Gaussian mutation, and local search.
        new_population = []
        new_population.extend(elites)
        while len(new_population) < population_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            sigma = 0.2 * (1 - generation / num_generations)

            child = gaussian_crossover(p1, p2)
            child = gaussian_mutation(child, sigma, mutation_rate)

            child = local_search(child, steps=3, sigma=0.1, T=1.0, alpha=0.9)
            child = np.clip(child, lower_bound, upper_bound)

            new_population.append(child)
        
        population = np.array(new_population)
    
    return best_params

def train_and_save(filename, population_size=200, num_generations=200, elite_frac=0.2,
                   mutation_rate=0.1, lower_bound=-5, upper_bound=5):

    print("Training new policy...")
    new_params = genetic_algorithm(population_size, num_generations, elite_frac,
                                   mutation_rate, lower_bound, upper_bound)

    print("Evaluating new policy...")
    new_score = evaluate_policy(new_params, episodes=10)

    # Check if old policy exists
    if os.path.exists(filename):
        print("Existing policy found. Evaluating...")
        old_params = np.load(filename)
        old_score = evaluate_policy(old_params, episodes=10)

        print(f"Old Score: {old_score:.2f}")
        print(f"New Score: {new_score:.2f}")

        if new_score > old_score:
            np.save(filename, new_params)
            print("New policy is better. Overwritten.")
            return new_params
        else:
            print("New policy is worse. Keeping old policy.")
            return old_params
    else:
        # No existing file → just save
        np.save(filename, new_params)
        print("No existing policy. Saved new policy.")
        return new_params

def load_policy(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    best_params = np.load(filename)
    print(f"Loaded best policy from {filename}")
    return best_params

def play_policy(best_params, episodes=5):
    test_reward = evaluate_policy(best_params, episodes=episodes, render=False)
    print(f"Average reward of the best policy over {episodes} episodes: {test_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play the best Lunar Lander policy using a genetic algorithm with tournament selection, arithmetic crossover, Gaussian mutation, and local search.")
    parser.add_argument("--train", action="store_true", help="Train the policy using GA and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy_2116.npy", help="Filename to save/load the best policy.")
    args = parser.parse_args()

    if args.train:
        # Train and save the best policy
        best_params = train_and_save(
            args.filename,
            population_size=200,
            num_generations=200,
            elite_frac=0.2,
            mutation_rate=0.1,
            lower_bound=-5,
            upper_bound=5
        )
    elif args.play:
        # Load and play with the best policy
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=100)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")
