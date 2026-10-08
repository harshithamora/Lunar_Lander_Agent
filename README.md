# Lunar Lander Agent using Genetic Algorithm

An AI agent for the **Gymnasium LunarLander-v3** environment using a **Genetic Algorithm** to optimize a linear policy.

## Objective

The goal is to control the lunar lander and achieve a high average reward by optimizing the policy parameters through evolutionary computation.

## Approach

The project uses:

* Linear policy with **36 parameters**
* Tournament selection
* Elitism
* Arithmetic/blend crossover
* Gaussian mutation
* Simulated-annealing-style local search
* Parameter clipping to `[-5, 5]`

### Policy

The environment provides an **8-dimensional observation** and has **4 possible actions**.

```text
8 observations → Linear Policy → 4 action scores → Best action
```

The policy contains:

```text
8 × 4 weights + 4 biases = 36 parameters
```

## Project Structure

```text
Lunar_Lander_Agent/
│
├── policy_2116.py
├── best_policy_2116.npy
├── train_agent_2116.py
├── evaluate_agent.py
├── evaluate_2116.bat
├── play_lunar_lander.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Python **3.13** is recommended.

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train

```powershell
.\venv\Scripts\python.exe train_agent_2116.py --train --filename best_policy_2116.npy
```

The training process evolves candidate policies and saves the best policy to:

```text
best_policy_2116.npy
```

## Evaluate

```powershell
.\venv\Scripts\python.exe evaluate_agent.py --filename best_policy_2116.npy --policy_module policy_2116
```

The evaluator tests the policy over **100 episodes**.

### Current Baseline

```text
Average Reward: 279.75
```

The final training result will be updated after training completes.

## Manual Play

```powershell
.\venv\Scripts\python.exe play_lunar_lander.py
```

Controls:

* `W` → Main engine
* `A` → Left engine
* `D` → Right engine
* `S` → No action
* `Q` → Quit

## Technologies

* Python
* NumPy
* Gymnasium
* Box2D
* Pygame
* Genetic Algorithms
