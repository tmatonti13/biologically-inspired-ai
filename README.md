# Biologically Inspired Artificial Intelligence

A collection of Python projects exploring artificial intelligence, evolutionary computation, reinforcement learning, neural networks, and heuristic search algorithms.

Developed as part of my Biologically Inspired Artificial Intelligence coursework at Sarah Lawrence College, these projects focus on implementing AI algorithms, experimenting with their parameters, and evaluating their performance across different computational problems.

## Projects

### 1. Genetic Algorithms — Evolutionary Optimization
**Directory:** `Hw1/`

Implemented and experimented with genetic algorithms inspired by biological evolution.

**Key concepts:**
- Population initialization and fitness evaluation
- Selection, crossover, and mutation
- Evolution across successive generations
- Comparing performance under different mutation and crossover settings

**Files:** `GA1.py`, `GAinspector.py`

**Experimental analysis:** Compared average and best fitness over 50 generations, evaluated mutation-rate effects, and examined crossover versus no-crossover performance.

![Average vs. Best Fitness](Hw1/Avg%20vs%20Best%20%2850%20Generations%29.png)

### 2. Robby the Robot — Genetic Algorithms
**Directory:** `Hw2/`

Applied genetic algorithms to a simulated autonomous robot tasked with collecting cans from an environment.

**Key concepts:**
- Evolutionary optimization of robot behavior
- Fitness-based evaluation
- Strategy improvement over generations
- Performance comparison across multiple experimental runs

**Files:** `HW2Rob.py`, `robby/`

### 3. Robby the Robot — Q-Learning
**Directory:** `Hw3/`

Explored reinforcement learning by training a simulated robot to improve its decision-making through rewards and repeated interaction with its environment.

**Key concepts:**
- Q-learning and state-action values
- Reward-based learning
- Exploration versus exploitation
- Learning-rate and exploration-parameter experiments
- Evaluation of training and testing performance

**Files:** `Hw3.py`, `hwtest.py`

**Experimental analysis:** Investigated how changing the learning rate (alpha) and exploration rate (epsilon) affected training rewards.

![Training Rewards](Hw3/Training%20Reward.png)

### 4. Neural Networks — Supervised Learning
**Directory:** `Hw5/assign5/`

Implemented and explored neural network learning, including the use of hidden layers to model nonlinear relationships.

**Key concepts:**
- Feedforward neural networks
- Activation functions
- Error calculation and backpropagation
- Weight updates and iterative training
- Learning nonlinear relationships such as XOR

**Files:** `assign5.py`, `NNinspector.py`

### 5. Perceptron Learning
**Directory:** `Perceptron Lab/`

Explored supervised classification using perceptron-based learning algorithms.

**Key concepts:**
- Linear classification
- Weighted inputs and activation
- Iterative weight adjustment
- Training and evaluating classification models

**Files:** `perceptron_lab.py`, `cancerPatterns.txt.txt`, `cancerTargets.txt.txt`

### 6. Heuristic Search — Sudoku and Sliding-Block Puzzles
**Directory:** `Conference Project/`

Explored classical AI search techniques through implementations of puzzle-solving algorithms.

**Key concepts:**
- Breadth-first search (BFS)
- A* search
- Heuristic evaluation
- State-space exploration
- Comparing search efficiency and solution paths

**Files:** `Sudoku_v1.py`, `Sliding_Block.py`, `Sliding_Block_Heuristic.py`

## Technologies and Skills

**Programming:** Python

**Artificial Intelligence:** Genetic Algorithms, Reinforcement Learning, Q-Learning, Perceptrons, Neural Networks, Heuristic Search

**Technical Concepts:** Optimization, Fitness Functions, Backpropagation, State-Space Search, Hyperparameter Experimentation, Performance Evaluation

## Project Structure

```text
biologically-inspired-ai/
├── Conference Project/
├── Hw1/
├── Hw2/
├── Hw3/
├── Hw5/
├── Perceptron Lab/
├── perceptron_lab.py
├── .gitignore
└── README.md
```

## About

These projects were developed as part of my undergraduate Computer Science studies at Sarah Lawrence College. They reflect my interest in artificial intelligence, algorithm design, and applying computational techniques to complex problems.

The repository includes algorithm implementations, supporting simulation files, and experimental results documenting the behavior and performance of different approaches.
