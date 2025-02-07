import torch
import torch.nn as nn
import random

# Define the environment for Maxwell's Demon
class Molecule:
    def __init__(self, velocity):
        self.velocity = velocity  # Molecule's velocity (could represent speed)

    def __repr__(self):
        return f"Molecule({self.velocity:.2f})"


class MaxwellsDemonEnv:
    def __init__(self, threshold):
        self.threshold = threshold  # Speed threshold for fast/slow sorting
        self.reset()

    def reset(self):
        self.molecules = [Molecule(random.uniform(0, 3)) for _ in range(10)]  # Generate random molecules
        self.fast_container = []  # List of fast molecules
        self.slow_container = []  # List of slow molecules
        return [m.velocity for m in self.molecules]  # Return the initial state (velocities)

    def step(self, action):
        # Perform the sorting action for each molecule
        reward = 0
        for molecule in self.molecules:
            if action == 1 and molecule.velocity > self.threshold:  # Fast molecule
                self.fast_container.append(molecule)
                reward += 1  # Reward for sorting correctly into fast
            elif action == 0 and molecule.velocity <= self.threshold:  # Slow molecule
                self.slow_container.append(molecule)
                reward += 1  # Reward for sorting correctly into slow
        # Return the new state (molecules remaining) and reward
        return [m.velocity for m in self.molecules], reward


# Define a deep learning model for Agent A (Maxwell's Demon)
class MaxwellDemonModel(nn.Module):
    def __init__(self):
        super(MaxwellDemonModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # Input layer for 10 molecules' velocities
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output action: 0 or 1 (slow or fast)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply relu activation
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Sigmoid for 0 or 1 action

class MaxwellDemonAgent:
    def __init__(self, threshold, model):
        self.env = MaxwellsDemonEnv(threshold)  # Environment setup
        self.model = model  # Deep learning model for decision making
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        action_prob = self.model(state_tensor)  # Get probability of sorting action (0 or 1)
        return 1 if action_prob.item() > 0.5 else 0  # Action is either 1 (fast) or 0 (slow)

    def train(self, cycles=1000):
        for cycle in range(cycles):
            state = self.env.reset()  # Reset environment
            action = self.select_action(state)  # Select action based on current state
            next_state, reward = self.env.step(action)  # Perform action and get feedback from environment
            
            # Training step: optimize model
            target = torch.tensor([reward], dtype=torch.float32)
            action_prob = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            loss = self.loss_fn(action_prob, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if cycle % 100 == 0:
                print(f"Cycle {cycle}: Loss: {loss.item()}, Reward: {reward}")


# Example usage
threshold_speed = 1.5  # Speed threshold for sorting fast/slow molecules
model = MaxwellDemonModel()  # Initialize the model for the agent
agent = MaxwellDemonAgent(threshold=threshold_speed, model=model)

# Train the agent over multiple cycles
agent.train(cycles=1000)
