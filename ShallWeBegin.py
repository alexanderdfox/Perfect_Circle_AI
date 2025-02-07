import torch
import torch.nn as nn
import random

# Define a simple deep learning model for the agents
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # Simple layer for input state
        self.fc2 = nn.Linear(10, 1)  # Output of decision value (state adjustment)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply relu activation
        return self.fc2(x)  # Output the decision value

class DeepSeekAgent:
    def __init__(self, name, initial_state):
        self.name = name
        self.state = torch.tensor([initial_state], dtype=torch.float32)  # Initial state as tensor
        self.model = SimpleModel()  # Deep learning model for decision making
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)  # Optimizer for training
        self.loss_fn = nn.MSELoss()  # Loss function to compare state adjustments
        self.memory = []  # To store state history

    def make_decision(self):
        # Model makes a decision based on current state
        decision = self.model(self.state)  # Model predicts state adjustment
        return decision

    def feedback(self, decision, target):
        # Feedback mechanism: adjust state based on received decision and target
        self.optimizer.zero_grad()  # Zero gradients
        loss = self.loss_fn(decision, target)  # Calculate loss (difference)
        loss.backward()  # Backpropagate error
        self.optimizer.step()  # Update weights based on the gradients

        # Update state after feedback
        self.state += decision.detach()  # Adjust state by decision made
        self.memory.append(self.state.item())  # Save state history

    def interact(self, other_agent):
        # Interaction between two agents
        print(f"\n{self.name} interacting with {other_agent.name}:")
        decision = self.make_decision()  # Make decision based on current state
        feedback_target = other_agent.state  # Use the other agent's state as feedback target
        other_agent.feedback(decision, feedback_target)  # Provide feedback to the other agent

    def run_cycle(self, other_agent, cycles=5):
        # Run the interaction for several cycles
        for cycle in range(cycles):
            print(f"\nCycle {cycle + 1}:")
            self.interact(other_agent)  # Agents interact with feedback loop

# Initialize two DeepSeek agents
agent_1 = DeepSeekAgent(name="Agent A", initial_state=10.0)
agent_2 = DeepSeekAgent(name="Agent B", initial_state=5.0)

# Run the circular feedback loop for interaction between the two agents
agent_1.run_cycle(agent_2, cycles=5)

# Print memory of final states for each agent after interaction
print(f"\nAgent A memory of states: {agent_1.memory}")
print(f"Agent B memory of states: {agent_2.memory}")
