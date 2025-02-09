import random

class MaxwellsDemonGPT:
	def __init__(self, gpt_model, threshold=0.7):
		"""
		Initialize the custom GPT model with Maxwell's Demon logic.
		- gpt_model: The instance of your GPT model (e.g., OpenAI API, GPT-2, etc.)
		- threshold: The cutoff for distinguishing hot (relevant) data from cold (irrelevant) data.
		"""
		self.gpt_model = gpt_model  # The external GPT model that will generate responses
		self.threshold = threshold  # The relevance threshold
		self.dialogue_history = []
		self.processed_responses = []
		self.filtered_inputs = []

	def evaluate_input(self, user_input):
		"""
		Evaluate the input to decide if it's hot (relevant) or cold (irrelevant).
		- Relevance is simulated by a random score between 0 and 1.
		"""
		relevance = random.random()  # Random relevance score between 0 and 1
		
		# If relevance exceeds threshold, it's "hot" (relevant) data
		if relevance >= self.threshold:
			self.generate_response(user_input, relevance)
		else:
			self.filter_input(user_input, relevance)

	def generate_response(self, user_input, relevance):
		"""
		Generate a response using the GPT model if the input is hot (relevant).
		"""
		# Here, you would call your GPT model to generate a response.
		# Example with a mock GPT model call (this will depend on the GPT model you use).
		response = self.gpt_model.generate_response(user_input)
		processed_response = f"Processing input: '{user_input}'. (Relevance: {relevance:.2f})\nResponse: {response}"
		self.processed_responses.append((user_input, processed_response))
		print(processed_response)

	def filter_input(self, user_input, relevance):
		"""
		Filter out cold (irrelevant) input.
		"""
		self.filtered_inputs.append((user_input, relevance))
		print(f"Filtered out input: '{user_input}'. (Relevance: {relevance:.2f})")

	def process_conversation(self, user_inputs):
		"""
		Process a list of user inputs in a conversation, evaluating each one.
		"""
		for user_input in user_inputs:
			self.evaluate_input(user_input)

	def get_processed_responses(self):
		"""
		Get all processed responses (relevant data).
		"""
		return self.processed_responses

	def get_filtered_inputs(self):
		"""
		Get all filtered inputs (irrelevant data).
		"""
		return self.filtered_inputs


# Simulating a mock GPT model (replace this with a real GPT model such as GPT-3 or GPT-2)
class MockGPTModel:
	def generate_response(self, user_input):
		"""
		Simulates the generation of a response based on the input.
		This is a placeholder for your actual GPT model API or local model.
		"""
		return f"Mock response to: {user_input}"


# Example usage:

# Initialize the Mock GPT model (replace with actual model in practice)
mock_gpt_model = MockGPTModel()

# Create the Maxwell's Demon GPT model instance with threshold = 0.7
maxwell_demon_gpt = MaxwellsDemonGPT(gpt_model=mock_gpt_model, threshold=0.7)

# Simulating user inputs in a conversation
user_inputs = [
	"Tell me about the weather today.",
	"What is the meaning of life?",
	"This is a very confusing statement.",
	"How does AI work in everyday life?",
	"Colorless green ideas sleep furiously.",
	"What's the capital of France?"
]

# Process the user inputs through the Maxwell's Demon model
maxwell_demon_gpt.process_conversation(user_inputs)

# Display the results of processed and filtered inputs
print("\nProcessed Responses:")
for user_input, response in maxwell_demon_gpt.get_processed_responses():
	print(response)

print("\nFiltered Inputs:")
for user_input, relevance in maxwell_demon_gpt.get_filtered_inputs():
	print(f"{user_input} (Relevance: {relevance:.2f})")