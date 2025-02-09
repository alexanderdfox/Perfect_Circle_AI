import openai
openai.api_key = 'your-api-key'

class GPT3Model:
	def generate_response(self, user_input):
		response = openai.Completion.create(
			engine="text-davinci-003",
			prompt=user_input,
			max_tokens=150
		)
		return response.choices[0].text.strip()