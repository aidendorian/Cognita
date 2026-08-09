from google.genai import Client
from config.env import api_key, llm_model

class LLM:
    def __init__(self):
        self.client = Client(api_key=api_key)
        self.model = llm_model
        
    def generate(self, prompt : str):
        response = self.client.models.generate_content(model = self.model,
                                                       contents = prompt)
        return response.text