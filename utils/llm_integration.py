from llama_cpp import Llama
from .intent_classifier import IntentClassifier

class GeoChatbot:
    def __init__(self):
        self.intent_classifier = IntentClassifier("./models/bert_model")
        self.llm = Llama(model_path="./models/gemma-2-2b-it-Q4_K_M.gguf", n_ctx=2048)
    
    def generate_response(self, query):
        intent_result = self.intent_classifier.predict(query)
        prompt = f"Geospatial question: {query} (Intent: {intent_result['intent']})"
        response = self.llm.create_chat_completion(messages=[{"role": "user", "content": prompt}])
        return response['choices'][0]['message']['content']