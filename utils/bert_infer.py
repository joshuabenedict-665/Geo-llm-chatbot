from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

class IntentClassifier:
    def __init__(self, model_path="bert_intent"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.labels = {
            0: ("tool_query", "Questions about geospatial tools"),
            1: ("district_query", "Questions about Tamil Nadu districts"), 
            2: ("general", "Conceptual geospatial questions")
        }

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        return {
            "text": text,
            "predicted_intent": self.labels[pred_idx][0],
            "description": self.labels[pred_idx][1],
            "confidence": float(confidence),
            "probabilities": {
                self.labels[i][0]: float(probs[i]) for i in self.labels
            }
        }

def print_result(result):
    print(f"\nQuery: {result['text']}")
    print(f"Predicted intent: {result['predicted_intent']}")
    print(f"Description: {result['description']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    print("\nDetailed probabilities:")
    for intent, prob in result['probabilities'].items():
        print(f"  {intent:<15}: {prob:.1%}")

if __name__ == "__main__":
    classifier = IntentClassifier()
    
    test_queries = [
        "how to clip rasters in QGIS",
        "population density in coimbatore",
        "explain NDVI vegetation index"
    ]
    
    for query in test_queries:
        result = classifier.predict(query)
        print_result(result)
    
    while True:
        try:
            query = input("\nEnter a query (or 'quit'): ").strip()
            if query.lower() in ['quit', 'exit']:
                break
            result = classifier.predict(query)
            print_result(result)
        except KeyboardInterrupt:
            break