from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
import evaluate
import logging
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
import evaluate
import logging
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Training dataset with clear examples"""
    return {
        'text': [
            # tool_query
            'how to merge tif files using gdal',
            'gdalwarp command for reprojection',
            'create slope map with whitebox tools',
            'clip vector layers in QGIS',
            
            # district_query
            'average elevation in nilgiris district',
            'land use classification in salem',
            'population density chennai 2023',
            
            # general
            'explain flow accumulation in GIS',
            'what is NDVI in remote sensing',
            'difference between DEM and DSM'
        ],
        'label': [
            0,0,0,0,  # tool_query
            1,1,1,    # district_query
            2,2,2     # general
        ]
    }

def main():
    try:
        logger.info("Preparing dataset...")
        dataset = Dataset.from_dict(load_data())
        
        logger.info("Initializing tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def tokenize(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=64
            )
        
        logger.info("Tokenizing dataset...")
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.remove_columns(['text'])
        dataset.set_format('torch')
        
        logger.info("Loading model...")
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        )
        
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            logging_dir='./logs',
            logging_steps=5,
            save_steps=50
        )
        
        logger.info("Starting training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
        
        logger.info("Evaluating model...")
        predictions = trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        
        report = classification_report(
            predictions.label_ids,
            preds,
            target_names=['tool_query', 'district_query', 'general']
        )
        logger.info(f"Classification Report:\n{report}")
        
        logger.info("Saving model...")
        trainer.save_model("bert_intent")
        tokenizer.save_pretrained("bert_intent")
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
def main():
    try:
        logger.info("Preparing dataset...")
        dataset = Dataset.from_dict(load_data())
        
        logger.info("Initializing tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def tokenize(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=64
            )
        
        logger.info("Tokenizing dataset...")
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.remove_columns(['text'])
        dataset.set_format('torch')
        
        logger.info("Loading model...")
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        )
        
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            logging_dir='./logs',
            logging_steps=5,
            save_steps=50
        )
        
        logger.info("Starting training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
        
        logger.info("Evaluating model...")
        predictions = trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        
        report = classification_report(
            predictions.label_ids,
            preds,
            target_names=['tool_query', 'district_query', 'general']
        )
        logger.info(f"Classification Report:\n{report}")
        
        logger.info("Saving model...")
        trainer.save_model("bert_intent")
        tokenizer.save_pretrained("bert_intent")
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()