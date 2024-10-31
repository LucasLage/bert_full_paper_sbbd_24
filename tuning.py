from transformers import BertTokenizerFast, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import json
import time
import torch
import pandas as pd

def main(config_file='config.json'):
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Start time
    first_time = time.time()
    
    # Load configuration
    path = './configs/' + config_file
    config = json.load(open(path))
    
    # Extract configuration parameters
    model_path = config['model_path']
    model_name = config['model_name']
    data_path = config['data_path'] + config['dataset'] + '.csv'
    text_col = config['text_col']
    max_length = config['max_seq_length']
    checkpoint_dir = config['checkpoint_dir']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    output_dir = config['output_dir']    

    # Load pre-trained tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)

    # Load data
    print('Loading data...')
    initial_time = time.time()
    data = pd.read_csv(data_path, sep=',', encoding='utf-8', usecols=[text_col])
    data = data.dropna()
    print(f'Data loaded in {time.time() - initial_time:.2f} seconds')

    # Preprocess data
    preprocessed_data = data[text_col].tolist()
    
    # Tokenize data
    print('Tokenizing data...')
    initial_time = time.time()
    tokenized_data = tokenizer(preprocessed_data, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    print(f'Data tokenized in {time.time() - initial_time:.2f} seconds')

    # Define data collator
    print('Creating data collator...')
    initial_time = time.time()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    print(f'Data collator created in {time.time() - initial_time:.2f} seconds')

    # Training arguments
    size = len(tokenized_data['input_ids'])
    training_args = TrainingArguments(
        output_dir=checkpoint_dir + model_name, # output directory
        num_train_epochs=num_epochs, # total number of training epochs
        per_device_train_batch_size=batch_size, # batch size per device during training
        logging_steps=size//num_epochs, # log stats at each epoch
        save_strategy='epoch', # save model at each epoch
        fp16=True, # use mixed precision
        dataloader_num_workers=8 # number of workers for data loading
    )

    # Trainer
    print('Creating trainer...')
    initial_time = time.time()
    trainer = Trainer(
        model=model, # the instantiated Transformers model to be trained
        args=training_args, # training arguments
        data_collator=data_collator, # data collator
        train_dataset=tokenized_data['input_ids'] # training dataset
    )

    # Train the model
    print('Training model...')
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    print(f'Model trained in {time.time() - initial_time:.2f} seconds')

    # Save the tokenizer
    print('Saving tokenizer...')
    tokenizer.save_pretrained(f"{output_dir}finetuned-{model_name}-{config['dataset']}/")

    # Save the model
    print('Saving model...')
    trainer.save_model(f"{output_dir}finetuned-{model_name}-{config['dataset']}/")

    print(f'Execution time: {time.time() - first_time:.2f} seconds')

if __name__ == '__main__':
    configs = ['config_bertimbau_political_small.json', 'config_labse_political_small.json', 'config_bertimbau_victor_small.json', 'config_labse_victor_small.json', 'config_bertimbau_victor_medium.json', 'config_labse_victor_medium.json']
    for config in configs:
        print(f'Running {config}...')
        main(config_file=config)