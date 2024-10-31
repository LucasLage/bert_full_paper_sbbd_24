import json

from classification.utils.np_encoder import NpEncoder

class Config:
    def __init__(self, hugging=None, model_name=None, model_path=None,
                 max_seq_length=512, num_classes=13, model_args={}, 
                 tokenizer_args={}, do_lower_case=False, pooling_mode=None, lr=0.001, 
                 batch_size=24, num_epochs=10, patience=5, artifacts_path='./transformer_data/',
                 log_path='./log.log', device='cuda', text_col='text', target_col='label',
                 input_path='./data/', input_name='data.csv', separator=',', compression=None,
                 task='item_classification', class_names={}):

        self.model_name = model_name
        self.model_path = model_path
        self.hugging = hugging
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.model_args = model_args
        self.tokenizer_args = tokenizer_args
        self.do_lower_case = do_lower_case
        self.pooling_mode = pooling_mode
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.artifacts_path = artifacts_path
        self.log_path = log_path
        self.device = device
        self.text_col = text_col
        self.target_col = target_col
        self.input_path = input_path
        self.input_name = input_name
        self.separator = separator
        self.compression = compression
        self.task = task
        self.class_names = class_names
        

    def get_config_dict(self):

        config_dict = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'hugging': self.hugging,
            'max_seq_length': self.max_seq_length,
            'num_classes': self.num_classes,
            'model_args': self.model_args,
            'tokenizer_args': self.tokenizer_args,
            'do_lower_case': self.do_lower_case,
            'pooling_mode': self.pooling_mode,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'artifacts_path': self.artifacts_path,
            'log_path': self.log_path,
            'device': self.device,
            'text_col': self.text_col,
            'target_col': self.target_col,
            'input_path': self.input_path,
            'input_name': self.input_name,
            'separator': self.separator,
            'compression': self.compression,    
            'task': self.task,
            'class_names': self.class_names
        }

        return config_dict

    def load_config(self, path):

        self.artifacts_path = path

        with open(self.artifacts_path + 'config.json', 'r') as config_file:
            config_dict = json.load(config_file)

        self.model_name = config_dict['model_name']
        self.model_path = config_dict['model_path']
        self.hugging = config_dict['hugging']
        self.max_seq_length = config_dict['max_seq_length']
        self.num_classes = config_dict['num_classes']
        self.model_args = config_dict['model_args']
        self.tokenizer_args = config_dict['tokenizer_args']
        self.do_lower_case = config_dict['do_lower_case']
        self.pooling_mode = config_dict['pooling_mode']
        self.lr = config_dict['lr']
        self.batch_size = config_dict['batch_size']
        self.num_epochs = config_dict['num_epochs']
        self.patience = config_dict['patience']
        self.artifacts_path = config_dict['artifacts_path']
        self.log_path = config_dict['log_path']
        self.device = config_dict['device']
        self.text_col = config_dict['text_col']
        self.target_col = config_dict['target_col']
        self.input_path = config_dict['input_path']
        self.input_name = config_dict['input_name']
        self.separator = config_dict['separator']
        self.compression = config_dict['compression']
        self.task = config_dict['task']
        self.class_names = config_dict['class_names']

    def save_config(self):

        config_dict = self.get_config_dict()
        
        with open(self.artifacts_path + '/config.json', 'w') as config_file:
            json.dumps(config_dict, cls=NpEncoder)
