import os
import json
import torch

from torch import nn
from typing import List
from typing import List, Dict, Optional
from classification.utils.np_encoder import NpEncoder
from transformers import AutoModel, AutoTokenizer, AutoConfig

class Transformer(nn.Module):
    """
    Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, model_name_or_path: str = "", model_paths: dict = {}, 
                 max_seq_length: Optional[int] = None, num_classes: int = 13,
                 model_args: Dict = {}, tokenizer_args: Dict = {},
                 do_lower_case: bool = False, pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False):

       
        assert model_name_or_path or model_paths.keys(), 'At least pre-trained model name or his paths are expected'
    
        super(Transformer, self).__init__()

        self.config_keys = ['max_seq_length', 'do_lower_case', \
                            'pooling_mode_cls_token', 'pooling_mode_mean_tokens', \
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        
        if model_paths.keys():
            config = AutoConfig.from_pretrained(model_paths['config'], **model_args)
            self.auto_model = AutoModel.from_pretrained(model_paths['model'], config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_paths['tokenizer'], **tokenizer_args)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, **model_args)
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
        
        self.word_embedding_dimension = config.hidden_size

        if pooling_mode is not None:        #Set pooling mode by sttring
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens,
                                       pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * self.word_embedding_dimension)
        self.sentence_embedding_dimension = self.pooling_output_dimension

        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(self.sentence_embedding_dimension, num_classes)


    def forward(self, features):

        features = self.get_features(features)
        pooled_output = self.dropout(features['all_layer_embeddings'])
        output = self.classifier(pooled_output)

        return output

    def pooling(self, features):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_features(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

        all_layer_idx = 1
        hidden_states = output_states[all_layer_idx]
        features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: List[str]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        to_tokenize = [texts]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        #strip
        to_tokenize = [[s.strip() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'classification_bert_config.json'), 'w') as fOut:
            json.dumps(self.get_config_dict(), cls=NpEncoder, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['classification_bert_config.json', 'classification_roberta_config.json', \
                            'classification_distilbert_config.json', 'classification_camembert_config.json', \
                            'classification_albert_config.json', 'classification_xlm-roberta_config.json', \
                            'classification_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return ClassificationBERT(model_name_or_path=input_path, **config)
    
    def predict_text(self, text, device, returns_label=True):
        tokenized = self.tokenize([text])
        tokenized['input_ids'] = torch.Tensor(tokenized['input_ids'].tolist()[0])
        tokenized['token_type_ids'] = torch.Tensor(tokenized['token_type_ids'].tolist()[0])
        tokenized['attention_mask'] = torch.Tensor(tokenized['attention_mask'].tolist()[0])
        tokenized['input_ids'] = tokenized['input_ids'].unsqueeze(0).long().to(device)
        tokenized['token_type_ids'] = tokenized['token_type_ids'].unsqueeze(0).long().to(device)
        tokenized['attention_mask'] = tokenized['attention_mask'].unsqueeze(0).long().to(device)
        
        output = self(tokenized)        
        if returns_label:
            _, prediction = torch.max(output, dim=1)
            predicted_label = prediction.cpu().tolist()
            return predicted_label[0]
        else:
            return output
        
    def predict_text_list(self, text_list, device, returns_label=True):
        tokenized = self.tokenize(text_list)
  
        input_ids = tokenized['input_ids'].long().to(device)
        token_type_ids = tokenized['token_type_ids'].long().to(device)
        attention_mask = tokenized['attention_mask'].long().to(device)
        
        output = self({'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})
        if returns_label:
            _, prediction = torch.max(output, dim=1)
            predicted_label = prediction.cpu().tolist()
            return predicted_label
        else:
            return output
