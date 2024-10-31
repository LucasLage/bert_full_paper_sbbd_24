import os
import json
import logging
import pandas as pd

from datetime import date
from train import train_model
from utils.log_utils import set_logging_config
from tqdm import tqdm

def main(config_file='config.json', runs=5):
      '''
      Main function to train a model
      '''
    
      # Load configuration
      path = './configs/' + config_file
      config = json.load(open(path))
      
      # Set logging configuration
      set_logging_config(config['log_path']) 
      today = date.today() # Get current date
      config['output_path'] = config['output_path'] + f'{config["model_name"]}_{config["input_name"].split(".")[0]}_{today.strftime("%Y%m%d")}' # Set output path
      if not os.path.exists(config['output_path']):
            os.makedirs(config['output_path']) # Create output path
      
      logging.info('Getting input data..')
      file_name_data = config['input_path'] + config['input_name']      
      
      logging.info('Reading saved data...')
      df_data = pd.read_csv(file_name_data, low_memory=False, sep=config['separator'], encoding='utf-8', compression=config['compression'], usecols=[config['columns']])
      df_data = df_data.rename(columns={
            config['doc_id']: 'doc_id', 
            config["target_col"]: 'label',
            config["text_col"]: 'text_preprocessed'}).dropna()
      df_data['city'] = 'NA' if 'city' not in df_data.columns else df_data['city']

      logging.info('Mapping labels...')
      df_data['label_int'] = df_data['label'].map(config['class_names'])
      
      for run in tqdm(range(runs), desc='Training model'):
            logging.info(f'Run {run+1}')
            print(f'Run {run+1}')
            
            logging.info('Training Model...')
            metrics_df, final_result_df = train_model(df_data, config, run+1)
            
            logging.info('Saving results...')
            metrics_df.to_csv(config['output_path'] + f'/results/metrics_{run+1}.csv')
            final_result_df.to_csv(config['output_path'] + f'/results/final_result_{run+1}.csv')

      logging.info('Model created successfully')
      
if __name__ == '__main__':
      
      config_files = ['config_lipset_licitbert.json', 'config_lipset_bertimbau.json', 'config_lipset_legalbert_pt_br.json', 'config_lipset_legalbert_pt.json', 'config_lipset_lener_br.json', 'config_lipset_libert_se.json', 'config_napex_licitbert.json', 'config_napex_bertimbau.json', 'config_napex_legalbert_pt_br.json', 'config_napex_legalbert_pt.json', 'config_napex_lener_br.json', 'config_napex_libert_se.json', 'config_victor_licitbert.json', 'config_victor_bertimbau.json', 'config_victor_legalbert_pt_br.json', 'config_victor_legalbert_pt.json', 'config_victor_lener_br.json', 'config_victor_libert_se.json', 'config_prodserv_licitbert.json', 'config_prodserv_bertimbau.json', 'config_prodserv_legalbert_pt_br.json', 'config_prodserv_legalbert_pt.json', 'config_prodserv_lener_br.json', 'config_prodserv_libert_se.json']
      for config_file in config_files:
            print(f'Running {config_file}')
            main(config_file, runs=5)