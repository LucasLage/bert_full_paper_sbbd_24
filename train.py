import os
import logging
import pandas as pd

from classification.models.Transformer.classification_transformer import DocumentClassification
from classification.utils.utils import split_stratified_into_train_val_test

def train_model(df_labeled, config, run):
    '''
    Train model
    '''

    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid issues with tokenizers parallelism
    
    logging.info('Splitting data...')
    df_train, df_val, df_test, y_train, y_val, y_test = split_stratified_into_train_val_test(df_labeled)
    df_train['label'] = y_train
    df_val['label'] = y_val
    df_test['label'] = y_test
    df_train['fold'] = 'train'
    df_val['fold'] = 'val'
    df_test['fold'] = 'test'
    df_labeled = pd.concat((df_train, df_val, df_test))
    
    train_metrics_list = []
    val_metrics_list = []
    test_metrics_list = []
    max_length_list = []
    metrics_list = []
    final_result_df = pd.DataFrame()

    # Initialize classification model
    document_classification = DocumentClassification(
        df_labeled, # DataFrame with text and labels
        config={
            'log_path': config['output_path'] + '/log.log', # log path
            'batch_size': config['batch_size'], # batch size 
            'lr': config['lr'], # learning rate
            'run': run, # run number
            'num_epochs': config['num_epochs'], # number of epochs
            'max_seq_length': config['max_seq_length'], # max sequence length
            'device': config['device'], # 'cuda' or 'cpu'
            'hugging': config['hugging'] if 'hugging' in config else None, # use huggingface models
            'model_name': config['model_name'], # model name
            'model_path': config['model_path'], # model path
            'text_col': config['text_col'], # text column name
            'target_col': config['target_col'], # target column name
            'class_names': config['class_names'], # class names
            'artifacts_path': config['output_path'] # artifacts path 
        }, 
        device=config['device'] # 'cuda' or 'cpu'
    )

    document_classification.train_model()

    print("="*20, "BEST MODEL", "="*20)
    train_df, train_metrics = document_classification.eval_model(fold="train", probability=True)

    print("Train:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f, F1-Micro %.3f" % (
        train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3], train_metrics[4]))
    logging.info("Train:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f, F1-Micro %.3f" % (
        train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3], train_metrics[4]))

    val_df, val_metrics = document_classification.eval_model(fold="val", probability=True)
    print("Val:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f, F1-Micro %.3f" % (
        val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3], val_metrics[4]))
    logging.info("Val:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f, F1-Micro %.3f" % (
        val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3], val_metrics[4]))

    test_df, test_metrics = document_classification.eval_model(fold="test", probability=True)
    print("Test:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f, F1-Micro %.3f" % (
        test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3], test_metrics[4]))
    logging.info("Test:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f, F1-Micro %.3f" % (
        test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3], test_metrics[4]))

    results_df = pd.concat([train_df, val_df, test_df], ignore_index=True, sort=False)
    results_df['max_length'] = config['max_seq_length']
    final_result_df = pd.concat([final_result_df, results_df], ignore_index=True, sort=False)

    train_metrics_list.extend(train_metrics)
    val_metrics_list.extend(val_metrics)
    test_metrics_list.extend(test_metrics)

    max_length_list.extend([config['max_seq_length']] * 5)
    metrics_list.extend(['loss', 'accuracy', 'F1-Macro', 'F1-Weighted', 'F1-Micro'])

    data = {
        'Max_length': max_length_list,
        'Metrics': metrics_list,
        'Train': train_metrics_list,
        'Val': val_metrics_list,
        'Teste': test_metrics_list
    }

    # Create DataFrame  
    metrics_df = pd.DataFrame(data) 

    return metrics_df, final_result_df