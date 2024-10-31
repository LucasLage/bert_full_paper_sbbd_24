import os
import glob
import pandas as pd

# Get all the files in the current directory
files = glob.glob("./outputs/*")

df_results = pd.DataFrame(columns=["Model", "Date", "Dataset", "Loss", "Accuracy", "F1-Macro", "F1-Weighted", "F1-Micro"])
df_training = pd.DataFrame(columns=["Model", "Date", "Dataset", "Epoch", "Loss", "Accuracy", "F1-Macro", "F1-Weighted"])

index = 0
# Loop through all the files
for file in files:
    # Get the file name
    file_name = os.path.basename(file)
    date = file_name.split("_")[-1] # Date    
    model_name = file_name.split("_")[0] # Model name    
    dataset_name = file_name.split("_")[1:-1] # Dataset name
    dataset_name = " ".join(dataset_name)
    
    if not os.path.exists(f'./outputs/{file_name}/setup_model/results/metrics.csv'):
        continue
    
    # Get results
    df_metrics = pd.read_csv(f'./outputs/{file_name}/setup_model/results/metrics.csv')
    df_metrics['Teste'] = df_metrics['Teste'].apply(lambda x: round(x, 4))
    df_results.loc[index] = [model_name, date, dataset_name, df_metrics['Teste'][0], df_metrics['Teste'][1], df_metrics['Teste'][2], df_metrics['Teste'][3], df_metrics['Teste'][4]]
    
    index += 1

index = 0
# Loop through all the files
for file in files:
    # Get the file name
    file_name = os.path.basename(file)
    date = file_name.split("_")[-1] # Date    
    model_name = file_name.split("_")[0] # Model name    
    dataset_name = file_name.split("_")[1:-1] # Dataset name
    dataset_name = " ".join(dataset_name)
    
    # check if there is a training_metrics.csv file
    if not os.path.exists(f'./outputs/{file_name}/setup_model/results/training_metrics.csv'):
        continue
    
    # Get training
    df_training_metrics = pd.read_csv(f'./outputs/{file_name}/setup_model/results/training_metrics.csv')
    for _, row in df_training_metrics.iterrows():
        
        try:
            epoch = int(row['epoch'])
            loss = float(row['val_loss'])
            accuracy = float(row['val_acc'])
            f1_macro = float(row['val_f1_macro'])
            f1_weighted = float(row['val_f1_weighted'])
        except ValueError:
            continue
        
        df_training.loc[index] = [model_name, date, dataset_name, epoch, round(loss, 4), round(accuracy, 4), round(f1_macro, 4), round(f1_weighted, 4)]
        index += 1
    
            
df_results.to_csv("./results_gpu2.csv", index=False)
df_training.to_csv("./training_gpu2.csv", index=False)
        