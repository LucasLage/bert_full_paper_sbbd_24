import os
import logging
import pandas as pd

import seaborn as sns
from tqdm.autonotebook import trange, tqdm

import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import Config
from .transformer import Transformer
from classification.utils.utils import (
    set_seed,
    # plot_training_loss
)
from classification.utils.load_data import (
    load_text_data
)
from classification.evaluate.evaluate import (
    calculate_metrics
)

sns.set(font_scale=1.5)
sns.set(rc={'figure.figsize': (10, 10)})
torch.multiprocessing.set_sharing_strategy('file_system')

class DocumentClassification:

    def __init__(self, input_data, config=None, seed=42, device=None):

        logging.info("Initializing DocumentClassification...")
        if config is not None:
            self.config = Config(**config)
        else:
            self.config = Config()

        logging.info("Setting device...")
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.seed = seed
        set_seed(self.seed)
        self.config.num_classes = len(self.config.class_names)
        
        if self.config.hugging:
            model_name_or_path = self.config.hugging
        else:
            model_name_or_path = self.config.model_path + self.config.model_name

        logging.info("Creating model...")
        self.model = Transformer(
            model_name_or_path = model_name_or_path,
            max_seq_length = self.config.max_seq_length,
            num_classes = self.config.num_classes,
            model_args = self.config.model_args,
            tokenizer_args = self.config.tokenizer_args,
            do_lower_case = self.config.do_lower_case,
            pooling_mode = self.config.pooling_mode
        )
        
        if not os.path.exists(self.config.artifacts_path):
            os.makedirs(self.config.artifacts_path)
            os.makedirs(self.config.artifacts_path + '/model')
            os.makedirs(self.config.artifacts_path + '/results')
        
        logging.info("Saving config file...")
        self.config.save_config()

        logging.info("Creating data loaders...")
        self.df_data = input_data
        self.data_loaders = self.data_load()

    def split_data(self, fold):
        """ Split data into train, validation and test sets

        Args:
            fold (string): fold name

        Returns:
            X (list): list of input data
            y (list): list of labels
        """       
        
        X = self.df_data.loc[self.df_data['fold'] == fold, ["city", "doc_id", "text_preprocessed"]].values 
        y = self.df_data.loc[self.df_data['fold'] == fold, 'label_int'].values

        return X, y
    
    def data_load(self):
        """ Load data and create data loaders
        
        Returns:
            data_loaders (dict): dictionary with data loaders
        """
                
        # Split data
        x_train, y_train = self.split_data("train")
        x_val, y_val = self.split_data("val")
        x_test, y_test = self.split_data("test")

        # Load dataset
        train_set = load_text_data(x_train, y_train)
        val_set = load_text_data(x_val, y_val)
        test_set = load_text_data(x_test, y_test)

        data_loaders = dict()

        # Create data loaders for train, validation and test sets
        data_loaders["train"] = torch.utils.data.DataLoader(
            train_set,
            batch_size=int(self.config.batch_size),
            shuffle=True, # set to True for training
            num_workers=8)        
        data_loaders["val"] = torch.utils.data.DataLoader(
            val_set,
            batch_size=int(self.config.batch_size),
            shuffle=False,
            num_workers=8)
        data_loaders["test"] = torch.utils.data.DataLoader(
            test_set,
            batch_size=int(self.config.batch_size),
            shuffle=False,
            num_workers=8)

        return data_loaders

    def train_model(self, show_progress_bar=True, extended_metrics=True, save_model_per_epoch=False):
        
        logging.info("Training model...")
        checkpoint_dir = self.config.artifacts_path
        data_loaders = self.data_loaders
        
        # Set run number
        run = self.config.run
        
        # Create directory to save best model
        path = os.path.join(checkpoint_dir.split('config')[0]) + f'/model/{run}/best/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Set number of epochs
        if not self.config.patience:
            self.config.patience = self.config.num_epochs
        patience_counter = 0
        
        # Get max sequence length
        max_seq_length = self.config.max_seq_length

        # Get data loaders
        train_loader = data_loaders["train"]
        val_loader = data_loaders["val"]

        # Set model to training mode
        self.model.to(self.device)
        best_loss = float("inf") # set to infinity
        best_macro = 0.0 # set to 0
        
        # Set optimizer and loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        # Get number of steps per epoch
        steps_per_epoch = len(train_loader)
        data_iterator = iter(train_loader)
        
        loss_values = []
        best_epoch = 0
        # loop over the dataset multiple times
        for epoch in trange(self.config.num_epochs, desc="Epoch", disable=not show_progress_bar):
            print("")
            print("=" * 20, "Epoch {:} / {:}".format(epoch + 1, self.config.num_epochs), "=" * 20)
            os.makedirs(os.path.join(checkpoint_dir.split('config')[0]) + f'/model/{run}/' + str(epoch+1), exist_ok=True) 
            
            self.model.train()
            training_loss = 0.0
            epoch_steps = 0
            y_pred = []
            y_true = []
            
            for _ in trange(steps_per_epoch, desc="Training...", smoothing=0.05, disable=not show_progress_bar):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_loader)
                    data = next(data_iterator)

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, _, _ = data
                labels = labels.long().to(self.device)
                inputs = self.model.tokenize(inputs)
                inputs['input_ids'] = torch.Tensor(inputs['input_ids'].tolist()).long().to(self.device)
                inputs['token_type_ids'] = torch.Tensor(inputs['token_type_ids'].tolist()).long().to(self.device)
                inputs['attention_mask'] = torch.Tensor(inputs['attention_mask'].tolist()).long().to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                epoch_steps += 1
                training_loss += loss.cpu().detach().numpy()

                _, predicted = torch.max(outputs.data, 1)

                y_pred.extend(predicted.cpu().tolist())
                y_true.extend(labels.cpu().tolist())
            
            acc, f1_macro, f1_weighted, f1_micro = calculate_metrics(y_true, y_pred)
            train_metrics = [(training_loss / epoch_steps), acc, f1_macro, f1_weighted]
            if extended_metrics:
                train_metrics += [f1_micro]
            
            loss_values.append(train_metrics[0])
            
            print("Train:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f" % (
                train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3]))

            _, val_metrics = self.eval_model(fold="val", probability=True)
            print("Val:\n loss %.3f, accuracy %.3f, F1-Macro %.3f, F1-Weighted %.3f \n" % (
                val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3]))
            
            data = {
                "epoch": epoch+1,
                "train_loss": train_metrics[0],
                "train_acc": train_metrics[1],
                "train_f1_macro": train_metrics[2],
                "train_f1_weighted": train_metrics[3],
                "val_loss": val_metrics[0],
                "val_acc": val_metrics[1],
                "val_f1_macro": val_metrics[2],
                "val_f1_weighted": val_metrics[3]
            }
            
            path = os.path.join(checkpoint_dir.split('config')[0]) + '/results/'
            metrics_df = pd.DataFrame(data, index=[0], columns=data.keys())
            metrics_df.to_csv(f"{path}/training_metrics_{run+1}.csv", mode='a', index=False)

            # updating best model
            if val_metrics[0] < best_loss - 0.001:
                best_epoch = epoch
                best_loss = val_metrics[0]
                best_macro = val_metrics[2]
                patience_counter = 0
                
                # Save model
                path = os.path.join(checkpoint_dir.split('config')[0]) + f'/model/{run}/best/'
                self.model.save(path)
                torch.save(self.model, f"{path}/bert_model_{max_seq_length}.pt") # Save
                with open(f"{path}/best_model_{max_seq_length}.txt", "w") as f:
                    f.write(f"Best epoch: {best_epoch}\n")
                    f.write(f"Best loss: {best_loss}\n")
                    f.write(f"Best macro: {best_macro}\n")
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                print("Model training was stopped early")
                break
            
            # save checkpoint
            if save_model_per_epoch:
                path = os.path.join(checkpoint_dir.split('config')[0]) + f'/model/{run}/{epoch}/'
                if not os.path.exists(path):
                    os.makedirs(path)

                self.model.save(path)
                torch.save(self.model, f"{path}/bert_model_{max_seq_length}.pt") # Save
                
            logging.info("Cleaning cuda cache...")
            torch.cuda.empty_cache()

        logging.info("Finished Training")
        # plot_training_loss(loss_values)
        self.model = torch.load(os.path.join(checkpoint_dir.split('config')[0]) + f'/model/{run}/best/bert_model_{max_seq_length}.pt')

    def eval_model(self, show_progress_bar=True, fold=None, probability=False, logits=False, extended_metrics=True):
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        loader = self.data_loaders[fold]
        steps = 0
        sum_loss = 0.0
        y_true = []
        y_pred = []
        cities = []
        docs = []
        probabilities = []
        logits = []
        with torch.no_grad():
            for data in tqdm(loader, total=len(loader), desc=f"Evaluating {fold}...", disable=not show_progress_bar):
                if fold:
                    inputs, labels, city, doc_id = data
                    cities.extend(city)
                    docs.extend(doc_id)
                else:
                    inputs, labels, _, _ = data

                labels = labels.long().to(self.device)
                inputs = self.model.tokenize(inputs)
                inputs['input_ids'] = torch.Tensor(inputs['input_ids'].tolist()).long().to(self.device)
                inputs['token_type_ids'] = torch.Tensor(inputs['token_type_ids'].tolist()).long().to(self.device)
                inputs['attention_mask'] = torch.Tensor(inputs['attention_mask'].tolist()).long().to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                steps += 1
                sum_loss += loss.cpu().detach().numpy()

                y_pred.extend(predicted.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

                prob = F.softmax(outputs, dim=1).cpu().detach().numpy()                
                probabilities.extend(prob)
                logits.extend(outputs.cpu().detach().numpy())

        acc, f1_macro, f1_weighted, f1_micro = calculate_metrics(y_true, y_pred)
        metrics = [(sum_loss / steps), acc, f1_macro, f1_weighted]
        if extended_metrics:
            metrics.append(f1_micro)
        
        df_predictions = pd.DataFrame()
        if fold:
            df_predictions = pd.DataFrame({
                "doc_id": docs, 
                "city": cities, 
                "label": y_true, 
                "pred": y_pred,
                "fold": fold
            })            
            
            if probability:
                df_predictions['probabilities'] = probabilities
            if logits:
                df_predictions['logits'] = logits
                
        return df_predictions, metrics