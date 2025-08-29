import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch

class trainhandler():
    def __init__(self, device, scheduler):
        self.device = device
        self.scheduler = scheduler
        super(trainhandler, self).__init__()
    
    # Calculates the total number of correctly classified labels
    def sum_correct(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat), pred_flat, labels_flat

    def evaluate_model_bert(self, model, dataloader, two_predictions = False):
        total_correct = 0
        total_count = 0
        model_prediction = []
        ground_truth = []
        model_prediction2 = []
        ground_truth2 = []

        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_token_ids = batch[1].to(self.device)
                b_input_mask = batch[2].to(self.device)
                b_labels = batch[3].to(self.device)
                if two_predictions:
                    b_labels2 = batch[4].to(self.device)

                if two_predictions:
                    output, output2 = model(b_input_ids, b_input_mask, b_input_token_ids)
                else:
                    output = model(b_input_ids, b_input_mask, b_input_token_ids)
                
                # Move labels and predictions to the CPU
                output = output.detach().cpu().numpy()
                if two_predictions:
                    output2 = output2.detach().cpu().numpy()
                
                label_ids = b_labels.to('cpu').numpy()
                if two_predictions:
                    label_ids2 = b_labels2.to('cpu').numpy()

                # Sum the correct predictions
                correct, predictions, labels = self.sum_correct(output, label_ids)
                if two_predictions:
                    correct2, predictions2, labels2 = self.sum_correct(output2, label_ids2)
                
                model_prediction.append(predictions.tolist())
                ground_truth.append(labels.tolist())

                if two_predictions:
                    model_prediction2.append(predictions2.tolist())
                    ground_truth2.append(labels2.tolist())
                
                total_correct += correct
                total_count += len(b_labels)

        model.train()

        ground_truth     = [item for sublist in ground_truth for item in sublist]
        model_prediction = [item for sublist in model_prediction for item in sublist]

        if two_predictions:
            ground_truth2     = [item for sublist in ground_truth2 for item in sublist]
            model_prediction2 = [item for sublist in model_prediction2 for item in sublist]

        accuracy = total_correct/total_count
        recall = metrics.recall_score(ground_truth, model_prediction, average=None)
        
        f1 = metrics.f1_score(ground_truth, model_prediction, average='macro')# average='macro'
        if two_predictions:
            f1_2 = metrics.f1_score(ground_truth2, model_prediction2, average='macro')# average='macro'

        if two_predictions:
            return f1, model_prediction, ground_truth, f1_2, model_prediction2, ground_truth2
        else:
            return f1, model_prediction, ground_truth

    def evaluate_model_distil_bert(self, model, dataloader, two_predictions = False):
        total_correct = 0
        total_count = 0
        model_prediction = []
        ground_truth = []
        model_prediction2 = []
        ground_truth2 = []

        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                if two_predictions:
                    b_labels2 = batch[3].to(self.device)
                    
                if two_predictions:
                    output, output2 = model(b_input_ids, b_input_mask)
                else:
                    output = model(b_input_ids, b_input_mask)
                
                # Move labels and predictions to the CPU
                output = output.detach().cpu().numpy()
                if two_predictions:
                    output2 = output2.detach().cpu().numpy()
                
                label_ids = b_labels.to('cpu').numpy()
                if two_predictions:
                    label_ids2 = b_labels2.to('cpu').numpy()

                # Sum the correct predictions
                correct, predictions, labels = self.sum_correct(output, label_ids)
                if two_predictions:
                    correct2, predictions2, labels2 = self.sum_correct(output2, label_ids2)
                
                model_prediction.append(predictions.tolist())
                ground_truth.append(labels.tolist())
                
                if two_predictions:
                    model_prediction2.append(predictions2.tolist())
                    ground_truth2.append(labels2.tolist())
                
                total_correct += correct
                total_count += len(b_labels)

        model.train()

        ground_truth     = [item for sublist in ground_truth for item in sublist]
        model_prediction = [item for sublist in model_prediction for item in sublist]
        if two_predictions:
            ground_truth2     = [item for sublist in ground_truth2 for item in sublist]
            model_prediction2 = [item for sublist in model_prediction2 for item in sublist]

        recall = metrics.recall_score(ground_truth, model_prediction, average=None)
        f1 = metrics.f1_score(ground_truth, model_prediction, average='macro')
        if two_predictions:
            f1_2 = metrics.f1_score(ground_truth2, model_prediction2, average='macro')
        if two_predictions:
            return f1, model_prediction, ground_truth, f1_2, model_prediction2, ground_truth2
        else:
            return f1, model_prediction, ground_truth

    # Training functions
    def train_bert(self, epoch, total_epoch, model, iterator, optimizer, criterion, scheduler, two_predictions = False, clip=1):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        total_correct = 0
        total_count = 0

        loop = tqdm(enumerate(iterator), total=len(iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        #for batch_index, (input_ids, input_token_ids, input_mask, labels, labels2) in loop:
        for batch_index, batch in loop:
            if two_predictions:
                input_ids, input_token_ids, input_mask, labels, labels2 = batch
            else:
                input_ids, input_token_ids, input_mask, labels = batch
                
            b_input_ids = input_ids.to(self.device)
            b_input_token_ids = input_token_ids.to(self.device)
            b_input_mask = input_mask.to(self.device)
            b_labels = labels.to(self.device)
            if two_predictions:
                b_labels2 = labels2.to(self.device)

            # Reset the values of the gradients
            model.zero_grad()
            if two_predictions:
                output, output2  = model(b_input_ids, b_input_mask, b_input_token_ids)
            else:
                output  = model(b_input_ids, b_input_mask, b_input_token_ids)
            
            # Compute batch accuracy
            total_correct += torch.sum(torch.eq(output.argmax(1), b_labels))#task 1
            total_count += len(b_labels)
            if two_predictions:
                total_correct += torch.sum(torch.eq(output2.argmax(1), b_labels2))#task 2
                total_count += len(b_labels2)
            
            batch_accuracy = total_correct / total_count

            if two_predictions:
                loss1 = criterion(output, b_labels)
                loss2 = criterion(output2, b_labels2)

                loss = (0.75*loss1) + (0.25*loss2)
            else:
                loss = criterion(output, b_labels)

            # Calculate the gradients
            loss.backward()
            # Prevent the gradient explotion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # Update the weigths of the model
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += batch_accuracy

            # Update progress bar
            loop.set_description(f"Epoch[{epoch}/{total_epoch}]")
            loop.set_postfix(loss = loss.item(), acc = (total_correct/total_count))

        mean_loss = epoch_loss / len(iterator)
        mean_acc = epoch_acc / len(iterator)
        # Update the LR
        #self.scheduler.step(mean_loss)

        print(f'Train accuracy: {(mean_acc):.6f}, Mean loss: {mean_loss:.6f}')

    def train_distil_bert(self, epoch, total_epoch, model, iterator, optimizer, criterion, scheduler, two_predictions = False, clip=1):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        total_correct = 0
        total_count = 0

        loop = tqdm(enumerate(iterator), total=len(iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        #for batch_index, (input_ids, input_mask, labels, labels2) in loop:
        for batch_index, batch in loop:
            if two_predictions:
                input_ids, input_mask, labels, labels2 = batch
            else:
                input_ids, input_mask, labels = batch
            
            b_input_ids = input_ids.to(self.device)
            b_input_mask = input_mask.to(self.device)
            b_labels = labels.to(self.device)
            if two_predictions:
                b_labels2 = labels2.to(self.device)

            # Reset the values of the gradients
            model.zero_grad()

            if two_predictions:
                output, output2  = model(b_input_ids, b_input_mask)
            else:
                output  = model(b_input_ids, b_input_mask)

            # Compute batch accuracy
            total_correct += torch.sum(torch.eq(output.argmax(1), b_labels))
            total_count += len(b_labels)

            if two_predictions:
                total_correct += torch.sum(torch.eq(output2.argmax(1), b_labels2))
                total_count += len(b_labels2)
            
            batch_accuracy = total_correct / total_count

            if two_predictions:
                loss1 = criterion(output, b_labels)
                loss2 = criterion(output2, b_labels2)

                loss = (0.75*loss1) + (0.25*loss2)
            else:
                loss = criterion(output, b_labels)

            # Calculate the gradients
            loss.backward()
            # Prevent the gradient explotion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # Update the weigths of the model
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_acc += batch_accuracy

            # Update progress bar
            loop.set_description(f"Epoch[{epoch}/{total_epoch}]")
            loop.set_postfix(loss = loss.item(), acc = (total_correct/total_count))

        mean_loss = epoch_loss / len(iterator)
        mean_acc = epoch_acc / len(iterator)
        
        # Update the LR
        #self.scheduler.step(mean_loss)

        print(f'Train accuracy: {(mean_acc):.6f}, Mean loss: {mean_loss:.6f}')
