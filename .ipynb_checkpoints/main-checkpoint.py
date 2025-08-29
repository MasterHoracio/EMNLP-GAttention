from Utilities import pipeline
import argparse
import torch
import copy

def main(dataset_name, task, PRE_TRAINED_MODEL_NAME, architecture_mode, MAX_LENGTH, encoding_dimension, is_tsv, unique, validation, save_labels):
    pipe = pipeline.pipeline(dataset_name, task, PRE_TRAINED_MODEL_NAME, architecture_mode, MAX_LENGTH, encoding_dimension, is_tsv, unique, validation)
    
    if dataset_name == "AMI" and task == "B":
        path, label_transform, label_transform_2, N_CLASSES, N_CLASSES_2, labels, target_names, labels_2, target_names_2, epochs, BATCH_SIZE, LEARNING_RATE = pipe.get_hyperparameters()
        x_train, y_train, y2_train, x_test, y_test, y2_test = pipe.get_full_dataset(path, label_transform, label_transform_2)
    else:
        path, label_transform, N_CLASSES, labels, target_names, epochs, BATCH_SIZE, LEARNING_RATE = pipe.get_hyperparameters()
        x_train, y_train, y2_train, x_test, y_test, y2_test = pipe.get_full_dataset(path, label_transform)

    train_dataloader, test_dataloader = pipe.tokenize_full_dataset(x_train, y_train, y2_train, x_test, y_test, y2_test, BATCH_SIZE)
    
    if dataset_name == "AMI" and task == "B":
        model, n_params, device = pipe.load_model(N_CLASSES, N_CLASSES_2)
    else:
        model, n_params, device = pipe.load_model(N_CLASSES)

    print(f"Total parameters: {n_params}")

    t_handler = pipe.train_model(model, device, train_dataloader, epochs, LEARNING_RATE)

    if dataset_name == "AMI" and task == "B":
        f1_test, model_prediction_test, ground_truth_test, f1_test_2, model_prediction_test_2, ground_truth_test_2 = pipe.evaluate(t_handler, model, test_dataloader)
        pipe.report(ground_truth_test, model_prediction_test, labels, target_names, f1_test, save_labels, ground_truth_test_2, model_prediction_test_2, labels_2, target_names_2, f1_test_2)
    else:
        f1_test, model_prediction_test, ground_truth_test = pipe.evaluate(t_handler, model, test_dataloader)
        pipe.report(ground_truth_test, model_prediction_test, labels, target_names, f1_test, save_labels)

    if dataset_name == "AMI" and task == "B":
        f1_test = (f1_test + f1_test_2)/2
    
    return model, f1_test
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subtask", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--architecture_mode", type=str, required=True)
    args = parser.parse_args()
    
    #dataset_name                  = "AMI" # "SEM" or "AMI" or "HAS"
    #task                          = "B" # "A" or "B"
    #PRE_TRAINED_MODEL_NAME        = "nghuyong/ernie-2.0-base-en" # "roberta-base", "nghuyong/ernie-2.0-base-en", "bert-base-uncased", "distilbert-base-uncased"
    #architecture_mode             = "MHG" # MHG, GAT, "NON"
    
    dataset_name = args.dataset
    task = args.subtask
    PRE_TRAINED_MODEL_NAME = args.model
    architecture_mode = args.architecture_mode
    
    # Define some network parameters
    encoding_dimension            = 768 # Define the encoding dimension (according to BERT) 768
    MAX_LENGTH                    = 64  # Define the maximum length of the sequence
    
    is_tsv                        = True
    unique                        = False
    validation                    = False
    save_labels                   = False
    
    iterations                    = 1
    save_best_model               = False
    best_macro_f1                 = 0.0
    
    for i in range(iterations):
        model, macro_f1 = main(dataset_name, task, PRE_TRAINED_MODEL_NAME, architecture_mode, MAX_LENGTH, encoding_dimension, is_tsv, unique, validation, save_labels)
        if save_best_model and macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            overall_best_state_dict = copy.deepcopy(model.state_dict())
            model_name = architecture_mode + "_" + dataset_name + "_" + task + "_"
            torch.save(model.state_dict(), model_name+'MODEL.pth')