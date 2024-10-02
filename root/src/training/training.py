import torch
from utils.data_utils import seed_everything, load_checkpoint, save_checkpoint, plot_metrics, calculate_metrics, plot_learning_rate_and_regularization, plot_accuracies, create_confusion_matrix

# Load the config file using the config_loader module
from utils.config_loader import load_config


cfg = load_config()

def train_and_validate_eeg(model, train_loader, valid_loader, epochs, optimizer, criterion, device, checkpoint_dir, logger, new_checkpoint, combination_idx, gamma, step_size, initial_model_state, initial_optimizer_state):
    # Reset the model and optimizer to their initial states before starting each new grid combination
    model.load_state_dict(initial_model_state)
    optimizer.load_state_dict(initial_optimizer_state)
    
    # Generate the checkpoint filename based on the combination index
    checkpoint_filename = f"eeg_checkpoint_index_{combination_idx}.pth.tar"
    
    seed_everything()
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler = load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer, new_checkpoint)

    # Lists to store metrics over epochs
    train_precisions, valid_precisions = [], []
    train_recalls, valid_recalls = [], []
    train_f1s, valid_f1s = [], []
    
    # Manually track epochs since the last learning rate reduction
    epochs_since_lr_drop = 0

    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting Epoch {epoch+1}/{epochs}")
        
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            labels = labels.argmax(dim=1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        train_loss = running_train_loss / total_train
        train_acc = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for data, labels in valid_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                running_valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                _, labels_max = torch.max(labels, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels_max).sum().item()

        valid_loss = running_valid_loss / total_valid
        valid_acc = 100. * correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.5f}%")
        logger.info(f"Epoch {epoch+1}/{epochs} - Valid Loss: {valid_loss:.5f}, Valid Accuracy: {valid_acc:.5f}%")

        # Manually adjust the learning rate based on gamma and step_size
        epochs_since_lr_drop += 1
        if epochs_since_lr_drop >= step_size:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
            logger.info(f"Learning rate reduced manually by gamma={gamma} with step_size={step_size}. New learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            epochs_since_lr_drop = 0  # Reset the counter

        # Save checkpoint at the end of each epoch
        if cfg['checkpointing_enabled']:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_accuracies': train_accuracies,
                'valid_accuracies': valid_accuracies,
                'lr_scheduler': lr_scheduler,
            }
            save_checkpoint(state, checkpoint_dir, checkpoint_filename)
            logger.info(f"Checkpoint saved at {checkpoint_dir}/{checkpoint_filename}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies
            
            
def train_and_validate_eeg_manual_lr_grid_search(
    model, train_loader, valid_loader, total_epochs, optimizer, criterion, 
    device, checkpoint_dir, logger, gamma_values, step_size_values, new_checkpoint):

    checkpoint_filename = "eeg_checkpoint_grid_search.pth.tar"
    seed_everything()
    logger.info(f"Inside training loop")
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler = load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer,new_checkpoint)

    # Initialize variables for manual learning rate scheduling
    best_valid_loss = float('inf')
    lr_drop_epoch = None  # Epoch when validation loss drops for the first time
    grid_search_started = False
    pre_lr_drop_state = None  # To store model state before grid search starts

    # Save initial model and optimizer state for grid search
    initial_model_state = model.state_dict()
    initial_optimizer_state = optimizer.state_dict()

    # Grid search parameters
    total_combinations = len(gamma_values) * len(step_size_values)
    current_gamma, current_step_size = gamma_values[0], step_size_values[0]
    combination_idx = 0

    # Variables to store the best combination and performance
    best_combination = None
    best_combination_train_losses = []
    best_combination_valid_losses = []
    best_combination_train_accuracies = []
    best_combination_valid_accuracies = []

    # Loop through epochs
    for epoch in range(total_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{total_epochs}")
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        logger.info(f"Training....")
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            labels = labels.argmax(dim=1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Calculate average train loss and accuracy
        logger.info(f"Calculating training metrics")
        train_loss = running_train_loss / total_train
        train_acc = 100. * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # log training results
        logger.info(f"Epoch {epoch+1}/{total_epochs} - Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.5f}%")
        
        
        # Validation phase
        model.eval()
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        logger.info(f"Validating")
        with torch.no_grad():
            for data, labels in valid_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                running_valid_loss += loss.item() * data.size(0)

                _, predicted = torch.max(outputs, 1)
                _, labels_max = torch.max(labels, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels_max).sum().item()

        valid_loss = running_valid_loss / total_valid
        valid_acc = 100. * correct_valid / total_valid
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        # validation log results
        logger.info(f"Epoch {epoch+1}/{total_epochs} - Valid Loss: {valid_loss:.5f}, Valid Accuracy: {valid_acc:.5f}%")

        # Check if validation loss has increased compared to best validation loss
        if valid_loss > best_valid_loss:
            # Validation loss increased, start grid search or decay learning rate
            if not grid_search_started:
                grid_search_started = True
                pre_lr_drop_state = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch
                }
                logger.info(f"Validation loss increased at epoch {epoch+1}. Starting grid search...")
                continue  # Skip learning rate decay in this epoch
        else:
            # Validation loss decreased or remained the same
            best_valid_loss = valid_loss

        # Perform grid search when validation loss increases
        if grid_search_started:
            # Check if we've exhausted all combinations
            if combination_idx < total_combinations:
                # Calculate gamma and step_size for this combination
                current_gamma = gamma_values[combination_idx // len(step_size_values)]
                current_step_size = step_size_values[combination_idx % len(step_size_values)]

                # Adjust learning rate if the step size condition is met
                if (epoch - pre_lr_drop_state['epoch']) % current_step_size == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= current_gamma
                    logger.info(f"Applied manual learning rate decay with gamma={current_gamma} and step-size = {current_step_size}. New LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Move to the next combination after current_step_size epochs
                if (epoch - pre_lr_drop_state['epoch']) // current_step_size > 0:
                    # Check if this combination is the best so far
                    if valid_acc > max(best_combination_valid_accuracies, default=0):
                        best_combination = {
                            'gamma': current_gamma,
                            'step_size': current_step_size
                        }
                        best_combination_train_losses = train_losses.copy()
                        best_combination_valid_losses = valid_losses.copy()
                        best_combination_train_accuracies = train_accuracies.copy()
                        best_combination_valid_accuracies = valid_accuracies.copy()

                    combination_idx += 1
                    if combination_idx < total_combinations:
                        # Load the pre-drop state to restart with next hyperparameter set
                        model.load_state_dict(pre_lr_drop_state['model_state'])
                        optimizer.load_state_dict(pre_lr_drop_state['optimizer_state'])
                        logger.info(f"Switching to next grid point: gamma={gamma_values[combination_idx // len(step_size_values)]}, step_size={step_size_values[combination_idx % len(step_size_values)]}")

        if cfg['checkpointing_enabled']:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_accuracies': train_accuracies,
                'valid_accuracies': valid_accuracies,
                'lr_scheduler': lr_scheduler,
            }
            save_checkpoint(state, checkpoint_dir, checkpoint_filename)
            logger.info(f"Checkpoint saved at {checkpoint_dir}/{checkpoint_filename}")

    # Save the best combination details after training
    logger.info(f"Best combination: gamma={best_combination['gamma']}, step_size={best_combination['step_size']}")
    logger.info(f"Best combination training losses: {best_combination_train_losses}")
    logger.info(f"Best combination validation losses: {best_combination_valid_losses}")
    logger.info(f"Best combination training accuracies: {best_combination_train_accuracies}")
    logger.info(f"Best combination validation accuracies: {best_combination_valid_accuracies}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies




    # # After training, plot the metrics
    # #Plot precision
    # plot_metrics(train_precisions, valid_precisions, 'Precision')
    # #plot Recall
    # plot_metrics(train_recalls, valid_recalls, 'Recall')
    # # plot F1 score
    # plot_metrics(train_f1s, valid_f1s, 'F1 Score')
    # # Plot learning rate decay and regularization loss
    # plot_learning_rate_and_regularization(lr_scheduler, regularization_losses)
    # # Plot accuracies
    # plot_accuracies(train_accuracies, valid_accuracies)
    # # Generate the confusion matrix
    # create_confusion_matrix(model, valid_loader, classes=cfg['classes'], device=cfg['device'], checkpoint_dir=checkpoint_dir, checkpoint_filename=checkpoint_filename, save_dir=cfg['save_dir'])

    