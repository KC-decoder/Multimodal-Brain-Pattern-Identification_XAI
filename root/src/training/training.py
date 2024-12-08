import torch
from utils.data_utils import seed_everything, load_checkpoint, save_checkpoint, plot_metrics, calculate_metrics, plot_learning_rate_and_regularization, plot_accuracies, create_confusion_matrix, linear_warmup_and_cosine_annealing

# Load the config file using the config_loader module
from utils.config_loader import load_config


cfg = load_config()

import math

def train_and_validate_eeg(model, train_loader, valid_loader, epochs, optimizer, criterion, device, checkpoint_dir, logger, new_checkpoint, initial_lr, peak_lr, warmup_epochs, min_lr):
    # Generate the checkpoint filename
    checkpoint_filename = f"eeg_warmup_cosine_annealing_9.pth.tar"
    
    seed_everything()
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler = load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer, new_checkpoint)

    for epoch in range(start_epoch, epochs):
        
        for param_group in optimizer.param_groups:
            lr =  param_group['lr'] 
        
        logger.info(f"Starting Epoch {epoch+1}/{epochs}, Current Learning Rate: {lr:.8f}")
        
        # Training phase
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
        
        current_lr = lr 
        # Warm-up and Cosine Annealing Learning Rate Adjustment
        if epoch < warmup_epochs:
            current_lr = initial_lr + (peak_lr - initial_lr) * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            current_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
        
        # Apply the updated learning rate to the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        lr_scheduler.append(current_lr)
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Learning Rate after warmup/cosine annealing: {current_lr:.8f}")

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.5f}%")
        logger.info(f"Epoch {epoch+1}/{epochs} - Valid Loss: {valid_loss:.5f}, Valid Accuracy: {valid_acc:.5f}%")

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





def train_spectrogram_model(
    model, train_loader, valid_loader, epochs, optimizer, criterion, device, checkpoint_dir, 
    logger, new_checkpoint, scheduler=None
):
    # Generate the checkpoint filename
    checkpoint_filename = f"spec_ViT_1.pth.tar"
    
    seed_everything()
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler = load_checkpoint(
        checkpoint_dir, checkpoint_filename, model, optimizer, new_checkpoint
    )

    for epoch in range(start_epoch, epochs):
        # Log the initial learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Starting Epoch {epoch+1}/{epochs}, Initial Learning Rate: {current_lr:.8f}")
        
        # Training phase
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

        # Update the learning rate using the scheduler if provided
        if scheduler:
            scheduler.step()

        # Update and log the current learning rate after scheduler adjustment
        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.append(current_lr)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.5f}%")
        logger.info(f"Epoch {epoch+1}/{epochs} - Valid Loss: {valid_loss:.5f}, Valid Accuracy: {valid_acc:.5f}%")
        logger.info(f"Learning Rate after Scheduler Update: {current_lr:.8f}")

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

    