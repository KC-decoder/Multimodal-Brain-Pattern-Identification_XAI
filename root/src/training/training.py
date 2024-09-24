import torch
from utils.data_utils import seed_everything, load_checkpoint, save_checkpoint, plot_metrics, calculate_metrics, plot_learning_rate_and_regularization, plot_accuracies, create_confusion_matrix

# Load the config file using the config_loader module
from utils.config_loader import load_config


cfg = load_config()

def train_and_validate_eeg(model, train_loader, valid_loader, epochs, optimizer, criterion, scheduler, device, checkpoint_dir, logger,new_checkpoint):
    checkpoint_filename = "eeg_checkpoint.pth.tar"
    seed_everything()
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler = load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer,new_checkpoint)
    # Lists to store metrics over epochs
    train_precisions, valid_precisions = [], []
    train_recalls, valid_recalls = [], []
    train_f1s, valid_f1s = [], []
    
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting Epoch {epoch+1}/{epochs}")
        
        
        model.train()
        running_train_loss = 0.0
        running_reg_loss = 0.0  # Track regularization loss
        correct_train = 0
        total_train = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            
            # Combine the loss and optimize (without adding reg_loss here)
            loss.backward()
            optimizer.step()

            # Accumulate losses and accuracy for the current batch
            running_train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class index
            labels = labels.argmax(dim=1)  # Convert one-hot encoded labels to class indices
            correct_train += (predicted == labels).sum().item()  # Now compare the predicted indices with actual class indices
            # You need this line to accumulate the number of samples
            total_train += labels.size(0)  # Update the total number of training samples

        # Calculate average loss and accuracy for the epoch
        train_loss = running_train_loss / total_train
        train_acc = 100. * correct_train / total_train
        
        
        
        

        train_losses.append(train_loss)
        #regularization_losses.append(reg_loss_avg)  # Save reg_loss for plotting
        train_accuracies.append(train_acc)
        
        train_precision, train_recall, train_f1 = calculate_metrics(model, train_loader, device)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        # with torch.no_grad():
        #     # Optionally log reg_loss (for analysis purposes, not for training)
        #     reg_loss = sum(torch.sum(param ** 2) for param in model.parameters())
        #     running_reg_loss += reg_loss.item()  # Accumulate the regularization loss for logging
        #     reg_loss_avg = running_reg_loss/total_train  # Average regularization loss
            

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
        valid_precision, valid_recall, valid_f1 = calculate_metrics(model, valid_loader, device)
        valid_precisions.append(valid_precision)
        valid_recalls.append(valid_recall)
        valid_f1s.append(valid_f1)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Train Accuracy: {train_acc:.5f}%")
        logger.info(f"Epoch {epoch+1}/{epochs} - Valid Loss: {valid_loss:.5f}, Valid Accuracy: {valid_acc:.5f}%")

        # Step the scheduler
        # Print learning rate before and after step() to check if it's changing
        logger.info(f"Learning Rate before step: {scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
        logger.info(f"Learning Rate after step: {scheduler.get_last_lr()[0]:.6f}")
        lr_scheduler.append(scheduler.get_last_lr()[0])

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

    return train_losses, valid_losses, train_accuracies, valid_accuracies