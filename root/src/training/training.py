import torch
from utils.data_utils import seed_everything, load_checkpoint, save_checkpoint, plot_metrics, calculate_metrics, plot_learning_rate_and_regularization, plot_accuracies

# Load the config file using the config_loader module
from utils.config_loader import load_config

cfg = load_config()

def train_and_validate_eeg(model, train_loader, valid_loader, epochs, optimizer, criterion, scheduler, device, checkpoint_dir, logger):
    checkpoint_filename = "eeg_checkpoint.pth.tar"
    seed_everything()
    start_epoch, train_losses, valid_losses, train_accuracies, valid_accuracies, lr_scheduler, regularization_losses = load_checkpoint(checkpoint_dir, checkpoint_filename, model, optimizer)
    # Lists to store metrics over epochs
    train_precisions, valid_precisions = [], []
    train_recalls, valid_recalls = [], []
    train_f1s, valid_f1s = [], []
    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting Epoch {epoch+1}/{epochs}")
        model.train()
        running_train_loss = 0.0
        running_reg_loss = 0.0  # Initialize running regularization loss
        correct_train = 0
        total_train = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Manually compute regularization loss
            reg_loss = sum(torch.sum(param ** 2) for param in model.parameters())
            reg_loss *= model.weight_decay  # Use weight_decay from the optimizer

            total_loss = loss + reg_loss  # Combine regularization loss with main loss

            total_loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_reg_loss += reg_loss.item()  # Accumulate the regularization loss

            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels_max).sum().item()

        train_loss = running_train_loss / total_train
        reg_loss_avg = running_reg_loss / total_train  # Average regularization loss
        train_acc = 100. * correct_train / total_train
        train_losses.append(train_loss)
        regularization_losses.append(reg_loss_avg)  # Track the regularization loss
        train_accuracies.append(train_acc)
        
        train_precision, train_recall, train_f1 = calculate_metrics(model, train_loader, device)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        

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

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.5f}, Reg Loss: {reg_loss_avg:.5f}, Train Accuracy: {train_acc:.5f}%")
        logger.info(f"Epoch {epoch+1}/{epochs} - Valid Loss: {valid_loss:.5f}, Valid Accuracy: {valid_acc:.5f}%")

        if scheduler:
            scheduler.step(valid_loss)
            logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
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
                'regularization_losses': regularization_losses  # Save the regularization losses
            }
            save_checkpoint(state, checkpoint_dir, checkpoint_filename)
            logger.info(f"Checkpoint saved at {checkpoint_dir}/{checkpoint_filename}")




    # After training, plot the metrics
    #Plot precision
    plot_metrics(train_precisions, valid_precisions, 'Precision')
    #plot Recall
    plot_metrics(train_recalls, valid_recalls, 'Recall')
    # plot F1 score
    plot_metrics(train_f1s, valid_f1s, 'F1 Score')
    # Plot learning rate decay and regularization loss
    plot_learning_rate_and_regularization(lr_scheduler, regularization_losses)
    # Plot accuracies
    plot_accuracies(train_accuracies, valid_accuracies)

    return train_losses, valid_losses, train_accuracies, valid_accuracies