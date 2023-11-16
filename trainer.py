import os
import torch
import torch.utils.tensorboard
import torch.hub

from torch.autograd import Variable


class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, use_gpu, step, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.step = step
        self.logger = logger

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        max_k = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

    # Define the training function:
    def train(self, epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.use_gpu:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)  # On GPU
            data, target = Variable(data), Variable(target)
            _, target_max = torch.max(target.data, 1)
            self.optimizer.zero_grad()
            output = self.model(data)
            target_max = Variable(target_max)
            loss = self.criterion(output, target_max)
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

            # TensorBoard logging
            # Log the scalar values (you may add more parameters to monitor them)
            info = {
                'train_loss': loss.item()
            }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, self.step)

            self.step += 1
        self.model.train()
        return self.step

    # Defining the validation function:
    def validate(self, epoch, learning_rate):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for i, (input_data, target) in enumerate(self.val_loader):
                if self.use_gpu:
                    input_data, target = input_data.cuda(non_blocking=True), target.cuda(non_blocking=True)

                input_var = Variable(input)
                target = Variable(target)

                # compute output
                output = self.model(input_var)

                # measure accuracy and record loss
                _, target_max = torch.max(target.data, 1)
                prec1, prec5 = self.accuracy(output.data, target_max, topk=(1, 2))
                self.update(prec1[0], input_data.size(0))

                # compute loss
                target_max = Variable(target_max)  # Note
                total_val_loss += self.criterion(output, target_max).item()

        avg_loss = total_val_loss / len(self.val_loader)

        # TensorBoard logging
        # Log the scalar values
        info = {
            'val_loss': avg_loss,
            'Top@1_Accuracy': self.avg.cpu(),
            'Learning_Rate': learning_rate
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch)
        return self.avg, avg_loss

    # Defining the snapshot method (for saving the best model & the training state):
    def save_snapshot(self, dir_path, run_name, state):
        snapshot_file = os.path.join(dir_path, run_name + '-model_best.pth')
        torch.save(state, snapshot_file)

    # Define a method for save the last snapshot & last model
    def save_last_snapshot(self, dir_path, run_name, state):
        snapshot_file = os.path.join(dir_path, run_name + '-model_last.pth')
        torch.save(state, snapshot_file)



