import logging
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from time import time

from . import utils
from .initializer import Initializer


class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        num_top1, num_sample = 0, 0
        train_iter = tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _, obj_name) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)
            y = y.long().to(self.device)

            # Calculating Output
            out, _ = self.model(x)

            # Updating Weights
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar(
                    'learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar(
                    'train_loss', loss.item(), self.global_step)
                
            train_iter.set_description(
                'Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Showing Train Results
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar(
                'train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%})'.format(
            epoch + 1, self.max_epoch, num_top1, num_sample, train_acc
        ))
        logging.info('')

    def eval(self, save_score=True):
        self.model.eval()
        start_eval_time = time()
        score = {}
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, name, obj_name) in enumerate(eval_iter):

                # Using GPU
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Calculating Output
                out, _ = self.model(x)

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                if save_score:
                    for n, c in zip(name, out.detach().cpu().numpy()):
                        score[n] = c

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out, min(5, self.num_class))[1]
                num_top5 += sum([y[n] in reco_top5[n, :]
                                for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        
        # MPCA
        acc_perclass = [0] * self.num_class
        for i in range(self.num_class):
            num_c = sum(cm[i])
            acc_perclass[i] = cm[i, i] / (num_c if num_c > 0 else 1e-6)
        mpca = sum(acc_perclass) / self.num_class
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * \
            self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), MPCA: {:.2%}, Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, mpca, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar(
                'eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar(
                'eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()

        return acc_top1, acc_top5, cm, score

    def run_fold(self, fold_idx):
        """Executes full training/eval sequence for a single fold."""
        # Update directories and loaders for current fold
        self.save_dir = os.path.join(self.args.work_dir, f'fold_{fold_idx}')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Re-initialize dataloaders, model, optimizer, scheduler for fold
        if hasattr(self, 'init_fold_environment'):
            self.init_fold_environment(fold_idx)

        start_time = time()
        start_epoch = 0
        best_state = {
            'acc_top1': 0, 
            'acc_top5': 0,
            'acc_top1_last': 0,
            'cm': 0, 
            'best_epoch': 0
        }
        best_score = {}

        if self.args.resume:
            logging.info(f'Loading fold {fold_idx} checkpoint ...')
            checkpoint = utils.load_checkpoint(self.save_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)

        logging.info(f'--- Starting Fold {fold_idx + 1}/10 ---')
        for epoch in range(start_epoch, self.max_epoch):
            self.train(epoch)

            is_best = False
            if (epoch + 1) % self.eval_interval(epoch) == 0:
                logging.info(f'Evaluating fold {fold_idx + 1} for epoch {epoch + 1}/{self.max_epoch} ...')
                acc_top1, acc_top5, cm, score = self.eval()
                
                if acc_top1 > best_state['acc_top1']:
                    is_best = True
                    best_state.update({
                        'acc_top1': acc_top1, 
                        'acc_top5': acc_top5, 
                        'cm': cm, 
                        'best_epoch': epoch + 1
                    })
                    best_score = score
                best_state.update({'acc_top1_last': acc_top1})

                # Save checkpoint in fold-specific directory
                utils.save_checkpoint(
                    self.model.module.state_dict(), 
                    self.optimizer.state_dict(), 
                    self.scheduler.state_dict(),
                    epoch + 1, 
                    best_state, 
                    is_best, 
                    self.args.work_dir, 
                    self.save_dir, 
                    self.model_name
                )
                logging.info('Fold {} Best top-1 accuracy: {:.2%}@{}th epoch'.format(
                    fold_idx + 1, best_state['acc_top1'], best_state['best_epoch']
                ))

        np.savetxt(f'{self.save_dir}/cm.csv', best_state['cm'], fmt="%s", delimiter=",")
        with open(f'{self.save_dir}/score.pkl', 'wb') as f:
            pickle.dump(best_score, f)

        return best_state['acc_top1'], best_state['acc_top5']

    def start(self):
        n_folds = getattr(self.args, 'n_splits', 10)

        if self.args.evaluate:
            logging.info('Starting evaluation across folds ...')
            fold_accs = []
            for fold_idx in range(n_folds):
                self.save_dir = os.path.join(self.args.work_dir, f'fold_{fold_idx}')
                if hasattr(self, 'init_fold_environment'):
                    self.init_fold_environment(fold_idx)
                
                checkpoint = utils.load_checkpoint(self.save_dir, self.model_name)
                if checkpoint:
                    self.model.module.load_state_dict(checkpoint['model'])
                acc_top1, _, _, _ = self.eval()
                fold_accs.append(acc_top1)
            
            logging.info(f'10-Fold Mean Evaluation Top-1 Accuracy: {np.mean(fold_accs):.2%} ± {np.std(fold_accs):.2%}')

        else:
            fold_top1_accs = []
            fold_top5_accs = []

            for fold_idx in range(n_folds):
                # 1. Re-initialize model weights and move to device
                self.init_model()

                # 2. Reset optimizer state
                self.init_optimizer()

                # 3. Reset learning rate scheduler state
                self.init_lr_scheduler()

                top1, top5 = self.run_fold(fold_idx)
                fold_top1_accs.append(top1)
                fold_top5_accs.append(top5)

            # Log overall Cross-Validation statistics
            logging.info('================ Cross Validation Complete ================')
            logging.info(f'Top-1 Accuracy across folds: {[round(a * 100, 2) for a in fold_top1_accs]}')
            logging.info(f'Mean Top-1 Accuracy: {np.mean(fold_top1_accs):.2%} ± {np.std(fold_top1_accs):.2%}')
            logging.info(f'Mean Top-5 Accuracy: {np.mean(fold_top5_accs):.2%} ± {np.std(fold_top5_accs):.2%}')
            logging.info('===========================================================')