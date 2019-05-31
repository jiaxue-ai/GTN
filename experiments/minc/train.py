from __future__ import division
import os.path as osp
import sys
import argparse

import torch
from config import config
from network import GTN
from minc import Dataloder

# global variable
best_pred = 0.0
errlist_train = []
errlist_val = []


def adjust_learning_rate(optimizer, epoch, best_pred):
    lr = config.lr * (0.1 ** max(0, (epoch - 1) // config.lr_decay))
    if (epoch-1) % config.lr_decay == 0:
        print('LR is set to {}'.format(lr))
    if epoch <=20:
        for param_group in optimizer.param_groups[:-9]:
            param_group['lr'] = 0
        for param_group in optimizer.param_groups[-9:]:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main():
    global best_pred, errlist_train, errlist_val

    # seed init
    seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # dataloader and model
    train_loader, test_loader = Dataloder(config).getloader()
    model = GTN(config.num_classes, pretrained=True)

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': model.aux.parameters()},
        {'params': model.efc1.parameters()},
        {'params': model.erelu1.parameters()},
        {'params': model.dp.parameters()},
        {'params': model.efc2.parameters()},
        {'params': model.en2.parameters(), 'lr': config.lr*10},
        {'params': model.dp2.parameters(), 'lr': config.lr*10},
        {'params': model.fc.parameters(), 'lr': config.lr*10},
        ], 
        lr=config.lr, momentum=config.momentum,
        weight_decay=config.weight_decay)

    model.cuda()
    model = torch.nn.DataParallel(model)

    def train(epoch):
        alpha = config.aux_weight
        model.train()
        global best_pred, errlist_train
        train_loss, correct, total = 0,0,0
        adjust_learning_rate(optimizer, epoch, best_pred)
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, aux = model(data)
            loss = criterion(output, target) + alpha * criterion(aux, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.data.max(1)[1] 
            correct += pred.eq(target.data).cpu().sum()
            total += target.size(0)
            acc = 100.*correct/total
            progress_bar(batch_idx, len(train_loader), 
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
                (train_loss/(batch_idx+1), 
                acc, total-correct, total))
        errlist_train += [acc]

    def validate(epoch):
        model.eval()
        global best_pred, errlist_train, errlist_val
        test_loss, correct, total = 0,0,0
        is_best = False

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output, aux = model(data)
                test_loss += criterion(output, target).item()
                # get the index of the max log-probability
                pred = output.data.max(1)[1] 
                correct += pred.eq(target.data).cpu().sum()
                total += target.size(0)

                acc = 100.*correct/total
                progress_bar(batch_idx, len(test_loader), 
                    'Loss: %.3f | Acc: %.3f%% (%d/%d)'% \
                    (test_loss/(batch_idx+1), 
                    acc, total-correct, total))

        if args.eval:
            print('Acc rate is %.3f'%acc)
            return
        # save checkpoint
        errlist_val += [err]
        if err < best_pred:
            best_pred = err 
            is_best = True

            
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'errlist_train':errlist_train,
            'errlist_val':errlist_val,
            }, config=config, is_best=is_best)

    for epoch in range(config.nepochs):
        print('Epoch:', epoch)
        train(epoch)
        test(epoch)


