from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter


class ICSRTrainer(object):
    def __init__(self, encoder, memory=None, memory_c = None):
        super(ICSRTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_c = memory_c

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, data_loader_cam = None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            loss = self.memory(f_out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
    
    def _parse_data_cam(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cids.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

