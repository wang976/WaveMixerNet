from exp.exp_basic import Exp_Basic
from models.model import WaveMixerNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from data_provider.data_factory import data_provider

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import optuna
from thop import profile
import warnings
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.min_test_loss = np.inf
        self.epoch_for_min_test_loss = 0
        
    def _build_model(self):
        model_dict = {'WaveMixerNet': WaveMixerNet}
        model = model_dict[self.args.model](self.args, self.device).float()  # device(type='cuda', index=0)
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_criterion(self):
        criterion = {'mse': torch.nn.MSELoss(), 'smoothL1': torch.nn.SmoothL1Loss()}
        try:
            return criterion[self.args.loss]
        except KeyError as e:
            raise ValueError(f"Invalid argument: {e} (loss: {self.args.loss})")


    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()        
        preds_mean, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                pred_mean, true = self._process_one_batch(vali_data, batch_x, batch_y, 'vali')
                
                preds_mean.append(pred_mean)
                trues.append(true)

            preds_mean = torch.cat(preds_mean).cpu()
            trues = torch.cat(trues).cpu()
            
            preds_mean = preds_mean.reshape(-1, preds_mean.shape[-2], preds_mean.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            
            mae, mse, rmse, mape, mspe = metric(preds_mean.numpy(), trues.numpy())
            self.model.train()
            return mse, mae


    def train(self, setting, optunaTrialReport = None):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion() 

        if self.args.use_amp:
            scaler =  torch.cuda.amp.GradScaler(init_scale = 1024)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            tracked_cuda_devices = []
            if self.args.use_gpu and torch.cuda.is_available():
                if getattr(self.args, 'use_multi_gpu', False) and getattr(self.args, 'device_ids', None):
                    device_ids = self.args.device_ids
                    if isinstance(device_ids, str):
                        device_ids = [int(dev.strip()) for dev in device_ids.split(',') if dev.strip()]
                    for dev_id in device_ids:
                        tracked_cuda_devices.append(torch.device(f'cuda:{dev_id}'))
                else:
                    tracked_cuda_devices.append(self.device)
                for dev in tracked_cuda_devices:
                    torch.cuda.reset_peak_memory_stats(dev)
            
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, desc = f'Epoch {epoch + 1}', bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt}")):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad(set_to_none = True)
                pred_mean, true = self._process_one_batch(train_data, batch_x, batch_y, 'train')
                loss = criterion(pred_mean, true)                
                train_loss.append(loss) #.item())
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # print("Epoch {}: cost time: {:.2f} sec".format(epoch + 1, time.time()-epoch_time))
            peak_memory_log = ""
            if tracked_cuda_devices:
                peak_memory_bytes = max(torch.cuda.max_memory_allocated(dev) for dev in tracked_cuda_devices)
                peak_memory_log = f" | Peak GPU memory: {peak_memory_bytes / (1024 ** 2):.2f} MB"
            print("Epoch {}: cost time: {:.2f} sec{}".format(epoch + 1, time.time()-epoch_time, peak_memory_log))
            train_loss = torch.tensor(train_loss).mean() # np.average(train_loss)
            vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae = self.vali(test_data, test_loader, criterion)

            if test_loss <  self.min_test_loss:
                self.min_test_loss = test_loss
                self.min_test_mae = test_mae
                self.epoch_for_min_test_loss = epoch            

            ########################### this part is just for optuna ###########
            if optunaTrialReport is not None:
                optunaTrialReport.report(test_loss, epoch)
                if optunaTrialReport.should_prune():
                    raise optuna.exceptions.TrialPruned()
            #############################################################
            
            print("\tEpoch {0}: Steps- {1} | Train Loss: {2:.5f} Vali.MSE: {3:.5f} Vali.MAE: {4:.5f} Test.MSE: {5:.5f} Test.MAE: {6:.5f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae, test_loss, test_mae))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("\tEarly stopping")
                break
            if torch.isnan(train_loss):
                print("\stopping: train-loss-nan")
                break
            adjust_learning_rate(model_optim, None, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        criterion =  self._select_criterion() 
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(test_data, batch_x, batch_y,  'test')
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    true_np = true.detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], pred_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))

            preds = torch.cat(preds).cpu()
            trues = torch.cat(trues).cpu()
            # result save   
            folder_path = './results/' + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
            mae, mse, rmse, mape, mspe = metric(preds.numpy(), trues.numpy())
            print('mse: {}, mae: {}'.format(mse, mae))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()
            return mse, mae

    
    def get_gflops(self):
        batch = self.args.batch_size
        seq = self.args.seq_len
        channel = self.args.c_in
        input_tensor = torch.randn(batch, seq, channel).to('cuda') # Dumy inputs
        
        self.model.eval()
        macs, params = profile(self.model, inputs = (input_tensor, ), verbose = True)
        gflops = 2 * macs / 1e9  # convert to GFLOPs
        print(f"Total GFLOPs: {gflops:.4f}")
        return gflops
    

    def predict(self, setting, load=False):
        raise NotImplementedError("not implemented for uncertainity")
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        return

    def _process_one_batch(self, dataset_object, batch_x, target, function):
        batch_x = batch_x.to(dtype = torch.float, device = self.device)
        target =  target.to(dtype = torch.float, device = self.device)
        
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                pred = self.model(batch_x)
        else:
            pred = self.model(batch_x)
        return pred, target
    
