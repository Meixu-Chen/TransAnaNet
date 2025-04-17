# Python Modules
import numpy as np
import time
import os
import math


# Torch Modules
import torch
import torch.nn.functional as F

# Self Modules
from lib.utils.general import prepare_input
from lib.utils.logger import log


# from lib.visual3D_temp.BaseWriter import TensorboardWriter


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion_pre, criterion_gtv, criterion_smooth, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion_pre = criterion_pre
        self.criterion_gtv = criterion_gtv
        self.criterion_smooth = criterion_smooth
        # self.metric = metric
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # self.log_step = int(np.sqrt(train_data_loader.batch_size))
        # self.writer = TensorboardWriter(args)

        self.save_frequency = 1
        # self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1
        self.val_loss = 0

        self.print_batch_spacing = 10
        self.acculumation_steps = 1
        #self.acculumation_steps = 5
        self.record_val_loss = 1000

    def training(self):
        # self.model.train()


        for epoch in range(self.start_epoch, self.args.nEpochs):

            for param_group in self.optimizer.param_groups:
                lr_show = param_group['lr']
            print('\n########################################################################')
            print(f"Training epoch: {epoch}, Learning rate: {lr_show:.8f}")
            print(f"\nTraining epoch: {epoch}, Learning rate: {lr_show:.8f}")

            self.train_epoch_alex(epoch)

            if self.do_validation:
                print(f"Validation epoch: {epoch}")
                print(f"\nValidation epoch: {epoch}")
                self.validate_epoch_alex(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # # comment out for speed test
            if epoch % self.save_frequency == 0:
                self.model.save_checkpoint(self.args.save,
                                       epoch, self.val_loss,
                                       optimizer=self.optimizer)
            print('\n\n')

    def train_epoch_alex(self, epoch):

        # Creates once at the beginning of training
        # scaler = torch.cuda.amp.GradScaler()

        def time_report(initial_time, time_name):
            get_time_diff = time.gmtime(time.time() - initial_time)
            readable_time = time.strftime("%M:%S", get_time_diff)
            del get_time_diff
            del readable_time

        epoch_start_time = time.time()
        self.model.train()
        time_report(epoch_start_time, 'model.train()')

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        sim_loss_cum = []
        dice_loss_cum = []
        dice2_loss_cum = []
        smooth_loss_cum = []

        print('-------------------------------------------------------------------------------------------')

        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            batch_timer = time.time()
            time_report(epoch_start_time, 'enter batch loop')


            # Gathering input data; prepare_input sends to gpu
            img, gtv = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor = torch.cat((img[:,0:3], gtv[:,0:4]), dim=1)
            target_img = img[:,3:4]
            target_gtv = gtv[:,4:5]
            target_gtvn = gtv[:,5:6]
            time_report(batch_timer, 'input manipulation')
            input_tensor.requires_grad = True
            time_report(batch_timer, 'grad equals True')
            # Model make prediction
            pred_img, pred_gtv, pred_gtvn, dvf = self.model(input_tensor)
            time_report(batch_timer, 'input and model prediction')


            # calculating loss and metrics
            # with torch.cuda.amp.autocast():
            loss_sim = self.criterion_pre(pred_img, target_img)/self.acculumation_steps
            loss_dice = self.criterion_gtv(pred_gtv, target_gtv)/self.acculumation_steps
            loss_dice2 = self.criterion_gtv(pred_gtvn, target_gtvn)/self.acculumation_steps
            loss_smooth = self.criterion_smooth(dvf)/self.acculumation_steps

            loss = loss_sim + loss_dice + loss_dice2 + loss_smooth*self.args.lamda

            time_report(batch_timer, 'loss calculation')

            # need to calculate gradient
            self.model.zero_grad()

            loss.backward()
            time_report(batch_timer, 'gradient calculation')

            def clip_gradient(optimizer, grad_clip):
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)


            clip_gradient(self.optimizer, 5)
            self.optimizer.step()

            time_report(batch_timer, 'grad clip and optim step')


            #Calculating and appending
            with torch.no_grad():
                # storing loss and metrics
                loss_cum.append(loss.item())
                sim_loss_cum.append(loss_sim.item())
                dice_loss_cum.append(loss_dice.item())
                dice2_loss_cum.append(loss_dice2.item())
                smooth_loss_cum.append(loss_smooth.item())
                time_report(batch_timer, 'store loss')


                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    print('\t**************************************************************************')
                    print(f"\tBatch {batch_idx + 1} of {len(self.train_data_loader)}")
                    print(f"\tLoss: {loss.item()}, SmoothLoss: {loss_smooth.item()}, SimilarityLoss: {loss_sim.item()}, DiceLoss: {loss_dice.item()}, Dice2Loss: {loss_dice2.item()}")
                    print('\t**************************************************************************')
                else:
                    pass

            time_report(batch_timer, 'finish one batch')

        # Calculating time per epoch
        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        print(f"Summary-----Loss: {sum(loss_cum) / len(loss_cum)}, SimlarityLoss: {sum(sim_loss_cum) / len(sim_loss_cum)}, DiceLoss: {sum(dice_loss_cum) / len(dice_loss_cum)}, Dice2Loss: {sum(dice2_loss_cum) / len(dice2_loss_cum)}, SmoothLoss: {sum(smooth_loss_cum) / len(smooth_loss_cum)}")

        print(f"\nSummary-----Loss: {sum(loss_cum) / len(loss_cum)}")
        print(f"\nSummary-----SimlarityLoss: {sum(sim_loss_cum) / len(loss_cum)}")
        print(f"\nSummary-----DiceLoss: {sum(dice_loss_cum) / len(loss_cum)}")
        print(f"\nSummary-----Dice2Loss: {sum(dice2_loss_cum) / len(loss_cum)}")
        print(f"\nSummary-----SmoothLoss: {sum(smooth_loss_cum) / len(loss_cum)}")
        print('-------------------------------------------------------------------------------------------')

    def validate_epoch_alex(self, epoch):
        self.model.eval()

        # Storing epoch values obtained from batch calculations
        loss_cum = []
        sim_loss_cum = []
        dice_loss_cum = []
        dice2_loss_cum = []
        smooth_loss_cum = []

        # starting epoch timer
        epoch_start_time = time.time()

        print('-------------------------------------------------------------------------------------------')
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            

            if (batch_idx + 1) % self.print_batch_spacing == 0:
                print('*************************************')
                print(f"\tBatch {batch_idx + 1} of {len(self.valid_data_loader)}")
            else:
                pass

            with torch.no_grad():
                # Gathering input data; prepare_input sends to gpu
                img, gtv = prepare_input(input_tuple=input_tuple, args=self.args)
                # input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor = torch.cat((img[:,0:3], gtv[:,0:4]), dim=1)
                target_img = img[:,3:4]
                target_gtv = gtv[:,4:5]
                target_gtvn = gtv[:,5:6]

                input_tensor.requires_grad = False
                # Model make prediction
                pred_img, pred_gtv, pred_gtvn, dvf = self.model(input_tensor)

                # calculating loss and metrics
                # with torch.cuda.amp.autocast():
                loss_sim = self.criterion_pre(pred_img, target_img)/self.acculumation_steps
                loss_dice = self.criterion_gtv(pred_gtv, target_gtv)/self.acculumation_steps
                loss_dice2 = self.criterion_gtv(pred_gtvn, target_gtvn)/self.acculumation_steps
                loss_smooth = self.criterion_smooth(dvf)/self.acculumation_steps


                loss = loss_sim + loss_dice + loss_dice2 + loss_smooth*self.args.lamda

                # #plot_val_images

                # if self.record_val_loss>=self.val_loss:
                #     self.record_val_loss=self.val_loss
                # for checking
                if (epoch+1)%10==0:
                    pred_img = pred_img.cpu().detach().numpy()
                    pred_gtv = pred_gtv.cpu().detach().numpy()
                    pred_gtvn = pred_gtvn.cpu().detach().numpy()
                    dvf = dvf.cpu().detach().numpy()

                    if not os.path.exists("Output/pred_val/"):
                        os.makedirs("Output/pred_val/")

                    np.save("Output/pred_val/pred_img_"+str(batch_idx)+".npy", pred_img)
                    np.save("Output/pred_val/pred_gtvp_"+str(batch_idx)+".npy", pred_gtv)
                    np.save("Output/pred_val/pred_gtvn_"+str(batch_idx)+".npy", pred_gtvn)
                    np.save("Output/pred_val/dvf_"+str(batch_idx)+".npy", dvf)
                    
                    # print('*************************************')                
                

                if (batch_idx + 1) % self.print_batch_spacing == 0:
                    print(f"\tLoss: {loss.item()}")
                else:
                    pass

                # storing loss and metrics
                loss_cum.append(loss.item())
                sim_loss_cum.append(loss_sim.item())
                dice_loss_cum.append(loss_dice.item())
                dice2_loss_cum.append(loss_dice2.item())
                smooth_loss_cum.append(loss_smooth.item())

        self.val_loss = sum(loss_cum) / len(loss_cum)
        if self.record_val_loss>self.val_loss:
            self.record_val_loss=self.val_loss
            for batch_idx, input_tuple in enumerate(self.valid_data_loader):
                with torch.no_grad():
                    # Gathering input data; prepare_input sends to gpu
                    img, gtv = prepare_input(input_tuple=input_tuple, args=self.args)

                    input_tensor = torch.cat((img[:,0:3], gtv[:,0:4]), dim=1)
                    target_img = img[:,3:4]
                    target_gtv = gtv[:,4:5]
                    target_gtvn = gtv[:,5:6]

                    input_tensor.requires_grad = False
                    # Model make prediction
                    pred_img, pred_gtv, pred_gtvn, dvf = self.model(input_tensor)

                    pred_img = pred_img.cpu().detach().numpy()
                    pred_gtv = pred_gtv.cpu().detach().numpy()
                    pred_gtvn = pred_gtvn.cpu().detach().numpy()
                    dvf = dvf.cpu().detach().numpy()

                    if not os.path.exists("Output/pred_val_better/"):
                        os.makedirs("Output/pred_val_better/")

                    np.save("Output/pred_val_better/pred_img_"+str(batch_idx)+".npy", pred_img)
                    np.save("Output/pred_val_better/pred_gtvp_"+str(batch_idx)+".npy", pred_gtv)
                    np.save("Output/pred_val_better/pred_gtvn_"+str(batch_idx)+".npy", pred_gtvn)
                    np.save("Output/pred_val_better/dvf_"+str(batch_idx)+".npy", dvf)

        ty_res = time.gmtime(time.time() - epoch_start_time)
        res = time.strftime("%M:%S", ty_res)
        print(f"Validation epoch completed in {res} (min:seconds)")
        print(f"Summary-----Loss: {sum(loss_cum) / len(loss_cum)}, SimlarityLoss: {sum(sim_loss_cum) / len(sim_loss_cum)}, DiceLoss: {sum(dice_loss_cum) / len(dice_loss_cum)}, Dice2Loss: {sum(dice2_loss_cum) / len(dice2_loss_cum)}, SmoothLoss: {sum(smooth_loss_cum) / len(smooth_loss_cum)}")
        print('-------------------------------------------------------------------------------------------')
