import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from torch import optim
from utils import *
from modules import *
import sde_lib
from solver import *
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s : %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

      
def sample(img_size, sde, model, solver, eps, n, device):
    logging.info(f'Sampling {n} new images')
    model.eval()

    with torch.no_grad(): 
        # Initial sample
        x = sde.prior_sampling((n, 3, img_size, img_size)).to(device)
        timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            vec_t = torch.ones(n, device = t.device) * t          
            score_fn  = get_score_fn(sde, model, train=False)

            if solver.lower() == 'em':
                solve = EulerMaruyamaSolver(sde, score_fn)
            elif solver.lower() == 'milstein':
                solve = MilsteinSolver(sde, score_fn)
            elif solver.lower() == 'srk1':
                solve = SRK1Solver(sde, score_fn)  
            elif solver.lower() == 'sie':
                solve = SIESolver(sde, score_fn)
            elif solver.lower() == 'weakorder2rk':
                solve = WeakOrder2RK(sde, score_fn)
            elif solver.lower() == 'srk2':
                solve = SRK2Solver(sde, score_fn)
            else:
                raise NotImplementedError(f"Solver unknown.")
            
            x = solve.update_fn(x, vec_t)
        
        x = (x.clamp(-1,1) + 1) / 2 # Addition of 1 and division of 2 to bring x into [0,1]
        # Comment this in evaluation of IS, FID, KID
        x = (x * 255).type(torch.uint8) # Bring x to valid pixel range
    
    model.train()    
    
    return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    sde = args.sde
    eps = args.eps
    solver = args.solver
    warmup = args.warmup
    grad_clip = args.grad_clip
    lr = args.lr
    img_size = args.img_size
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    #diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    if args.ckpt != "" :
        checkpoint = torch.load(args['ckpt'])
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.epoch
        loss = checkpoint.loss
    
    # Algorithm 1 in DDPM Paper
    for epoch_ in range(args.epochs):
        logging.info(f'Starting epoch {epoch_} :')
        pbar = tqdm(dataloader)
        for i, (images,_) in enumerate(pbar):
            images = images.to(device)
            score_fn = get_score_fn(sde, model, train=True)
            # Bring t \in [eps, sde.T]
            t = torch.rand(images.shape[0], device=device) * (sde.T - eps) + eps
            z = torch.randn_like(images)
            mean, std = sde.marginal_prob(images, t)
            perturbed_data = mean + std[:, None, None, None] * z
            score = score_fn(perturbed_data, t)
            
            # Multiply with weight function \lambda(t)
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)
          
            optimizer.zero_grad()
            loss.backward()
            if warmup > 0:
              for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(epoch_ / warmup, 1.0)
            if grad_clip >= 0:
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            ema.step_ema(ema_model, model)
            
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch_ * l + 1)
        
        checkpoint = {'epoch' : epoch_, 
                        'model_state_dict' : model.state_dict(),
                        'ema_model_state_dict' : ema_model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss' : loss}
        torch.save(checkpoint, os.path.join("models", args.run_name, f"sde_cifar_ckpt_32.pt"))
        if epoch_ % 5 == 0:
          sampled_images = sample(img_size, sde, model, solver=solver, eps=1e-3, n=images.shape[0], device=device)
          ema_sampled_images = sample(img_size, sde, ema_model, solver=solver, eps=1e-3, n=images.shape[0], device=device)
          save_images(sampled_images, os.path.join("results", args.run_name, f'{epoch_}.jpg'))
          save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch_}_ema.jpg"))

def launch():
    import argparse
    '''For Colab
    Must use args['...'] in Colab 
    
    from imutils.video import VideoStream 
    from imutils.video import FPS 
    import imutils 
    import time 
    import cv2 
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str,help="path to input video file",) 
    parser.add_argument("-t", "--tracker", required=False,type=str, 
                        default="kcf", help="OpenCV object tracker type") 
    parser.add_argument("-f", "--file", required=False) 
    args = vars(parser.parse_args())
    '''
    parser = argparse.ArgumentParser() # comment this in colab
    args = parser.parse_args()         # comment this in colab
    args.run_name = "DDPM_Unconditional"
    args.epochs = 100
    args.batch_size = 5
    args.image_size = 32
    args.dataset_path = r"D:\University\毕业设计\Programs\landscape_img_folder"
    args.device = "cuda"
    args.lr = 2e-4
    args.warmup = 5000
    args.grad_clip = 1
    args.sde = sde_lib.VPSDE()
    args.eps = 1e-5
    args.solver = 'EM' # ['EM', 'SRK1', 'SRK2', 'SIE', 'WeakOrder2RK']
    args.ckpt = ""
    train(args)
    
    
if __name__ == "__main__":
    # Uncomment this to train model
    #launch()
    
    #'''
    device = "cuda"
    model = UNet().to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    mse = nn.MSELoss()
    checkpoint = torch.load(r"unconditional_cifar_32_vpsde_200e.pt")
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    sde = sde_lib.VPSDE()
    x = sample(img_size=32, sde=sde, model=model, solver='em', n=10, eps=1e-3, device=device)
    plot_images(x)
    #'''