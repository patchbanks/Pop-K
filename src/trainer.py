from lightning_fabric import Fabric
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm.auto import tqdm
import datetime
import math
import gc
import os

USE_WANDB = 1

project = "POPK"
name = "POPK"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class TrainerConfig:
    batch_size = 8
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    warmup_tokens = 0
    final_tokens = 0
    epoch_save_frequency = 0
    epoch_save_path = os.getcwd()
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

from src.model import GPT, GPTConfig


class Trainer(Fabric):
    def run(self, m_cfg, train_dataset, val_dataset, config):
        self.cuda_id = int(str(self.device).strip('cuda:'))
        model = GPT(
            GPTConfig(
                train_dataset.vocab_size, 
                train_dataset.ctx_len, 
                model_type=m_cfg.model_type,
                n_layer=m_cfg.n_layer, 
                n_embd=m_cfg.n_embd
            )
        )

        with torch.no_grad():
            if m_cfg.LOAD_MODEL:
                m2 = torch.load(m_cfg.MODEL_NAME + '.pth', map_location='cpu', weights_only=True)
                model.load_state_dict(m2)
                del m2
        model.to(self.device)

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.avg_loss = -1
        self.EPOCH_BEGIN = m_cfg.EPOCH_BEGIN

        self.steps = self.EPOCH_BEGIN * (len(self.train_dataset) // (config.batch_size))

        if self.cuda_id == 0:
            log_file = open("log.txt", "a")
            if USE_WANDB:
                print('logging to wandb... (comment it if you don\'t have wandb)')
                import wandb
                cfg = model.config
                for k in config.__dict__:
                    setattr(cfg, k, config.__dict__[k])
                wandb.init(project=project, name=name, config=cfg, save_code=False)

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        model, optimizer = self.setup(model, optimizer)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.val_dataset
            data.idx_begin = self.steps * config.batch_size + 1
            data.cuda_id = self.cuda_id
            
            if config.num_workers > 0:
                loader = DataLoader(data, shuffle=False, pin_memory=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
            else:
                loader = DataLoader(data, shuffle=False,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(
                loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)
            loader = self.setup_dataloaders(loader)
            gc.collect()
            torch.cuda.empty_cache()
            
            for it, (x, y) in pbar:
                with torch.set_grad_enabled(is_train):
                    loss = model(x, y)

                all_loss = [loss.clone()]

                if is_train:
                    model.zero_grad()
                    self.backward(loss)

                    optimizer.step()

                    self.tokens += (y >= 0).sum()
                    lr_final_factor = config.lr_final / config.learning_rate
                    if self.tokens < config.warmup_tokens:
                        lr_mult = lr_final_factor + \
                            (1 - lr_final_factor) * float(self.tokens) / \
                            float(config.warmup_tokens)
                        progress = 0
                    else:
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        if progress >= 1:
                            lr_mult = lr_final_factor
                        else:
                            lr_mult = math.exp(math.log(lr_final_factor) * pow(progress, 1))
                    lr = config.learning_rate * lr_mult

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    self.lr = lr
                    self.steps += 1
                    
                    now_loss = all_loss[0].item()
            
                    if USE_WANDB and self.cuda_id == 0:
                        wandb.log({"loss": now_loss}, step = self.steps)

                    if self.avg_loss < 0:
                        self.avg_loss = now_loss
                    else:
                        factor = 1 / (it + 1)
                        self.avg_loss = self.avg_loss * (1.0 - factor) + now_loss * factor

                    pbar.set_description(f"miniE {epoch+1+self.EPOCH_BEGIN} s {self.steps} prog {progress*100.0:.2f}% : ppl {math.exp(self.avg_loss):.6f} loss {self.avg_loss:.6f} lr {lr:e}")

        self.tokens = 0
        for epoch in range(99999999):

            run_epoch('train')
            if math.isnan(self.avg_loss):
                exit(0)

            if self.cuda_id == 0:
                log_file.write(f'{epoch+1+self.EPOCH_BEGIN} {self.avg_loss:.6f} {math.exp(self.avg_loss):.4f} {self.lr:.8f} {datetime.datetime.now()} {epoch+1} \n')
                log_file.flush()
            
                if (self.config.epoch_save_frequency > 0 and epoch % self.config.epoch_save_frequency == 0) or (epoch == config.max_epochs - 1):
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    torch.save(raw_model.state_dict(), self.config.epoch_save_path + '.pth')
                    print("model saved")