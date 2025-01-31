
from src.utils import Dataset
import torch
import types
import os


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

os.environ['RWKV_FLOAT_MODE'] = "fp16"

model_type = 'RWKV'
datafile = "data/pop_k_dataset.txt"
datafile_encoding = 'utf-8'

n_layer = 12
n_embd = 768
ctx_len = 2048
batch_size = 4
lr_init = 8e-06
lr_final = 8e-06
n_epoch = 1
num_workers = 1
betas = (0.90, 0.95)
eps = 1e-8


EPOCH_BEGIN = 0
LOAD_MODEL = False

if LOAD_MODEL and EPOCH_BEGIN > 0:
    warmup_tokens = 0
else:
    warmup_tokens = 100 * ctx_len * batch_size


tokens_per_iter = batch_size * ctx_len
print(f"tokens per iteration: {tokens_per_iter:,}")

epoch_length_fixed = (500 // batch_size) * batch_size
epoch_save_frequency = 4
epoch_save_path = ""

os.environ['RWKV_LOAD_MODEL'] = str(LOAD_MODEL)
MODEL_NAME = epoch_save_path + str(EPOCH_BEGIN)


train_dataset = Dataset(open(datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed, batch_size)

if __name__ == '__main__':
    from src.trainer import Trainer, TrainerConfig

    print('\nmodel:', model_type, 'precision:', os.environ.get('RWKV_FLOAT_MODE'), 
        'epoch:', n_epoch, 'batch size:', batch_size, 'betas:', betas, 
        'eps:', eps, 'ctx:', ctx_len, 'layer:', n_layer, 'embd:', n_embd, '\n')

    tconf = TrainerConfig(
        model_type=model_type, 
        max_epochs=n_epoch, 
        batch_size=batch_size,
        learning_rate=lr_init, 
        lr_decay=True, 
        lr_final=lr_final, 
        betas=betas, 
        eps=eps,
        warmup_tokens=warmup_tokens, 
        final_tokens=n_epoch * len(train_dataset) * ctx_len, 
        num_workers=num_workers, 
        epoch_save_frequency=epoch_save_frequency, 
        epoch_save_path=epoch_save_path
    )

    m_cfg = types.SimpleNamespace()
    m_cfg.model_type = model_type
    m_cfg.n_layer = n_layer
    m_cfg.n_embd = n_embd
    m_cfg.EPOCH_BEGIN = EPOCH_BEGIN
    m_cfg.LOAD_MODEL = LOAD_MODEL
    m_cfg.MODEL_NAME = MODEL_NAME

    trainer = Trainer(devices=1, accelerator="gpu", precision="16-mixed")
    trainer.run(m_cfg, train_dataset, None, tconf)