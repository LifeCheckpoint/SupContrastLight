import pytorch_lightning as L
import torch
from networks.resnet import *
from losses import *
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

USE_PRETRAINED_MODEL = True
PRETRAINED_MODEL = "50"
FEAT_DIM = 384

class SimCLR2(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SupConResNet(feat_dim=FEAT_DIM, use_pretrained_model=USE_PRETRAINED_MODEL, pretrained_model=PRETRAINED_MODEL)
    
    def forward(self, x):
        return self.model(x, USE_PRETRAINED_MODEL)

    def training_step(self, batch, idx):
        images, labels = batch
        images = torch.cat([images[0], images[1]], dim=0)
        batch_size = labels.shape[0]

        features = self(images)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        loss = SupConLoss(features)
        return loss
    
    def set_args(self, opt):
        self.args = opt

    def training_epoch_end(self, outs):
        loss = 0
        for out in outs:
            loss += out["loss"].detach().item()
        loss /= len(outs)

        self.history["loss"].append(loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs - self.args.warm_epochs,
            eta_min=self.args.learning_rate * (self.args.lr_decay_rate ** 3)
        )

        def warmup_lambda(epoch):
            if epoch < self.args.warm_epochs:
                return self.args.warmup_from / self.args.learning_rate + epoch / (self.args.warm_epochs / (self.args.warmup_to - self.args.warmup_from))
            else:
                return 1.0
            
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        
        return [optimizer], [warmup_scheduler, cosine_scheduler]

def save_model(model, save_file):
    print('==> Saving...')
    torch.save(model, save_file)