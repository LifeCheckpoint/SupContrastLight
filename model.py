import pytorch_lightning as L
import torch
from networks.resnet import *
from losses import *
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

USE_PRETRAINED_MODEL = True
PRETRAINED_MODEL = "50"
FEAT_DIM = 256

class SimCLR2(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SupConResNet(feat_dim=FEAT_DIM, use_pretrained_model=USE_PRETRAINED_MODEL, pretrained_model=PRETRAINED_MODEL)
        self.example_input_array = torch.zeros(1, 3, 256, 256)
    
    def forward(self, x):
        return self.model(x, USE_PRETRAINED_MODEL)

    def training_step(self, batch, idx):
        images, labels = batch
        images = torch.cat([images[0], images[1]], dim=0)
        batch_size = labels.shape[0]

        features = self(images)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)

        sup_loss = SupConLoss(temperature=self.args.temp)
        loss = sup_loss(features)
        self.log("trainning_loss", loss, logger=True, enable_graph=True)
        return loss
    
    def set_args(self, opt):
        self.args = opt

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs - self.args.warm_epochs,
            eta_min=self.args.learning_rate * (self.args.lr_decay_rate ** 3)
        )

        def warmup_lambda(epoch):
            if epoch < self.args.warm_epochs:
                return self.args.warmup_from / self.args.learning_rate + epoch / (self.args.warm_epochs / (self.args.learning_rate - self.args.warmup_from))
            else:
                return 1.0
            
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        
        return [optimizer], [warmup_scheduler, cosine_scheduler]