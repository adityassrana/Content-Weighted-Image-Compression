import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
from torchvision.utils import make_grid
import glob
from PIL import Image
from models import Encoder,Decoder
from argparse import ArgumentParser

class ImageDataset(Dataset):
    """Dataset class for creating data pipeline"""
    def __init__(self, glob_pattern, patchsize):
        self.image_paths, self.patchsize = glob.glob(glob_pattern), patchsize

    def __len__(self): return len(self.image_paths)

    def transform(self, image):
        if image.mode == 'L':
            image = image.convert('RGB')
        self.data_transforms = transforms.Compose([transforms.RandomCrop(size = self.patchsize),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomVerticalFlip(),
                                                   transforms.ToTensor()])
        return self.data_transforms(image)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image= self.transform(image)
        return image

class AE(pl.core.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss_func = nn.MSELoss()
        #self.train_glob = '/home/adityassrana/datatmp/Datasets/CLIC/*png'
        #self.valid_glob = '/home/adityassrana/datatmp/Datasets/kodak/*.png'
        
    def forward(self,x):
        return self.decoder(self.encoder(x))
    
    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_func(x, x_hat)
        psnr = -10*torch.log10(loss)
        return {'loss':loss,'psnr':psnr}
    
    def training_epoch_end(self,training_step_outputs):
        # [results, results, results]
        avg_psnr = torch.stack([x['psnr'] for x in training_step_outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.logger.experiment.add_scalar('loss/avg_train',avg_loss,self.current_epoch)
        self.logger.experiment.add_scalar('psnr/train',avg_psnr,self.current_epoch)
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_func(x, x_hat)
        psnr = -10*torch.log10(loss)
        return {'val_loss':loss,'psnr':psnr, 'x':x,'x_hat':x_hat}
    
    def validation_epoch_end(self,validation_step_outputs):
        # [results, results, results]
        avg_psnr = torch.stack([x['psnr'] for x in validation_step_outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        x = validation_step_outputs[-1]['x']
        x_hat = validation_step_outputs[-1]['x_hat']
        x_grid = make_grid(x,4)
        x_hat_grid = make_grid(x_hat, 4)
        
        self.logger.experiment.add_image('val/original', x_grid, self.current_epoch)
        self.logger.experiment.add_image('val/reconstructions',x_hat_grid,self.current_epoch)
        self.logger.experiment.add_scalar('loss/avg_val',avg_loss,self.current_epoch)
        self.logger.experiment.add_scalar('psnr/val',avg_psnr,self.current_epoch)
        return {'val_loss':avg_loss}
    
    def train_dataloader(self):
        train_ds = ImageDataset(self.hparams.train_glob, 128)
        return DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        valid_ds = ImageDataset(self.hparams.valid_glob, 128)
        return DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_glob", default = "/home/adityassrana/datatmp/Datasets/CLIC/*png")
    parser.add_argument("--valid_glob", default = "/home/adityassrana/datatmp/Datasets/kodak/*.png")
    args = parser.parse_args()
    
    dict_args = vars(args)
    model = AE(**dict_args)
    trainer = pl.Trainer(gpus=1,max_epochs=5,check_val_every_n_epoch=1)
    trainer.fit(model)