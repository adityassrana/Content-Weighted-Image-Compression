import glob
from argparse import ArgumentParser
from typing import Dict

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

from models import Decoder, Encoder

from pydantic import BaseModel
import yaml


class ImageDataset(Dataset):
    """Dataset class for creating data pipeline"""

    def __init__(self, glob_pattern, patchsize):
        self.image_paths, self.patchsize = glob.glob(glob_pattern), patchsize

    def __len__(self): return len(self.image_paths)

    def transform(self, image):
        if image.mode == 'L':
            image = image.convert('RGB')
        self.data_transforms = transforms.Compose([transforms.RandomCrop(size=self.patchsize),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomVerticalFlip(),
                                                   transforms.ToTensor()])
        return self.data_transforms(image)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = self.transform(image)
        return image


class Config(BaseModel):
    train_files: str
    val_files: str
    patch_size: int
    lr: float
    batch_size: int
    num_epochs: int


class AE(pl.core.LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss_func = nn.MSELoss()

        self.config: Config = config

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_func(x, x_hat)
        return loss

    def training_epoch_end(self, training_step_outputs):
        # [results, results, results]
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        avg_psnr = -10*torch.log10(avg_loss)
        self.logger.experiment.add_scalar('loss/avg_train', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('psnr/train', avg_psnr, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = self.loss_func(x, x_hat)
        return {'val_loss': loss, 'x': x, 'x_hat': x_hat}

    def validation_epoch_end(self, validation_step_outputs):
        # [results, results, results]
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        avg_psnr = -10*torch.log10(avg_loss)
        x = validation_step_outputs[-1]['x']
        x_hat = validation_step_outputs[-1]['x_hat']
        x_grid = make_grid(x, 4)
        x_hat_grid = make_grid(x_hat, 4)

        self.logger.experiment.add_image('val/original', x_grid, self.current_epoch)
        self.logger.experiment.add_image('val/reconstructions', x_hat_grid, self.current_epoch)
        self.logger.experiment.add_scalar('loss/avg_val', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('psnr/val', avg_psnr, self.current_epoch)

    def train_dataloader(self):
        train_ds = ImageDataset(self.config.train_files, self.config.patch_size)
        return DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        valid_ds = ImageDataset(self.config.val_files, self.config.patch_size)
        return DataLoader(valid_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path", default="config.yml")
    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        config_dict: Dict = yaml.safe_load(stream)
        config: Config = Config.parse_obj(config_dict)

    model = AE(config)
    trainer = pl.Trainer(gpus=1,
                         max_epochs=config.num_epochs,
                         check_val_every_n_epoch=1)
    trainer.fit(model)
