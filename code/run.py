import glob
import os
import argparse
import shutil

import imageio
import scipy
import tqdm
import numpy as np
import torch
from PIL import Image
import logging

import warnings
import pandas as pd
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
from progressbar import *
from torchvision import transforms
import sys

sys.path.append("..")

from M_DI2_FGSM import M_DI2_FGSM_Attacker
from utils import load_model

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Running script', add_help=False)
parser.add_argument('--input_dir', default='../input_dir', type=str)
parser.add_argument('--output_dir', default='../output_dir', type=str)
parser.add_argument('--img_size', default=299, type=int, help='pic size')
parser.add_argument('--batch_size', '-b', default=5, type=int, help='mini-batch size')
parser.add_argument("--use_gpu", type=bool, default=True)
parser.add_argument("--m_di2_fgsm_attack1", type=bool, default=True)
parser.add_argument("--tmp_out_path", type=str, default='./tmp')
parser.add_argument("--models1", type=str, default='Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg_hrn_jpeg')
parser.add_argument("--m_di2_fgsm_attack2", type=bool, default=True)
parser.add_argument("--models2", type=str, default='Ensemble_dsn161_jpeg_rn162_jpeg_wrn101_jpeg')

args = parser.parse_args()
logging.info(args)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and args.use_gpu) else 'cpu')
logging.info("use device is "+ str(DEVICE))

def load_data(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    transformer = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        b = Image.fromarray(imageio.imread(image_path))
        image = self.transformer(b)

        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample
logging.info("step: 1/2 starting")
if args.m_di2_fgsm_attack1:
    data_loader = load_data(os.path.join(args.input_dir,'dev.csv'), os.path.join(args.input_dir,'images'))['dev_data']

    model = load_model(args.models1).to(DEVICE).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    m_di2_fgsm_attacker = M_DI2_FGSM_Attacker(steps=70,
                                              max_norm=10 / 255.0,
                                              div_prob=0.9,
                                              low_bound=224,
                                              momentum = 1,
                                              device=DEVICE)
    for batch_data in tqdm.tqdm(data_loader):
        clean_images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                          batch_data['filename']

        adv = m_di2_fgsm_attacker.attack(model, clean_images, labels)
        out_path = args.tmp_out_path

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for t in range(clean_images.shape[0]):
            name = filenames[t]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out =os.path.join(out_path, name)
            adv_img = np.transpose(adv[t].detach().cpu().numpy(), (1, 2, 0))
            adv_img = scipy.misc.imresize(adv_img, size=(500, 500))
            scipy.misc.imsave(out, adv_img)

logging.info("step: 2/2 starting")
if args.m_di2_fgsm_attack2:
    data_loader = load_data(os.path.join(args.input_dir,'dev.csv'), args.tmp_out_path)['dev_data']
    data_loader_noresize = load_data(os.path.join(args.input_dir,'dev.csv'), args.tmp_out_path, img_size=500)['dev_data']
    model = load_model(args.models2).to(DEVICE).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    m_di2_fgsm_attacker = M_DI2_FGSM_Attacker(steps=10,
                                              max_norm=10 / 255.0,
                                              div_prob=1.1,
                                              low_bound=270,
                                              momentum=0.9,
                                              return_delta=True,
                                              device=DEVICE)
    for batch_data, batch_data_noresize in tqdm.tqdm(zip(data_loader, data_loader_noresize)):
        clean_images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                          batch_data['filename']
        clean_images_noresizze = batch_data_noresize['image'].to(DEVICE)
        delta = m_di2_fgsm_attacker.attack(model, clean_images, labels)
        out_path = args.output_dir

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for t in range(clean_images.shape[0]):
            name = filenames[t]
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, name)

            clean_image_single = np.transpose(clean_images_noresizze[t].detach().cpu().numpy(), (1, 2, 0))
            clean_image_single = scipy.misc.imresize(clean_image_single, size=(500, 500))

            adv_img = np.transpose(delta[t].detach().cpu().numpy(), (1, 2, 0))
            adv_img = scipy.misc.imresize(adv_img, size=(500, 500), interp='bicubic')
            adimg = adv_img * 0.05 + clean_image_single * 0.95
            scipy.misc.imsave(out, adimg)

shutil.rmtree(args.tmp_out_path)
logging.info("attack finished")