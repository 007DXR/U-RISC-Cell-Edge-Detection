from utils import URISC
from models import UNet, ResNetUNet, CASENet, DDS, DFF
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision import transforms
from tqdm import tqdm
import os
import torch
import numpy as np
from PIL import Image
import cv2
from utils.options import Options


def test_simple(image, model):
    with torch.no_grad():
        predicted = model.module.tta_eval(image)
        predicted = predicted * 255.0
        mask = predicted.cpu().squeeze().detach().numpy().astype(np.uint8)
    return mask


def test_complex(image, model, split_size=4000):
    h, w = image.shape[2], image.shape[3]
    result = np.zeros((h, w), dtype=np.uint8)
    with torch.no_grad():
        for i in range(h // split_size + 1):
            for j in range(w // split_size + 1):
                endi, endj = (i + 1) * split_size, (j + 1) * split_size
                patch = image[:, :, i * split_size: endi, j * split_size: endj]
                patch = patch.contiguous()
                predicted = model.module.tta_eval(patch)
                predicted = predicted * 255.0
                mask = predicted.cpu().squeeze().detach().numpy()
                mask = np.round(mask).astype(np.uint8)
                result[i * split_size: endi, j * split_size: endj] = mask
    return result


if __name__ == "__main__":
    args = Options().parse()
    if "simple" in str.lower(args.dataset):
        dataset = "simple"
    else:
        assert "complex" in str.lower(args.dataset)
        dataset = "complex"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.520, std=0.185),
    ])

    testset = URISC(dir="data/datasets", mode="test", transform=transform, data_rank=dataset)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    if args.model == "UNet":
        model = DataParallel(UNet(backbone=args.backbone)).cpu()
    elif args.model == "ResNetUNet":
        model = ResNetUNet(backbone=args.backbone).cpu()
    elif args.model == "CASENet":
        model = DataParallel(CASENet(backbone=args.backbone)).cpu()
    elif args.model == "DDS":
        model = DataParallel(DDS(backbone=args.backbone)).cpu()
    elif args.model == "DFF":
        model = DataParallel(DFF(backbone=args.backbone)).cpu()
    else:
        print("Not a proper model name")
        exit(0)

    model_dir = os.path.join("data/models", dataset,  args.model)
    model_names = os.listdir(model_dir)
    model_names = [model_name for model_name in model_names if model_name.startswith("best")]

    for model_name in model_names:
        print("Loading model:", args.model, model_name)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=torch.device('cpu') ), strict=False)
        model.eval()

        score = model_name.replace("best_", "").replace(".pth", "").replace("_base", "")
        outdir = os.path.join("outdir", dataset, "masks", args.model)
        if not os.path.exists(os.path.join(outdir, score)):
            os.makedirs(os.path.join(outdir, score))
        outdir = os.path.join(outdir, score)
        tbar = tqdm(testloader)

        for image,filename in tbar:
            # image = image.cuda()
            print(filename,image)
            if dataset == "simple":
                result = test_simple(image, model)
            else:
                result = test_complex(image, model, args.split_size)
            filename = filename[0][filename[0].rfind("/")+1:]
            cv2.imwrite(os.path.join(outdir, filename), result)

    # ensemble and binarize predicted map
    num_of_models = len(model_names)
    print("\nEnsembling %i models...\n" % num_of_models)
    mask_dir = os.path.join("outdir", dataset, "masks")
    if not os.path.exists(os.path.join(mask_dir, "ensemble")):
        os.mkdir(os.path.join(mask_dir, "ensemble"))
    scores = os.listdir(mask_dir)
    scores = [score for score in scores if score.startswith("0")]

    filenames = os.listdir(os.path.join(mask_dir, scores[0]))
    for filename in filenames:
        img = None
        for score in scores:
            cur_img = np.array(Image.open(os.path.join(mask_dir, score, filename))).astype(np.float32)
            if img is None:
                img = cur_img
            else:
                img += cur_img
        img[img < 255 * num_of_models / 2] = 0
        img[img >= 255 * num_of_models / 2] = 255
        img = img.astype(np.uint8)
        cv2.imwrite(os.path.join(mask_dir, "ensemble", filename), img)


