from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from main import img_transform, datasets_params
from dataset import SliceDataset, infer_in_channels
from utils import probs2class, save_images, tqdm_
from functools import partial
from main import img_transform, gt_transform, datasets_params

# val part of main.py but on test, no gt
dataset = "SEGTHOR_full_split42"
SUBSET = "val"
dataDir = Path("splits/SEGTHOR_full_split42")
runDir = Path("results/final/seed42")
predDir = runDir / f"predict_{SUBSET}"

'''
dataDir = f"./data/{dataset}"
runDir = Path("./experiments/final_ceDiceBoundary_200ep_16kernels/split42_seed0/ceDiceBoundary")
predDir = runDir / "test"
'''
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"picked {device} to run predictions")

params = datasets_params[dataset]
K = params['K']
net = params['net'](infer_in_channels(dataDir), K, kernels=params['kernels'], factor=params['factor'])
net.load_state_dict(torch.load(runDir / "bestweights.pt", map_location=device))
net.to(device)
# needed or batchnorm/dropout act like training
net.eval()

testSet = SliceDataset(SUBSET, dataDir, img_transform=img_transform,
                       gt_transform=partial(gt_transform, K), augmentation="none")
testLoader = DataLoader(testSet, batch_size=params['B'], num_workers=5, shuffle=False)

#testSet = SliceDataset('test', dataDir, img_transform=img_transform, augmentation="none")
#testLoader = DataLoader(testSet, batch_size=params['B'], num_workers=12, shuffle=False)

with torch.no_grad():
    for data in tqdm_(testLoader):
        img = data['images'].to(device)
        predProbs = F.softmax(net(img), dim=1)
        predictedClass = probs2class(predProbs)
        # *63 same as main.py, stitch.py divides it back
        save_images(predictedClass * 63, data['stems'], predDir)

print(f"!saved {len(testSet)} predictions to {predDir}")
