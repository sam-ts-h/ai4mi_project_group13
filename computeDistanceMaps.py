import pickle
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

#We downsample but check if same after kieran.....
ORIGINAL_SHAPE: int = 512


def loadPatientLabels(gtFolder, patientId) -> np.ndarray:
    slicePaths = sorted(gtFolder.glob(f"{patientId}_*.png"),key=lambda p: int(p.stem.split('_')[-1]))
    slices: list[np.ndarray] = []
    for path in slicePaths:
        labels = np.array(Image.open(path)) // 63
        slices.append(labels)
    return np.stack(slices, axis=-1).astype(np.uint8)

#BE AWARE: Spacing is used as is here please see changes by kieran later, maybe spacing not right...
def signedDistance(mask, sampling) -> np.ndarray:
    outside = ~mask.astype(bool)
    #below method looks at the dist from the (1) to nearest not (0) so border
    distOutside = distance_transform_edt(outside, sampling=sampling)
    distInside = distance_transform_edt(mask, sampling=sampling)
    return distOutside * outside - distInside * mask


def processPatient(patientId, gtFolder, destFolder, K, spacingDict) -> int:
    labels = loadPatientLabels(gtFolder, patientId)
    dx, dy, dz = spacingDict[patientId]
    scale = ORIGINAL_SHAPE / labels.shape[0]
    sampling = (scale * dx, scale * dy, dz)

    distMaps = np.zeros((K, *labels.shape), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        # If no borders dont run code, errors
        if mask.any() and not mask.all():
            distMaps[k] = signedDistance(mask, sampling)
    for idz in range(labels.shape[-1]):
        np.save(destFolder / f"{patientId}_{idz:04d}.npy",distMaps[:, :, :, idz].astype(np.float16))
    return labels.shape[-1]


def main(args):
    dataPath = Path(args.data_dir)
    #load spacings, subject to change....
    with open(dataPath / "spacing.pkl", 'rb') as f:
        spacingDict = pickle.load(f)

    for subset in ['train', 'val']:
        gtFolder = dataPath / subset / 'gt'
        destFolder = dataPath / subset / 'distmap'
        #parents should exist but to be sure (defensive)
        destFolder.mkdir(parents=True, exist_ok=True)
        uniqueIds = set()
        for path in gtFolder.glob("*.png"):
            #right split so last part away
            uniqueIds.add(path.stem.rsplit('_', 1)[0])
        patientIds = sorted(uniqueIds)
        print(f"Computing distance maps for {len(patientIds)}")

        counts = []
        for i, patientId in enumerate(patientIds):
            print(f"{i}/{len(patientIds)}")
            counts.append(processPatient(patientId, gtFolder, destFolder, args.num_classes, spacingDict))

        print(f"rote {sum(counts)} distance maps to {destFolder}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Signed distance maps for the boundary loss')
    parser.add_argument('--data_dir', type=str, default='data/SEGTHOR')
    parser.add_argument('--num_classes', type=int, default=5)
    args = parser.parse_args()
    print(args)
    return args
if __name__ == "__main__":
    main(get_args())
