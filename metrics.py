import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

nrClasses = 5
classNames = ["background", "aorta", "heart", "trachea", "esophagus"]
metricNames = ["dsc", "hd95", "assd", "fpSliceRate"]


def dice(GT, Prediction):
    overlap = np.logical_and(GT, Prediction).sum()
    return 2 * overlap / (GT.sum() + Prediction.sum())


def surfaceDistances(GT, Prediction, spacing):
    """Distances in mm from each surface to the other one, both directions in one array."""
    GT = GT.astype(bool)
    Prediction = Prediction.astype(bool)

    # erosion takes the boundarie (voxel that has all other voxels in mask) then we subtract it so we get boundary
    GTSurface = GT & ~binary_erosion(GT)
    PredictionSurface = Prediction & ~binary_erosion(Prediction)

    # argwhere wfinds pos in 3d list, then we is tuple of dimensions... (check after puck&kieran are done if still correct)
    GTPoints = np.argwhere(GTSurface) * np.asarray(spacing)
    PredictionPoints = np.argwhere(PredictionSurface) * np.asarray(spacing)

    #cdktree is a way to get fast nearest points (split search space in half often to get smaller), then we search for all closest boundary points
    toGT, _ = cKDTree(GTPoints).query(PredictionPoints)
    toPrediction, _ = cKDTree(PredictionPoints).query(GTPoints)

    return np.concatenate([toGT, toPrediction])


def falsePositiveSliceRate(GT, Prediction):
    emptySlices = 0
    falsePositives = 0
    #sliceje = z axis
    for sliceje in range(GT.shape[2]):
        if not GT[:, :, sliceje].any():
            emptySlices += 1
            if Prediction[:, :, sliceje].any():
                falsePositives += 1

    if emptySlices == 0:
        return np.nan
    return falsePositives / emptySlices


def main(args):
    args.dest_folder.mkdir(exist_ok=True)
    predictedPatientFiles = sorted(args.pred_folder.glob("*.nii.gz"))
    assert len(predictedPatientFiles) > 0, 'no predictions found.....'

    scores = {}
    for name in metricNames:
        scores[name] = {}

    for predPath in predictedPatientFiles:
        patientId = predPath.name.replace(".nii.gz", "")

        gtNib = nib.load(str(args.gt_folder / patientId / "GT.nii.gz"))
        gt = np.asarray(gtNib.dataobj)
        predNib = nib.load(str(predPath))
        pred = np.asarray(predNib.dataobj)
        spacing = gtNib.header.get_zooms()[:3]

        assert pred.shape == gt.shape, 'stopped because shapes not same between gt and pred'
        assert np.allclose(predNib.affine, gtNib.affine), "stopped because affine not same between gt and pred"

        patient = {}
        for name in metricNames:
            patient[name] = np.full(nrClasses, np.nan)

        for k in range(nrClasses):
            gtMask = gt == k
            predMask = pred == k

            if not gtMask.any():
                continue

            patient["dsc"][k] = dice(gtMask, predMask)
            patient["fpSliceRate"][k] = falsePositiveSliceRate(gtMask, predMask)

            if predMask.any():
                distances = surfaceDistances(gtMask, predMask, spacing)
                patient["hd95"][k] = np.percentile(distances, 95)
                patient["assd"][k] = distances.mean()

        for name in metricNames:
            scores[name][patientId] = patient[name]

        print(f"{patientId}: dsc {np.nanmean(patient['dsc'][1:]):.2f}, "
              f"hd95 {np.nanmean(patient['hd95'][1:]):.2f} mm")

    for name in metricNames:
        np.savez(args.dest_folder / f"{name}.npz", **scores[name])

    header = 'metric'.ljust(14)
    for name in classNames + ['combined']:
        header += name.rjust(12)

    print()
    print('mean over patients, combined all foreground')
    print(header)
    print('-' * len(header))

    for name in metricNames:
        #rows is patients
        stacked = np.stack(list(scores[name].values()))
        columns = [stacked[:, k] for k in range(nrClasses)] + [stacked[:, 1:]]

        line = name.ljust(14)
        for values in columns:
            if np.all(np.isnan(values)):
                line += '-'.rjust(12)
            else:
                line += f'{np.nanmean(values):.3f}'.rjust(12)
        print(line)

    print(f'saved to {args.dest_folder}')


def get_args():
    #like other files to be able to use makefile easily (makes exp easier)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_folder', type=Path, required=True)
    parser.add_argument('--gt_folder', type=Path, required=True)
    parser.add_argument('--dest_folder', type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
