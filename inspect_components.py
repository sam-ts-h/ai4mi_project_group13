from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import label


INPUT_DIR = Path(
    "results/SEGTHOR_full_split42/volumes"
)

LABELS = {
    1: "Esophagus",
    2: "Heart",
    3: "Trachea",
    4: "Aorta",
}

CONNECTIVITY = 1


for path in sorted(INPUT_DIR.glob("*.nii.gz")):

    volume = np.asarray(
        nib.load(str(path)).dataobj
    )

    print()
    print("=" * 70)
    print(path.name)
    print("=" * 70)

    for class_id, class_name in LABELS.items():

        mask = volume == class_id

        labeled = label(
            mask,
            connectivity=CONNECTIVITY
        )

        n_components = labeled.max()

        if n_components == 0:
            print(
                f"{class_name:<10}: not present"
            )
            continue

        sizes = np.bincount(
            labeled.ravel()
        )[1:]

        sizes = np.sort(sizes)[::-1]

        print()
        print(
            f"{class_name:<10}: "
            f"{n_components} components"
        )

        print(
            f"  Total voxels : {sizes.sum():,}"
        )

        print(
            f"  Largest      : {sizes[0]:,}"
        )

        print(
            f"  Top components: "
            f"{sizes[:10].tolist()}"
        )

        for fraction in [
            0.001,
            0.005,
            0.01,
            0.02,
            0.05,
        ]:

            threshold = max(
                50,
                int(sizes[0] * fraction)
            )

            print(
                f"    fraction={fraction:<6} "
                f"threshold={threshold:,}"
            )