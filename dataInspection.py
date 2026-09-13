from pathlib import Path

import numpy as np
import nibabel as nib


dataPath = Path("data/segthor_part1/train")

allSpacings = []
allShapes = []
badPatients = []
header = f"{' patient   '}{' shape   '}{' spacing mm   '}{' extent mm   '}{'orient.  '}{'ct type.  '}{'gt type.  '}{'shape ok.  '}{'affine ok.  '}"
print(header)
print("-" * len(header))

for folder in sorted(dataPath.iterdir()):
    patientId = folder.name

    # load header for info
    ctImage = nib.load(str(folder / f"{patientId}.nii.gz"))
    gtImage = nib.load(str(folder / "GT.nii.gz"))
    shape = ctImage.shape
    spacing = ctImage.header.get_zooms()

    # aff2axcodes reduces 4by4 to letters. for LPS this means +i goes to the patient's
    # Left, +j to Posterior, +k to Superior. Numbers shown in order.
    orientation = "".join(nib.aff2axcodes(ctImage.affine))

    ctType = str(ctImage.header.get_data_dtype())
    gtType = str(gtImage.header.get_data_dtype())

    # do the two images have same shape? (maybe unnescessary)
    shapeMatch = ctImage.shape == gtImage.shape

    #byte identical
    affineMatch = np.array_equal(ctImage.affine, gtImage.affine)

    #The two images ct and ground truth dont match (just for sure)
    if not shapeMatch or not affineMatch:
        badPatients.append(patientId)

    allSpacings.append(spacing)
    allShapes.append(shape)

    shapeText = f"{shape[0]}x{shape[1]}x{shape[2]}"
    spacingText = f"{spacing[0]:.3f} / {spacing[1]:.3f} / {spacing[2]:.3f}"
    extentText = f"{shape[0] * spacing[0]:.0f} x {shape[2] * spacing[2]:.0f}"

    line = f"{patientId}; {shapeText}; {spacingText}; {extentText};"
    line += f"{orientation}; {ctType}; {gtType}; {str(shapeMatch)}; {str(affineMatch)};"
    print(line)


print("\nmore info summary\n")

print(f"patients               {len(allShapes)}")

totalSlices = 0
for shape in allShapes:
    totalSlices += shape[2]
print(f"total axial slices     {totalSlices}")

inPlaneValues = []
throughPlaneValues = []
sliceCounts = []
for spacing in allSpacings:
    inPlaneValues.append(spacing[0])
    throughPlaneValues.append(spacing[2])
for shape in allShapes:
    sliceCounts.append(shape[2])

uniqueInPlane = sorted(set(round(float(value), 3) for value in inPlaneValues))
uniqueThroughPlane = sorted(set(round(float(value), 2) for value in throughPlaneValues))

print(f"in plane spacing       {min(inPlaneValues):.2f} to {max(inPlaneValues):.2f} mm, {len(uniqueInPlane)} distinct: {uniqueInPlane}")
print(f"slice thickness        {min(throughPlaneValues):.2f} to {max(throughPlaneValues):.2f} mm, {len(uniqueThroughPlane)} distinct: {uniqueThroughPlane}")
print(f"slices per patient     {min(sliceCounts)} to {max(sliceCounts)}")

smallestFieldOfView = min(inPlaneValues) * 512
largestFieldOfView = max(inPlaneValues) * 512
scaleSpread = largestFieldOfView / smallestFieldOfView
print(f"field of view          {smallestFieldOfView:.0f} to {largestFieldOfView:.0f} mm wide")
#somehow different sizes, not handy for organ scales.....
print(len(badPatients))

#Above shows that ct and gt align. but some patients have a larger area captured. Seen by the diff in spacing in z direction. 
#more spacing therefore larger area captured. probably due to larger patients.Will keep in mind...
# seen in the x/y spacing, 1.27 and 1.37 mm against 0.98 for most. <- bigger
#Also pixel size differ thus some pat have larger/smaller pixel size thus contianing more/less info


print('--looking at hu values in voxels--')

header = "patient; huMin; huMax; p1; p50; p99; voxelsOverBone; metal"
print(header)
print("-" * len(header))

metalPatients = []

for folder in sorted(dataPath.iterdir()):
    patientId = folder.name

    ctImage = nib.load(str(folder / f"{patientId}.nii.gz"))
    ct = np.asarray(ctImage.dataobj)

    huMin = int(ct.min())
    huMax = int(ct.max())

    # see what 1 percent, 50 percentile and 99 to see what we are looking at (later maybe metal +3000 ish)
    p1, p50, p99 = np.percentile(ct, [1, 50, 99])

    # bone stops around 2000 HU so anything past 3000 is not tissue.
    voxelsOverBone = int((ct > 3000).sum())

    metal = voxelsOverBone > 0
    if metal:
        metalPatients.append(patientId)

    line = f"{patientId}; {huMin}; {huMax}; {p1:.0f}; {p50:.0f}; {p99:.0f}; "
    line += f"{voxelsOverBone}; {metal}"
    print(line)


print("\n--summary--\n")

print(f"patients with metal    {len(metalPatients)} of 20 (any voxel over 3000 HU)")
print(f"                       {metalPatients}")



print("\n --huvalues per class---\n")

import matplotlib.pyplot as plt

classNames = {0: "background", 1: "aorta", 2: "heart", 3: "trachea", 4: "esophagus"}
classColors = {0: "#8A96A3", 1: "#D6413F", 2: "#2D7FC2", 3: "#A8720C", 4: "#7B4FA8"}

# random sample for run times
sampleSize = 50000
rng = np.random.default_rng(0)

classCounts = {}
classValues = {}
for classId in classNames:
    classCounts[classId] = 0
    classValues[classId] = []

totalVoxels = 0

for folder in sorted(dataPath.iterdir()):
    patientId = folder.name

    ct = np.asarray(nib.load(str(folder / f"{patientId}.nii.gz")).dataobj)
    gt = np.asarray(nib.load(str(folder / "GT.nii.gz")).dataobj)
    totalVoxels += ct.size

    for classId in classNames:
        values = ct[gt == classId]
        classCounts[classId] += len(values)

        if len(values) > sampleSize:
            picks = rng.choice(len(values), size=sampleSize, replace=False)
            values = values[picks]
        if len(values) > 0:
            classValues[classId].append(values)


header = "class; voxels; percent of all; huMin; huMax; p5; p95"
print(header)
print("-" * len(header))

for classId in classNames:
    count = classCounts[classId]
    percent = 100 * count / totalVoxels

    if count == 0:
        print(f"{classId} {classNames[classId]}; 0; 0.0000%; -; -; -; -")
        continue

    values = np.concatenate(classValues[classId])
    p5, p95 = np.percentile(values, [5, 95])
    line = f"{classId} {classNames[classId]}; {count}; {percent:.2f}%; "
    line += f"{int(values.min())}; {int(values.max())}; {p5:.0f}; {p95:.0f}"
    print(line)


# scale to own values as background will be a lot
plt.figure(figsize=(12, 5))

for classId in classNames:
    if classCounts[classId] == 0:
        plt.plot([], [], color=classColors[classId], linewidth=2,
                 label=f"{classNames[classId]} - 0 voxels, not in this data")
        continue

    values = np.concatenate(classValues[classId])
    counts, edges = np.histogram(values, range=(-1100, 1000))
    centers = (edges[:-1] + edges[1:]) / 2
    counts = counts / counts.max()

    plt.fill_between(centers, counts, color=classColors[classId])
    plt.plot(centers, counts, color=classColors[classId],
             label=f"{classNames[classId]} - {classCounts[classId]} voxels")

plt.xlabel("HU")
plt.ylabel("relative frequency")
plt.title("plot of HU")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("huPerClass.png")
print("\nsaved huPerClass.png")


print("\n2D slices!!!\n")

header = "patient; slices; empty; empty percent; aortaSlices; heartSlices; tracheaSlices"
print(header)
print("-" * len(header))

totalSlices = 0
totalEmpty = 0
totalAorta = 0
totalHeart = 0
totalTrachea = 0

for folder in sorted(dataPath.iterdir()):
    patientId = folder.name

    gt = np.asarray(nib.load(str(folder / "GT.nii.gz")).dataobj)
    numSlices = gt.shape[2]
    hasAorta = (gt == 1).any(axis=(0, 1))
    hasHeart = (gt == 2).any(axis=(0, 1))
    hasTrachea = (gt == 3).any(axis=(0, 1))
    #We do or as some slices can contain more than one
    hasSomething = hasAorta | hasHeart | hasTrachea
    emptyCount = numSlices - int(hasSomething.sum())

    totalSlices += numSlices
    totalEmpty += emptyCount
    totalAorta += int(hasAorta.sum())
    totalHeart += int(hasHeart.sum())
    totalTrachea += int(hasTrachea.sum())

    line = f"{patientId}; {numSlices}; {emptyCount}; {100 * emptyCount / numSlices:.0f}%; "
    line += f"{int(hasAorta.sum())}; {int(hasHeart.sum())}; {int(hasTrachea.sum())}"
    print(line)


print("\nslice summaryy\n")

print(f"total slices           {totalSlices}")
print(f"empty slices           {totalEmpty} ({100 * totalEmpty / totalSlices:.1f}%)")
print(f"aorta slices           {totalAorta} ({100 * totalAorta / totalSlices:.1f}%)")
print(f"heart slices           {totalHeart} ({100 * totalHeart / totalSlices:.1f}%)")
print(f"trachea slices         {totalTrachea} ({100 * totalTrachea / totalSlices:.1f}%)")
