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



'''
output:

patient    shape    spacing mm    extent mm   orient.  ct type.  gt type.  shape ok.  affine ok.  
---------------------------------------------------------------------------------------------------
Patient_01; 512x512x229; 0.977 / 0.977 / 2.000; 500 x 458;LPS; int16; uint8; True; True;
Patient_02; 512x512x246; 0.977 / 0.977 / 2.500; 500 x 615;LPS; int32; uint8; True; True;
Patient_03; 512x512x147; 0.977 / 0.977 / 2.500; 500 x 368;LPS; int32; uint8; True; True;
Patient_04; 512x512x158; 0.977 / 0.977 / 2.500; 500 x 395;LPS; int32; uint8; True; True;
Patient_05; 512x512x284; 0.977 / 0.977 / 2.000; 500 x 568;LPS; int16; uint8; True; True;
Patient_06; 512x512x199; 0.977 / 0.977 / 2.500; 500 x 498;LPS; int32; uint8; True; True;
Patient_07; 512x512x179; 0.977 / 0.977 / 2.000; 500 x 358;LPS; int32; uint8; True; True;
Patient_08; 512x512x171; 0.977 / 0.977 / 2.500; 500 x 428;LPS; int32; uint8; True; True;
Patient_09; 512x512x154; 0.977 / 0.977 / 2.500; 500 x 385;LPS; int32; uint8; True; True;
Patient_10; 512x512x163; 0.977 / 0.977 / 2.000; 500 x 326;LPS; int32; uint8; True; True;
Patient_11; 512x512x180; 1.270 / 1.270 / 2.500; 650 x 450;LPS; int32; uint8; True; True;
Patient_12; 512x512x171; 0.977 / 0.977 / 2.000; 500 x 342;LPS; int32; uint8; True; True;
Patient_13; 512x512x228; 0.977 / 0.977 / 2.000; 500 x 456;LPS; int16; uint8; True; True;
Patient_14; 512x512x162; 0.977 / 0.977 / 2.500; 500 x 405;LPS; int32; uint8; True; True;
Patient_15; 512x512x150; 0.896 / 0.896 / 2.500; 459 x 375;LPS; int32; uint8; True; True;
Patient_16; 512x512x166; 0.977 / 0.977 / 2.500; 500 x 415;LPS; int32; uint8; True; True;
Patient_17; 512x512x166; 0.977 / 0.977 / 2.500; 500 x 415;LPS; int32; uint8; True; True;
Patient_18; 512x512x183; 0.977 / 0.977 / 2.500; 500 x 458;LPS; int32; uint8; True; True;
Patient_19; 512x512x190; 0.977 / 0.977 / 2.500; 500 x 475;LPS; int32; uint8; True; True;
Patient_20; 512x512x206; 1.367 / 1.367 / 2.500; 700 x 515;LPS; int32; uint8; True; True;

more info summary

patients               20
total axial slices     3732
in plane spacing       0.90 to 1.37 mm, 4 distinct: [0.896, 0.977, 1.27, 1.367]
slice thickness        2.00 to 2.50 mm, 2 distinct: [2.0, 2.5]
slices per patient     147 to 284
field of view          459 to 700 mm wide
0
--looking at hu values in voxels--
patient; huMin; huMax; p1; p50; p99; voxelsOverBone; metal
----------------------------------------------------------
Patient_01; -1000; 3071; -1000; -994; 245; 538; True
Patient_02; -1000; 26613; -1000; -965; 367; 5051; True
Patient_03; -1000; 25292; -1000; -978; 276; 1063; True
Patient_04; -1000; 4551; -1000; -967; 312; 7; True
Patient_05; -1000; 3071; -1000; -996; 384; 3544; True
Patient_06; -1000; 3071; -1000; -989; 313; 3998; True
Patient_07; -1000; 3059; -1000; -984; 259; 3; True
Patient_08; -1000; 3071; -1000; -993; 317; 2899; True
Patient_09; -1000; 3071; -1000; -994; 366; 3868; True
Patient_10; -1000; 3071; -1000; -949; 390; 1088; True
Patient_11; -1000; 8834; -1000; -988; 114; 417; True
Patient_12; -1000; 3071; -1000; -994; 367; 537; True
Patient_13; -1000; 3071; -1000; -977; 378; 1717; True
Patient_14; -1000; 1892; -1000; -981; 214; 0; False
Patient_15; -1000; 17762; -1000; -970; 345; 877; True
Patient_16; -1000; 11362; -1000; -913; 321; 495; True
Patient_17; -1000; 14614; -1000; -983; 166; 859; True
Patient_18; -1000; 15566; -1000; -959; 246; 3022; True
Patient_19; -1000; 31743; -1000; -974; 330; 2589; True
Patient_20; -1000; 3071; -1000; -996; 182; 340; True

--summary--

patients with metal    19 of 20 (any voxel over 3000 HU)
                       ['Patient_01', 'Patient_02', 'Patient_03', 'Patient_04', 'Patient_05', 'Patient_06', 'Patient_07', 'Patient_08', 'Patient_09', 'Patient_10', 'Patient_11', 'Patient_12', 'Patient_13', 'Patient_15', 'Patient_16', 'Patient_17', 'Patient_18', 'Patient_19', 'Patient_20']

 --huvalues per class---

class; voxels; percent of all; huMin; huMax; p5; p95
----------------------------------------------------
0 background; 968101701; 98.96%; -1000; 17514; -1000; 62
1 aorta; 2357913; 0.24%; -1000; 2264; -14; 203
2 heart; 7496021; 0.77%; -1000; 5909; -81; 167
3 trachea; 365773; 0.04%; -1000; 3071; -1000; -248
4 esophagus; 0; 0.0000%; -; -; -; -

saved huPerClass.png

2D slices!!!

patient; slices; empty; empty percent; aortaSlices; heartSlices; tracheaSlices
------------------------------------------------------------------------------
Patient_01; 229; 97; 42%; 132; 40; 63
Patient_02; 246; 124; 50%; 122; 37; 48
Patient_03; 147; 37; 25%; 110; 27; 52
Patient_04; 158; 39; 25%; 119; 40; 57
Patient_05; 284; 142; 50%; 142; 62; 60
Patient_06; 199; 88; 44%; 111; 37; 49
Patient_07; 179; 40; 22%; 139; 46; 63
Patient_08; 171; 65; 38%; 106; 43; 46
Patient_09; 154; 44; 29%; 108; 40; 54
Patient_10; 163; 37; 23%; 126; 46; 59
Patient_11; 180; 79; 44%; 101; 32; 42
Patient_12; 171; 34; 20%; 137; 45; 65
Patient_13; 228; 95; 42%; 133; 51; 57
Patient_14; 162; 59; 36%; 103; 42; 35
Patient_15; 150; 37; 25%; 113; 47; 50
Patient_16; 166; 60; 36%; 106; 34; 45
Patient_17; 166; 48; 29%; 118; 43; 52
Patient_18; 183; 58; 32%; 125; 48; 49
Patient_19; 190; 72; 38%; 118; 42; 55
Patient_20; 206; 104; 50%; 102; 36; 43

slice summaryy

total slices           3732
empty slices           1359 (36.4%)
aorta slices           2371 (63.5%)
heart slices           838 (22.5%)
trachea slices         1044 (28.0%)

'''
