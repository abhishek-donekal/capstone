import pandas as pd
import os

import autokeras as ak


path = r'C:\Users\Student\OneDrive - Florida International University\Desktop\Capstone Project\Tanvir data\BRCA-subtypes-combined-tumor-samples-only-gdc-tcga-refined.csv'
BRCA = pd.read_csv(path)
path = r'C:\Users\Student\OneDrive - Florida International University\Desktop\Capstone Project\Tanvir data\GDC_rnaseq_fpkm_unstranded_v4.csv'
GDC = pd.read_csv(path)
path = r'C:\Users\Student\OneDrive - Florida International University\Desktop\Capstone Project\Tanvir data\MUT_combined_v4.csv'
MUT = pd.read_csv(path)
path = r'C:\Users\Student\OneDrive - Florida International University\Desktop\Capstone Project\Tanvir data\cnv_segment_gene_centric_v4.csv'
CNV = pd.read_csv(path)

MUT.head()

for i in range(238):
    MUT[f'padding_{i}'] = 0



from sklearn.preprocessing import LabelEncoder
df=MUT
# Assuming your class labels are in a Series named 'Subtype_mRNA_PANCAN'
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(df['Subtype_mRNA_PANCAN'])
df.drop(df.columns[0], axis=1, inplace=True)
df.drop('Subtype_mRNA_PANCAN', axis=1, inplace=True)

import numpy as np
import matplotlib.pyplot as plt

# Create an empty list to store the ndarray representations of the plots
plots_as_arrays = []

for i in range(969):
    df.loc[i]
    data = df.loc[i].values
    data = np.array(data)
    grid = np.reshape(data, (130, 130))

    # Append the grid (plot) as a NumPy array to the list
    plots_as_arrays.append(grid)

# Save the list of arrays as a NumPy ndarray
plots_as_ndarray = np.array(plots_as_arrays)

# Save the NumPy ndarray to a file
np.save('plots_as_arrays.npy', plots_as_ndarray)

plots_as_ndarray = plots_as_ndarray.astype('float32') / 130
X=plots_as_ndarray
y=integer_labels

import numpy as np

# Assuming 'gray_images' contains your grayscale image data with shape (969, 130, 130)  # Replace with your actual data
X_rgb = np.empty((969, 130, 130, 3))

# Fill all three RGB channels with the grayscale values
X_rgb[..., 0] = X
X_rgb[..., 1] = X
X_rgb[..., 2] = X

# Now, 'rgb_images' contains the grayscale images in RGB format

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node = ak.ImageAugmentation(horizontal_flip=False,vertical_flip=False)(output_node)
output_node = ak.ResNetBlock()(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=10
)
clf.fit(X_train, y_train, epochs=1)

best_model=clf.export_model
y_pred=clf.predict(X_test)
y_pred = np.array(y_pred, dtype=int)
y_pred = y_pred.reshape(y_test.shape)


tuner = clf.tuner
trial_ids = tuner.oracle.trials
trials_hparams=[]
score=[]
for trial_id in trial_ids:
    trial = clf.tuner.oracle.get_trial(trial_id)
    trial.best_step
    trial.metrics
    hparams = trial.hyperparameters.values
    score.append(trial.score)
  # Append to list
    trials_hparams.append(hparams)
df = pd.DataFrame(trials_hparams)
df['val_Loss']=score

# Group the DataFrame by 'image_block_1/block_type' and calculate the mean 'val_loss' within each group
result = df.groupby('image_block_1/block_type')['val_Loss'].mean()
print(result)