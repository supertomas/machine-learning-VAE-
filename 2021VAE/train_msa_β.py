import tensorflow as tf

import argparse
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint

from models.vaes_β import MSAVAE
from utils.io import load_gzdata
from utils.io import load_data
from utils.data_loaders import one_hot_generator

# Define training parameters
batch_size = 32
seed = 23645
n_epochs = 14
verbose = 1
save_all_epochs = False

seed = np.random.seed(seed)

# Load aligned sequences
_, msa_seqs = load_data('data/training_data/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_ver2.a2m', one_hot=False)
_, val_msa_seqs = load_data('data/training_data/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_ver2.a2m', one_hot=False)

# Define data generators
train_gen = one_hot_generator(msa_seqs, padding=None)
val_gen = one_hot_generator(val_msa_seqs, padding=None)

# Define model
print('Building model')
model = MSAVAE(original_dim=253, latent_dim=10)

# (Optionally) define callbacks
callbacks=[CSVLogger('output/logs/msavae.csv')]

if save_all_epochs:
    callbacks.append(ModelCheckpoint('output/weights/msavae'+'.{epoch:02d}-{luxa_errors_mean:.2f}.hdf5',
                                     save_best_only=False, verbose=1))

print('Training model')
# Train model https://github.com/keras-team/keras/issues/8595
model.VAE.fit_generator(generator=train_gen,
                        steps_per_epoch=len(msa_seqs) // batch_size,
                        verbose=verbose,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_msa_seqs) // batch_size,
                        callbacks=callbacks)

if not save_all_epochs:
  model.save_weights('output/weights/msavae_BLAT_seed.h5')
