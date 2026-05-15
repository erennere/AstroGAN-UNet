"""Custom Keras callback for checkpointing, validation, and training logs."""

import os
import logging
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from src.training.utils import (
    save_fits,
    ensure_directory_exists,
    ensure_parent_dir_exists,
    build_checkpoint_filename,
    save_checkpoint_model,
)
from src.training.math_helpers import (
    inverse_min_max_normalization,
    inverse_zscore_normalization,
    inverse_adaptive_log_transform_and_denormalize,
)


class Callback(tf.keras.callbacks.Callback):
    """Track training metrics, checkpoints, and preview artifacts per epoch."""

    def __init__(self, dataset, dataset_size, epoch, start_epoch, eval_save_percentage,
                 ds_save_percentage, save_freq, sample_generator, kwargs_validation,
                validation_dataset, validation_dataset_size, optimizer, change_learning_rate, batch_size, scaling, use_gan,
                training_results_dir, checkpoints_dir, training_metrics_csv_path,
                validation_loss_filename='validation_loss.txt',
                training_metrics_filename='training_metrics.txt',
                config=None):
        """Initialise the custom training callback.

        Parameters
        ----------
        dataset : tf.data.Dataset
            The training dataset.
        dataset_size : int
            Total number of samples in the dataset.
        epoch : int
            Total number of training epochs planned.
        start_epoch : int
            Epoch index from which training resumes.
        eval_save_percentage : float
            Fraction (percent) of validation images used each epoch.
        ds_save_percentage : float
            Fraction (percent) of training samples for which preview images
            are saved.
        save_freq : int
            Epoch interval at which checkpoints and preview images are saved.
        sample_generator : callable
            Generator function used to produce validation samples.
        kwargs_validation : dict
            Keyword arguments forwarded to the validation dataset builder.
        validation_dataset : tf.data.Dataset
            Prebuilt validation dataset.
        validation_dataset_size : int
            Total number of validation samples represented in the dataset.
        optimizer : tf.keras.optimizers.Optimizer
            The optimiser used for training (used for LR scheduling).
        change_learning_rate : list of tuple
            List of ``(epoch, lr)`` pairs; the last applicable entry is used
            to update the learning rate at the start of each epoch.
        batch_size : int
            Training batch size (used to normalise discriminator loss).
        scaling : str or None
            Scaling mode (``'log_min_max'``, ``'z_scale'``, ``'min_max'``, or
            ``None``).
        use_gan : bool
            Whether the model is a GAN (affects loss key extraction).
        training_results_dir : str
            Root directory for training artefacts.
        checkpoints_dir : str
            Directory where model checkpoints are saved.
        training_metrics_csv_path : str
            Path for the per-epoch metrics CSV file.
        validation_loss_filename : str, optional
            Filename for the validation loss log. Default is
            ``'validation_loss.txt'``.
        training_metrics_filename : str, optional
            Filename for the training metrics log. Default is
            ``'training_metrics.txt'``.
        config : dict, optional
            Full runtime config dict (from load_config). If provided, models_dir
            is ensured to exist at training begin. Default is ``None``.
        """
        super().__init__()
        self.dataset = dataset
        self.dataset_size = int(dataset_size)
        self.kwargs_validation = kwargs_validation
        self.eval_save_percentage = eval_save_percentage
        self.ds_save_percentage = ds_save_percentage
        self.save_freq = save_freq
        self.epoch = epoch
        self.current_epoch = start_epoch
        self.best_loss = np.inf
        self.training_metrics = []
        self.start_time = time.time()
        self.sample_generator = sample_generator
        self.validation_dataset = validation_dataset
        self.validation_dataset_size = int(validation_dataset_size)
        self.optimizer = optimizer
        self.change_learning_rate = change_learning_rate
        self.lr = 0
        self.batch_size = batch_size
        self.scaling = scaling
        self.use_gan = use_gan
        self.training_results_dir = training_results_dir
        self.checkpoints_dir = checkpoints_dir
        self.training_metrics_csv_path = training_metrics_csv_path
        self.validation_loss_filename = validation_loss_filename
        self.training_metrics_filename = training_metrics_filename
        self.config = config
        if self.config is None or 'training' not in self.config:
            raise ValueError('Callback requires config["training"] for checkpoint naming.')
        checkpoint_filename_pattern = self.config['training']['checkpoint_filename_pattern']
        if not isinstance(checkpoint_filename_pattern, str) or not checkpoint_filename_pattern:
            raise ValueError('config["training"]["checkpoint_filename_pattern"] must be a non-empty string.')
        self.checkpoint_filename_pattern = checkpoint_filename_pattern
        self.type_of_image = self.config['data']['type_of_image']
        self.validation_total_batches = (
            (self.validation_dataset_size + self.batch_size - 1) // self.batch_size
            if self.batch_size > 0 else 0
        )

        # Maps scaling name → inverse function.
        # The function receives (image, *params) where params = decoded[1:] cast to float.
        # org params are decoded[1 : n_org+1], noise params are decoded[n_org+1 : n_org+n_noise+1].
        # n_org / n_noise are the second/third elements of each tuple.
        self._inverse_fn_map = {
            'log_min_max': (inverse_adaptive_log_transform_and_denormalize, 3, 3),
            'z_scale':     (inverse_zscore_normalization,                    2, 2),
            'min_max':     (inverse_min_max_normalization,                   2, 2),
        }
        logging.info(
            'Callback initialized: dataset_size=%s, start_epoch=%s, save_freq=%s, eval_save_percentage=%s, ds_save_percentage=%s, scaling=%s',
            self.dataset_size, self.current_epoch, self.save_freq, self.eval_save_percentage,
            self.ds_save_percentage, self.scaling,
        )

    def create_directory(self, path):
        """Create a directory if it does not already exist.

        Parameters
        ----------
        path : str
            Filesystem path of the directory to create.
        """
        try:
            created = ensure_directory_exists(path)
            if created is not None:
                logging.debug('Created directory: %s', path)
        except Exception as err:
            logging.warning(f"An error occurred while creating the directory {path}: {err}")

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Parameters
        ----------
        logs : dict or None, optional
            Training logs dictionary. Default is ``None``.
        """
        self.train_start = time.time()

        logging.info(
            'Training started: dataset_size=%s, planned_epochs=%s, start_epoch=%s, save_freq=%s, use_gan=%s.',
            self.dataset_size, self.epoch, self.current_epoch, self.save_freq, self.use_gan,
        )

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of each epoch; adjusts the learning rate if scheduled.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based, as provided by Keras).
        logs : dict or None, optional
            Training logs dictionary. Default is ``None``.
        """
        self.epoch_start = time.time()
        logging.info('Epoch %d begin.', self.current_epoch + 1)
        # Find the last scheduled lr whose trigger epoch has been reached.
        applicable = [(ep, lr) for ep, lr in self.change_learning_rate if self.current_epoch >= ep]
        if applicable:
            _, lr = applicable[-1]
            if lr != self.lr:
                self.optimizer.learning_rate.assign(lr)
                self.lr = lr
                logging.warning(f'epoch: {self.current_epoch} changing to lr: {lr}')

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch.

        Performs periodic model saving, validation inference, metric tracking,
        and learning-rate adjustments.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based, as provided by Keras).
        logs : dict or None, optional
            Training logs including ``'loss'`` (or ``'d_loss'``/``'g_loss'``
            for GANs). Default is ``None``.
        """
        
        self.current_epoch += 1
        if self.use_gan:
            train_loss = logs.get('d_loss', 0) if logs else 0
            g_loss = logs.get('g_loss', 0) if logs else 0
            train_loss = train_loss / self.batch_size if self.batch_size > 0 else train_loss
        else:
            train_loss = logs.get('loss', 0) if logs else 0
            g_loss = train_loss
        epoch_time = time.time() - self.epoch_start 

        # Save training images periodically
        if self.current_epoch % max(int(self.save_freq), 1) == 0:
            result_path = f"{os.path.dirname(self.checkpoints_dir)}/results_from_epochs/{self.current_epoch:04d}"
            self.create_directory(result_path)

            n_batches = sum(1 for _ in self.dataset)
            target_count = int(self.dataset_size * max(float(self.ds_save_percentage), 0.1) // 100)
            temp = 0
            if n_batches <= 0:
                logging.warning('Epoch %d: dataset has zero batches; skipping preview generation.', self.current_epoch)
            else:
                logging.info(
                    'Epoch %d: generating up to %d preview samples from %d batches.',
                    self.current_epoch, target_count, n_batches,
                )
                batch_indices = np.arange(n_batches)
                np.random.shuffle(batch_indices)
                for index in batch_indices:
                    if temp >= target_count:
                        break
                    processed_any = False
                    for x_trains, y_trains, args in self.dataset.skip(index).take(1):
                        predictions = self.model(x_trains, training=False)
                        for x_train, y_train, prediction, arg in zip(x_trains, y_trains, predictions, args):
                            if temp >= target_count:
                                break
                            decoded = [a.decode('utf-8') for a in arg.numpy()]
                            filepath_str = decoded[0] if decoded else "sample.fits"
                            filename = os.path.basename(filepath_str)
                            org_img = y_train[:, :, 0]
                            prd_img = prediction[:, :, 0]
                            noise_img = x_train[:, :, 0]

                            try:
                                if self.scaling in self._inverse_fn_map:
                                    inv_fn, n_org, n_noise = self._inverse_fn_map[self.scaling]
                                    params = list(map(float, decoded[1:]))
                                    org_params   = params[:n_org]
                                    noise_params = params[n_org : n_org + n_noise]
                                    org_img   = inv_fn(y_train[:, :, 0],      *org_params)
                                    noise_img = inv_fn(x_train[:, :, 0],      *noise_params)
                                    prd_img   = inv_fn(prediction[:, :, 0],   *noise_params)
                            except (ValueError, TypeError, IndexError) as err:
                                logging.warning(f"Falling back to raw tensors for sample preview: {err}")
                            save_fits(org_img,   filename,         result_path, type_of_image=self.type_of_image)
                            save_fits(prd_img,   filename.replace('.fits', '_output.fits'),  result_path, type_of_image=self.type_of_image)
                            save_fits(noise_img, filename.replace('.fits', '_noise.fits'),   result_path, type_of_image=self.type_of_image)
                            temp += 1
                            processed_any = True
                    if not processed_any:
                        logging.debug('Epoch %d: batch index %d produced no preview samples.', self.current_epoch, index)
            logging.info('Saved %d preview samples for epoch %d in %s.', temp, self.current_epoch, result_path)

        # PERIODIC VALIDATION
        self.validation_start = time.time()
        if self.validation_total_batches <= 0:
            logging.warning('Epoch %d: validation dataset is empty; validation loss defaults to 0.', self.current_epoch)
            selected_validation_size = 0
            n_batches_to_use = 0
            validation_data = self.validation_dataset.take(0)
        else:
            eval_pct = max(float(self.eval_save_percentage), 1.0)
            selected_validation_size = int(self.validation_dataset_size * eval_pct // 100)
            selected_validation_size = min(max(selected_validation_size, 1), self.validation_dataset_size)
            n_batches_to_use = min(
                self.validation_total_batches,
                max(1, (selected_validation_size + self.batch_size - 1) // self.batch_size),
            )
            validation_data = self.validation_dataset.take(n_batches_to_use)
        logging.info(
            'Epoch %d: validating on ~%d samples (%d batches).',
            self.current_epoch,
            selected_validation_size,
            n_batches_to_use,
        )

        validation_loss = 0
        num_batches = 0
        for x_val, y_val, _ in validation_data:
            predictions = self.model(x_val, training=False)
            if self.use_gan:
                reconstruction_loss_fn = getattr(self.model, 'reconstruction_loss_fn', None)
                if reconstruction_loss_fn is not None:
                    loss_value = reconstruction_loss_fn(y_val, predictions)
                else:
                    loss_value = tf.reduce_mean(tf.abs(y_val - predictions))
            elif hasattr(self.model, 'compiled_loss') and self.model.compiled_loss is not None:
                loss_value = self.model.compiled_loss(y_val, predictions)
            elif hasattr(self.model, 'loss_fn'):
                loss_value = self.model.loss_fn(y_val, predictions)
            else:
                loss_value = tf.reduce_mean(tf.abs(y_val - predictions))
            validation_loss += float(tf.reduce_mean(loss_value))
            num_batches += 1
        avg_validation_loss = validation_loss / (num_batches if num_batches > 0 else 1)

        # Track best validation loss
        if avg_validation_loss < self.best_loss:
            self.best_loss = avg_validation_loss
            checkpoint_path = self.checkpoints_dir
            self.create_directory(checkpoint_path)
            try:
                best_model_filename = build_checkpoint_filename('best_model', self.current_epoch, self.checkpoint_filename_pattern)
                save_checkpoint_model(
                    self.model,
                    os.path.join(checkpoint_path, best_model_filename),
                    checkpoint_info={
                        'model_type': 'GAN' if self.use_gan else 'UNET',
                        'scaling': self.scaling,
                        'config': self.config,
                    },
                )
                logging.info('New best validation loss %.6f at epoch %d.', avg_validation_loss, self.current_epoch)
            except Exception as err:
                logging.warning(f"An error occurred while saving the best model: {err}")
        try:
            ensure_directory_exists(self.training_results_dir)
            ensure_parent_dir_exists(os.path.join(self.training_results_dir, self.validation_loss_filename))
            with open(os.path.join(self.training_results_dir, self.validation_loss_filename), "a") as file:
                file.write(f"Epoch {self.current_epoch}: Validation Loss = {avg_validation_loss:.4f}, "
                           f"Time: {(time.time() - self.validation_start):.2f}s\n")
        except Exception as err:
            logging.warning(f"An error occurred while saving the validation results: {err}")
        
        # PERIODIC MODEL SAVE
        if self.current_epoch % max(int(self.save_freq), 1) == 0:
            checkpoint_path = self.checkpoints_dir
            self.create_directory(checkpoint_path)
            try:
                model_filename = build_checkpoint_filename('model', self.current_epoch, self.checkpoint_filename_pattern)
                save_checkpoint_model(
                    self.model,
                    os.path.join(checkpoint_path, model_filename),
                    checkpoint_info={
                        'model_type': 'GAN' if self.use_gan else 'UNET',
                        'scaling': self.scaling,
                        'config': self.config,
                    },
                )
                logging.info('Saved periodic checkpoint for epoch %d.', self.current_epoch)
            except Exception as err:
                logging.warning(f"An error occurred while saving the periodic model: {err}")

        #PERIODIC SAVING OF METRICS
        current_metrics = {
        'epoch': self.current_epoch,
        'train_loss': train_loss,
        'g_loss' : g_loss,
        'validation_loss' : avg_validation_loss,
        'epoch_time': epoch_time,
        }
        self.training_metrics.append(current_metrics)
        csv_path = self.training_metrics_csv_path
        # Load existing CSV or create a new one
        df = pd.DataFrame([current_metrics])
        try:
            ensure_parent_dir_exists(csv_path)
            if os.path.exists(csv_path):
                df.to_csv(csv_path, index=False, mode='a', header=False)
            else:
                df.to_csv(csv_path, index=False)
        except Exception as err:
            logging.warning(f"An error occurred while saving the CSV file: {err}")

        logging.info(
            'Epoch %d end: d_loss=%.6f g_loss=%.6f val_loss=%.6f epoch_time=%.2fs',
            self.current_epoch,
            float(train_loss),
            float(g_loss),
            float(avg_validation_loss),
            float(epoch_time),
        )

    def on_train_end(self, logs=None):
        """Called at the end of training; saves the final model and training metrics.

        Parameters
        ----------
        logs : dict or None, optional
            Training logs dictionary. Default is ``None``.
        """
  
        checkpoint_path = self.checkpoints_dir
        self.create_directory(checkpoint_path)

        try:
            final_model_path = f'{checkpoint_path}/{build_checkpoint_filename("final_model", self.current_epoch, self.checkpoint_filename_pattern)}'
            save_checkpoint_model(
                self.model,
                final_model_path,
                checkpoint_info={
                    'model_type': 'GAN' if self.use_gan else 'UNET',
                    'scaling': self.scaling,
                    'config': self.config,
                },
            )
            logging.info('Saved final model to %s.', final_model_path)
        except Exception as err:
            logging.warning(f"An error occurred while saving the final model: {err}")

        # Save the training metrics (loss)
        training_time = time.time() - self.start_time
        try:
            ensure_directory_exists(self.training_results_dir)
            ensure_parent_dir_exists(os.path.join(self.training_results_dir, self.training_metrics_filename))
            with open(os.path.join(self.training_results_dir, self.training_metrics_filename), 'w') as file:
                file.write(f"Training completed in {training_time:.2f} seconds\n")
                file.write(f"Epoch Metrics (Epoch, Loss, Time):\n")
                for epoch, metrics in enumerate(self.training_metrics):
                    file.write(f"Epoch: {epoch}" + "\t" + 
                               f"Train_loss: {metrics.get('train_loss', 0):.4f}" + "\t" + 
                               f"Val_loss: {metrics.get('validation_loss', 0):.4f}" + "\t" + 
                               f"Time: {metrics.get('epoch_time', 0):.2f}\n")
            logging.info('Saved training metrics summary to %s.', os.path.join(self.training_results_dir, self.training_metrics_filename))
        except Exception as err:
            logging.warning(f"An error occurred while writing the training stats: {err}")
        logging.info('Training finished in %.2f seconds.', training_time)
