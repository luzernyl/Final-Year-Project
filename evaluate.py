import gc
import io
import sys
import os
from torchvision.utils import make_grid, save_image
import tensorflow as tf
import tensorflow_gan as tfgan
import numpy as np
import evaluation
import ddpm_sde
import copy
import torch
import sde_lib
import logging

#inceptionv3 = config.data.image_size >= 256
inceptionv3 = False
inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
num_samples = 10000
device = 'cuda'
model = ddpm_sde.UNet().to(device)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)
checkpoint = torch.load(r"/content/unconditional_cifar_32_vpsde_200e.pt")
model.load_state_dict(checkpoint['model_state_dict'])
ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
sde = sde_lib.VPSDE()
diffusion = ddpm_sde.Diffusion(img_size=32, noise_steps=1000, device=device)
batch_size = 100
start = 94

num_sampling_rounds = num_samples // 100 + 1
for r in range(start, num_sampling_rounds):
    #logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
    # Directory to save samples. Different for each host to avoid writing conflicts
    tf.io.gfile.makedirs("evaluate")
    tf.io.gfile.makedirs("evaluate/samples")
    tf.io.gfile.makedirs("evaluate/statistics")
    tf.io.gfile.makedirs("results")
    samples = diffusion.sample(model=ema_model, sde=sde, solver='sie', eps=1e-3, n=batch_size)

    nrow = int(np.sqrt(samples.shape[0]))
    image_grid = make_grid(samples, nrow, padding=2)
    with tf.io.gfile.GFile(
        os.path.join("results", f"samples_{r}.png"), "wb") as fout:
        save_image(image_grid, fout)

    samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
    samples = samples.reshape((-1, 32, 32, 3))

    '''
    #plt.imshow(samples[0,:,:,:])
    plt.figure(figsize=(32,32))
    plt.imshow(np.concatenate([
        np.concatenate([i for i in samples[0:10,:,:,:]]),
    ]).permute(1, 2, 0).cpu())
    plt.show()
    '''
    #sample = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    #with tf.io.gfile.GFile(
    #    os.path.join("", "sample.np"), "wb") as fout:
    #    np.save(fout, samples)

    # Write samples to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join("evaluate/samples", f"samples_{r}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, samples=samples)
      fout.write(io_buffer.getvalue())

    # Force garbage collection before calling TensorFlow code for Inception network
    gc.collect()
    latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
    # Force garbage collection again before returning to JAX code
    gc.collect()
    # Save latent represents of the Inception network to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join("evaluate/statistics", f"statistics_{r}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
      fout.write(io_buffer.getvalue())

    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    #this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
    stats = tf.io.gfile.glob(os.path.join("evaluate/statistics", "statistics_*.npz"))
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])

    if not inceptionv3:
      all_logits = np.concatenate(all_logits, axis=0)[:num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:num_samples]

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats("CIFAR10")
    data_pools = data_stats["pool_3"]

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
      inception_score = -1

    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    kid = tfgan.eval.kernel_classifier_distance_from_activations(tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    logging.info("inception_score: %.6e, FID: %.6e, KID: %.6e" %(inception_score, fid, kid))
    print(f"Inception score : ", end="")
    tf.print(inception_score, output_stream=sys.stderr)
    print(f"FID : ", end="")
    tf.print(fid, output_stream=sys.stderr)
    print(f"KID : ", end="")
    tf.print(kid, output_stream=sys.stderr)

    ckpt = 1
    with tf.io.gfile.GFile(os.path.join("evaluate", f"report_{ckpt}.npz"),
                             "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      f.write(io_buffer.getvalue())