The integrator is fully specificed with a `config.yaml` file. 
Below are the default configuration settings. 

``` yaml title="config.yaml"
global:
  dmodel: &dmodel 64
  mc_samples: &mc_samples 100

integrator:
  name: integrator
  learning_rate: 0.001
  mc_samples: 100
  profile_threshold: 0.01
  renyi_scale: 0.00
  d: 3
  h: 21
  w: 21
  weight_decay: 1.0e-4

components:
  profile_encoder:
    name: shoebox_encoder
    params:
      out_dim: 64
      in_channels: 1
      conv1_out_channels: 64
      conv2_out_channels: 128
      conv1_kernel: [1,3,3]
      conv1_padding: [0,1,1]

  intensity_encoder:
    name: shoebox_encoder
    params:
      out_dim: 64
      in_channels: 1
      conv1_out_channels: 64
      conv2_out_channels: 128
      conv3_out_channels: 256
      conv1_kernel: [3,3,3]
      conv1_padding: [1,1,1]

  profile:
    name: dirichlet
    params:
      dmodel: 64
      input_shape: [3,21,21]

  decoder:
    name: default_decoder
    params:
      mc_samples: *mc_samples

  q_bg:
    name: folded_normal
    params:
      dmodel: dmodel
      out_features: 2
      use_metarep: false

  q_i:
    name: folded_normal
    params:
      dmodel: *dmodel
      out_features: 2
      use_metarep: false

  loss:
    name: loss2
    params:
      p_I_name: half_cauchy
      p_I_params: 
        scale: 0.2

      p_bg_name: half_cauchy
      p_bg_params: 
        scale: 0.05

      p_prf_name: dirichlet
      p_prf_params:
        concentration: 1.0

      p_prf_scale: 0.0001
      p_bg_weight: 0.0001
      p_I_weight: 0.0001
      prior_tensor: "/Users/luis/Downloads/new/concentration.pt"
      use_robust: false


data_loader:
  name: default
  params:
    data_dir: "/Users/luis/integratorv3/integrator/data/hewl_816/"
    batch_size: 10
    val_split: 0.3
    test_split: 0.0
    num_workers: 1
    include_test: false
    subset_size: 100
    cutoff: null
    use_metadata: false
    shoebox_features: true
    shoebox_file_names:
      counts: "/Users/luis/Downloads/new/counts.pt"
      metadata: "/Users/luis/Downloads/new/standardized_metadata.pt"
      masks: "/Users/luis/Downloads/new/masks.pt"
      stats: "/Users/luis/Downloads/new/stats.pt"
      reference: "/Users/luis/Downloads/new/reference.pt"
      standardized_counts: null

trainer:
  params:
    max_epochs: 50
    accelerator: auto
    devices: 1
    logger: true
    precision: "32"
    check_val_every_n_epoch: 2
    log_every_n_steps: 1
    deterministic: false
    callbacks:
      pred_writer:
        output_dir: null
        write_interval: epoch
    enable_checkpointing: true

output:
  refl_file: "/Users/luis/integratorv3/integrator/data/hewl_816/reflections_.refl"
```

