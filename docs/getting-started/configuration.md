The integrator is fully specificed with a `config.yaml` file. 
Below are the default configuration settings. 

=== "2D"

    ```{.yaml .annotate}
    global_vars:
      in_features: &in_features 32
      mc_samples: &mc_samples 100
      data_dir: &data "/path/to/data"

    integrator:
      name: integrator_2d
      args:
        lr: 0.001
        mc_samples: *mc_samples
        renyi_scale: 0.0
        h: 21
        w: 21
        weight_decay: 0.0

    components:
      encoders: 
        - encoder1:
            name: shoebox_encoder_2d
            args:
              out_features: *in_features
              in_channels: 1
              conv1_out_channels: 16
              conv2_out_channels: 32
              conv1_kernel: [3,3]
              conv1_padding: [1,1]

        - encoder2:
            name: intensity_encoder_2d
            args:
              out_features: *in_features
              in_channels: 1
              conv1_out_channels: 16
              conv2_out_channels: 32
              conv3_out_channels: 64
              conv1_kernel: [3,3]
              conv1_padding: [1,1]

      qp:
        name: dirichlet
        args:
          in_features: *in_features
          input_shape: [21,21]

      qbg:
        name: folded_normal
        args:
          in_features: *in_features
          out_features: 2

      qi: 
        name: folded_normal
        args: 
          in_features: *in_features
          out_features: 2

      loss:
        name: loss
        args:
          pi_name: half_cauchy
          pi_params:
            scale: 0.5

          pbg_name: half_cauchy
          pbg_params:
            scale: 0.5

          pprf_name: dirichlet
          pprf_params:
            concentration: 1.0
          pprf_weight: 0.02
          pbg_weight: 0.5
          pi_weight: 0.4
          prior_tensor: "concentration_2d.pt"
          use_robust: false

    data_loader:
      name: shoebox_data_module_2d
      args:
        data_dir: "/path/to/data"
        batch_size: 256
        val_split: 0.3
        test_split: 0.0
        num_workers: 3
        include_test: false
        subset_size: null
        cutoff: null
        use_metadata: true
        shoebox_file_names:
          counts: "counts_2d_subset.pt"
          masks: "masks_2d_subset.pt"
          stats: 'stats_2d.pt'
          reference: 'reference_2d_subset.pt'
          standardized_counts: null

    trainer:
      args:
        max_epochs: 1000
        accelerator: auto
        devices: 1
        logger: true
        precision: "32"
        check_val_every_n_epoch: 2
        log_every_n_steps: 3
        deterministic: false
        callbacks:
          pred_writer:
            output_dir: null
            write_interval: epoch
        enable_checkpointing: true

    logger:
        d: 3
        h: 21
        w: 21

    output:
      refl_file: "/path/to/.refl"
    ```

=== "3D"

    ```{.yaml .annotate}
    global_vars:
      in_features: &in_features 64
      mc_samples: &mc_samples 100
      data_dir: &data "/path/to/data"

    integrator:
      name: integrator
      args:
        lr: 0.001
        mc_samples: *mc_samples
        renyi_scale: 0.0
        d: 3
        h: 21
        w: 21
        weight_decay: 0.0

    components:
      encoders: 
      - encoder1:
          name: shoebox_encoder
          args:
            out_features: 64
            in_channels: 1
            conv1_out_channels: 64
            conv2_out_channels: 128
            conv1_kernel: [1,3,3]
            conv1_padding: [0,1,1]

      - encoder2:
          name: intensity_encoder
          args:
            out_features: 64
            in_channels: 1
            conv1_out_channels: 64
            conv2_out_channels: 128
            conv3_out_channels: 256
            conv1_kernel: [3,3,3]
            conv1_padding: [1,1,1]

      qp:
        name: dirichlet
        args: 
          in_features: *in_features
          input_shape: [3,21,21]

      qbg:
        name: folded_normal
        args:
          in_features: *in_features
          out_features: 2

      qi:
        name: folded_normal
        args:
          in_features: 64
          out_features: 2

      loss:
        name: loss
        args:

          pi_name: half_normal
          pi_params:
            scale: 0.5

          pbg_name: half_normal
          pbg_params:
            scale: 1.0

          pprf_name: dirichlet
          pprf_params: null

          pprf_weight: 0.0001
          pbg_weight: 0.5
          pi_weight: 1.0
          use_robust: true
          data_dir: *data

          prior_tensor: 'concentration_3d.pt'

    data_loader:
      name: default
      args:
        data_dir: *data
        batch_size: 256
        val_split: 0.3
        test_split: 0.0
        num_workers: 0
        include_test: false
        subset_size: 100
        cutoff: 2000
        use_metadata: true
        shoebox_file_names:
          counts: "counts_3d_subset.pt"
          masks: "masks_3d_subset.pt"
          stats: "stats_3d.pt"
          reference: "reference_3d_subset.pt"
          standardized_counts: null

    trainer:
      args:
        max_epochs: 600
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

    logger: 
      d: 3
      h: 21
      w: 21

    output:
      refl_file: "/path/to/.refl"
    ```


