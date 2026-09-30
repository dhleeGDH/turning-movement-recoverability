# A conservation-bounded demand anchor for turning-movement estimation

Code and derived data for the paper *A Conservation-Bounded Demand Anchor for
Turning-Movement Estimation at Uninstrumented Signalized Intersections*
(manuscript under review).

At an uninstrumented fixed-time signalized intersection the estimator must be
chosen before any model can be fitted. The study resolves that choice with two
scalars computed from records an agency already holds. The residual
same-clock-hour demand variability measures the headroom over the target's own
history; the conservation imbalance bounds the reach of the mechanism able to
recover it. A training-free demand anchor then selects between a
boundary-diffused demand ratio and a global growth factor by hop depth and
imbalance. The recovered demand level reaches the timing plan, while the turn
split stays below a positive-control detection floor of about 0.028 RI.

## Repository structure

    src/
      graph/                       Movement-graph package
        net_parser.py                SUMO network parsing
        movement_graph.py            Movement-graph construction and hop-depth indexing
        local_demand.py              Boundary-diffused local demand ratio (the anchor core)
        observe.py                   Instrumentation masks (observed approaches and exits)
        scenarios.py                 Clustered-missing configurations and grid coordinates
        labels.py                    Turning counts and exit volumes from route traces
        to_graphpfn.py               Feature export for the DART probe
      models/                      DART probe components (cluster attention, joint model)
      inject/rho.py                Mid-link source/sink injection (conservation bias)
      eval/metrics.py              Relative-improvement (RI), RMSE, and GEH scoring
      od_gen.py                    SUMO grid and OD demand-scenario generation
      sumo_inject.py               Source/sink application over the synthetic scenarios
      rho_estimate.py              Field imbalance estimate from in/out volumes
      pilot.py                     Demand anchor and RI scoring (single held-out node)
      pilot_cluster.py             Clustered-missing recovery and RI by hop depth
      achievable_benchmark.py      Achievable-accuracy benchmark from a tree-ensemble proxy
      benchmark_proxies.py         Proxy-family comparison for the benchmark
      known_ceiling_calibration.py Known-limit calibration on synthetic conditional laws
      gaussian_calibration.py      Gaussian arm of that calibration, with an analytic limit
      count_calibration.py         Count-level calibration the known-limit protocol uses
      ipf_baseline.py              Classical IPF/Furness baseline
      split_positive_control.py    Positive control calibrating the split detection floor
      split_benchmark.py           Benchmark for the turn split on the synthetic grids
      split_benchmark_site1.py     Benchmark for the turn split at site 1
      split_benchmark_robustness.py  The site-1 split benchmark under other learners
      split_power_site1.py         Detection power for a split adjustment at site 1
      split_probe_odep.py          OD-dependence control for the split probes
      injection_sweep.py           Mid-block source/sink severity sweep
      crossover_at_depth.py        Conservation crossover by hop depth, for both contests
      microsim_delay.py            Delay response to split error, measured in microsimulation
      delay_knee_infill.py         Added design points inside the unresolved split-error gap
      delay_pooled.py              Delay response on the realized split-error axis
      dart_pretrain.py             Domain continued-pretraining of the DART probe
      bucheon.py                   Bucheon site loader (camera counts)
      bucheon_rho.py               Bucheon field-imbalance estimation
      bucheon_apply.py             Bucheon recovery evaluation
      corner_blocks.py             The four 2x2 corner blocks of site 1 and their deep corners
      realsite_hopstrat.py         Diffused against global demand ratio by stratum at both sites
      interior_sensitivity.py      Camera-error sensitivity of the site-1 interior difference
      screen_input_transfer.py     Per-intersection residual demand variability, site 1
      xc_network.py                Site-2 network, movement graph and movement tables
      xc_depth_ri.py               Site-2 recovery by depth within the masked block
      xc_rho_by_day.py             Site-2 conservation imbalance by weekday
      gate_by_day.py               Selection rule applied day by day at site 2
      xc_split_noise.py            Site-2 recovery under split noise in the reconstruction
      xc_conservation_noise.py     Site-2 interior recovery under conservation-breaking noise
      xc_block_placement.py        Site-2 recovery by masked-block size and placement
      xc_screen_input_transfer.py  Per-intersection residual demand variability, site 2
      gfm_split_probe.py           Turn-split probe of the graph foundation model
      fig_characterization.py      Figure 1
      fig_site_map.py              Figure 2 (site geometry)
      fig_reconcile.py             Figure 3
      fig_recoverability.py        Figure 4
    data/                          Derived movement-by-window tables (see Data availability)

## Data availability

The synthetic networks and their scenarios are produced by the code
(`od_gen.py`) and require a SUMO installation. The Bucheon
turning-movement counts are available from AI-Hub, a Korean-government open-data
platform that requires registration and provides an English-language application
process for international researchers. The Xuancheng data are available from
Figshare (doi:10.6084/m9.figshare.29925824) under CC BY 4.0; the turning
movements used here are reconstructed from the released vehicle trajectories,
and the derived movement-by-window tables are redistributed in `data/` with
attribution (see `data/XUANCHENG_TABLES.md`), together with the site-2 network
structure the movement graph is built from (`data/xc_feas_network.json`). The `data/` directory therefore
holds what is needed to reproduce the reported RI computations at either site
without direct platform access.

## Reproduction

Generate the synthetic scenarios, then run each stage. The demand anchor, the
benchmarks, and the split-probe battery run on CPU. Python 3.9 or later is
required.

    pip install -r requirements.txt
    export TMR_ROUTING=$PWD/routing        # where od_gen.py writes the SUMO networks
    python -m src.od_gen                        # generate SUMO grids and scenarios
    python -m src.pilot_cluster                 # anchor recovery and RI by hop depth
    python -m src.achievable_benchmark          # achievable-accuracy benchmark
    python -m src.split_positive_control        # split detection floor
    python -m src.known_ceiling_calibration     # known-limit calibration
    python -m src.injection_sweep               # source/sink severity sweep
    python -m src.crossover_at_depth            # crossover by hop depth and contest
    python -m src.split_benchmark               # split benchmark, synthetic grids
    python -m src.split_benchmark_site1         # split benchmark, site 1
    python -m src.delay_knee_infill             # delay response, added design points
    python -m src.delay_pooled                  # delay response, realized-error axis
    python -m src.anchor_lodo                   # site-1 boundary modes, date folds
    python -m src.interior_corners_lodo         # site-1 interior blocks, date folds
    python -m src.xc_aggregate_demandcv         # site-2 aggregate screening scalar
    python -m src.realsite_hopstrat             # both sites by stratum (depth table)
    python -m src.interior_sensitivity          # site-1 interior under camera error
    python -m src.screen_input_transfer         # site-1 per-intersection screen input
    python -m src.xc_screen_input_transfer      # site-2 per-intersection screen input
    python -m src.xc_depth_ri                   # site-2 recovery by depth
    python -m src.xc_rho_by_day                 # site-2 imbalance by weekday
    python -m src.gate_by_day                   # selection rule day by day, site 2
    python -m src.xc_split_noise                # site-2 split-noise test
    python -m src.xc_conservation_noise         # site-2 conservation-breaking test
    python -m src.xc_block_placement            # site-2 block size and placement
    python -m src.fig_reconcile                 # Figure 3
    python -m src.ipf_baseline                  # classical baseline
    python -m src.bucheon_apply                 # Bucheon site evaluation

The DART probe (`dart_pretrain.py`) and the graph-foundation-model split probe
(`gfm_split_probe.py`, `python -m src.gfm_split_probe`, deterministic, GPU) additionally
require PyTorch and the pinned GraphPFN and LimiX backbones. Every demand-anchor and
selection-rule result reproduces without them.

## Citation

    @misc{lee2026anchor,
      author = {Lee, Donghoun},
      title  = {A Conservation-Bounded Demand Anchor for Turning-Movement
                Estimation at Uninstrumented Signalized Intersections},
      year   = {2026},
      note   = {Manuscript under review}
    }

## License

Released under the MIT License (see `LICENSE`).
