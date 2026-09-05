# Site-2 derived tables

`xc2023_feas_dataset.npz` and `xc_feas_arrays.npz` hold the movement-by-window counts and
approach volumes for the masked block at site 2, reconstructed from the trip-level release

    City-scale high-resolution traffic datasets with refined networks for hierarchical
    traffic control, figshare, doi:10.6084/m9.figshare.29925824

which is licensed CC BY 4.0. These tables are derived works redistributed under the same
licence, with attribution to that release.

    Y        movement counts, (window, movement)
    APPR     approach-link volume per movement, same shape
    date     the date of each window
    window   the within-day slot index
    binmin   the window length in minutes
    inter_of the intersection of each movement
