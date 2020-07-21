How to run
-----------
My code requires `lsssys` from the des-science/lss_sys repository (this is a private repository).

I also use `healcorr` by `pierfied <https://github.com/pierfied/healcorr>`, and `healpix_util` by `esheldon <https://github.com/esheldon/healpix_util>`.

Other requirements: scipy, numpy, matplotlib, healpy, astropy, twopoint, chainconsumer, emcee, treecorr

I've only tried the code with Python 3.7. Some of the built in packages I use don't work in Python 2 (e.g., pathlib).

1. Run `linear_systematics_pipeline_lsssys.py` on data
#. Run `make_mocks.py` to generate mock catalogs
#. Run `linear_systematics_pipeline_lsssys.py` on mocks
#. Optionally, run `systematics_pipeline_plots.py` for plots and tables

Things that will need to be changed
-----------------------------------
linear_systematics_pipeline_lsssys.py
=====================================
Most of the options are handled starting at line 2716. I use a dictionary to pass options into the various functions.

The path options that will need to be updated are:

1. Line 32, 'lsssys_path': Point this to the location of the `lss_sys` repo
#. Line 2727-2728, 'des_root_path': The location where the DES data (catalogs, masks, systematics maps, fiducial data vector) can be found.
#. Line 2729-2730, 'fracgood_fname': This is a template for the name of the redMaGiC mask file for different resolutions. If you use a different mask file, you will likely need to change this.
#. Line 2746-2747, 'mock_run': The name of the mock generation run, which is used as the name for the subdirectories containing the mock results and chains (only needed when running on mocks)
#. Line 2748-2750, 'mock_top_path': The location in which the mock catalogs are stored (only needed when running on mocks)
#. Line 2751-2752, 'sys_root_path': The root directory containing the DES systematics maps
#. Line 2753-2754, 'cat_root_path': The root directory containing the DES catalogs (data and random) and mask
#. Line 2755-2757, 'results_top_path': The root directory that will contain both the results and the chains
#. Line 2762-2764, 'dcat_path': The path to the DES redMaGiC catalog
#. Line 2765-2767, 'rcat_path': The path to the DES redMaGiC random catalog
#. Line 2819-2820, 'des_data': The path to the fiducial DES data vector 2point file

Other options that you might want to change:

1. Line 2711, 'des_year': I'm using this to make it easier to switch between DES year 1 and year 3 (redshift binning, etc.). If you aren't using either of these, you'll need to change this line as well as add an option in lines 2712-2717 for the correct redshift binning to be sued
#. Lines 2713 and 2715, 'zedges': An array of the redshift bin edges. The first is for Y1 and the second is for Y3. If you are using either of these years and you want different binning, change the appropriate line.
#. Line 2718, 'theta_edges': The angular bin edges
#. Line 2719, 'contamination': The contamination levels of the mocks that are being fit (only used for naming purposes, and only if running on mocks)
#. Line 2722, 'rotated': Change this to 'False' if you don't want to work in the eigenbasis (results aren't tested with this option, use at your own risk)
#. Line 2723, 'zbins': Specify which redshift bins you want to run on (in case you only want to do a subset of them)
#. Line 2725, 'is_mock': This should be 'False' when running on data and 'True' when running on mocks
#. Line 2731, 'fit_type': The options for this are 'diag' (which only does the first iteration fit) or 'const_cov' (which does both iterations)
#. Line 2732, 'chain_version': Change this if you want to name the output as a different version
#. Line 2733, 'dcat_zcol': This should be the column name for redshift in the DES data catalog
#. Line 2734, 'rcat_zcol': This should be the column name for redshift in the DES random catalog
#. Line 2736-2737, 'nside_fit': The resolution at which to fit, per redshift bin. Only change the list, as the code expects it to be a dictionary
#. Line 2738, 'do_fits': Change to 'False' to skip fitting (if already done, for instance)
#. Line 2739, 'do_corr': Change to 'False' to skip calculating the correlation function (so that fit results can be checked first, for instance)
#. Line 2743, 'max_nthreads': The maximum number of threads to use
#. Line 2744, 'max_nprocesses': The maximum number of processes to use for the MCMC
#. Line 2772, 'force_do_fits': Change to 'True' to always run the chain, even if the chain file already exists (appends the new chain to the old one)
#. Lines 2779-2780 and 2782-2783, 'nburnin' and 'nsteps': The burn-in phase to remove from the chain and the number of steps after the burn-in to keep. The first two lines are the options when running on the mocks, the second two are the options for running on data
#. Line 2803, 'force_do_corr': Change to 'True' to always calculate the correlation functions, even if the files already exist
#. Line 2804, 'delta_elin_max': The maximum systematic overdensity to allow in the systematics mask
#. Line 2805, 'nside': The resolution at which to apply the weights and the systematics mask
#. Line 2821, 'nreal': The number of steps in the data chain at which to evaluate the weighted correlation function (for calculating the statistical covariance matrix due to the weights)

make_mocks.py
=============
Most of the options are supplied at the command line. Run `make_mocks.py --help` to see all the arguments.

The one thing that will need to change is line 11, 'lsssys_path': this should again be the path to the `lss_sys` repo

systematics_pipeline_plots.py
=============================
All changes start at line 2043. 'chain_version' should match the version specified in `linear_systematics_pipeline_lsssys.py`. 'ddir' is the directory containing DES data (including the fiducial data vector). 'sdir' contains the systematics maps. 'rdir' is the root results directory, and should match 'results_top_path' from `linear_systematics_pipeline_lsssys.py`. 'pdir' is the subdirectory in the results directory containing results (correlation functions, coefficents) and 'cdir' is the subdirectory containing chains. 'odir' is the directory in which to store the plots and tables. 'des_2pt_name' is the name of the fiducial data vector file. The other options supplied to the main function (the numbers appearing between 'chain_version' and 'des_2pt_name) are:

1. The resolution at which the fits were performed (128)
#. The resolution at which the weights and systematics mask were applied (4096)
#. The number of steps of the chain being used to estimate the statistical covariance matrix from the weights (250)
#. The number of mock catalogs (100)
#. The number of systematics maps (18 -- this is for DES Y1)
#. The minimum angular separation to include in the bias plots, in arcminutes (8.0)
#. The minimum angular separation to include in the correlation function and covariance matrix plots, in arcminutes (0.0)
#. The maximum systematic overdensity that was kept in the systematics mask (0.2)
#. The number of steps after burn-in for the mock chains (700)
#. The number of steps removed as a burn-in for the mock chains (300)