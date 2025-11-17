supress reconstruction gpu naive

in weno_cuda, pourquoi ne pas utiliser les fonctions optimisées de reconstruction de weno et utiliser autre chose


Approach 3: Create a new central GPU utility file, for example numerics/gpu/kernels.py, move all related kernels (weno5_reconstruction_kernel, _compute_flux_divergence_weno_kernel, etc.) into it, and update all files to import from this new single source of truth. This would be a larger refactoring but could improve long-term maintainability.



Model Parameters, est ce que c'est pas en fait un legacy