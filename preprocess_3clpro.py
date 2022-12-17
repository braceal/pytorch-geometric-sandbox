from pathlib import Path
from mdgraph.data.preprocess import parallel_preprocess
from mdgraph.data.utils import concatenate_h5
md_run_dirs = sorted(
    filter(
        lambda p: p.is_dir(),
        Path("/homes/heng.ma/summit_backup/proj_dir/proj_shared/mpro/entk_cvae_md_comp/MD_exps/mpro/").glob("omm_runs*"),
    )
)
traj_files = [next(p.glob("*.dcd")) for p in md_run_dirs]
topology_files = [next(p.glob("*.pdb")) for p in md_run_dirs]
ref_topology = "/homes/heng.ma/summit_backup/proj_dir/proj_shared/mpro/entk_cvae_md_comp/Parameters/input_protein/prot.pdb"
save_files = [f"/lambda_stor/homes/abrace/data/3clpro/entk_cvae_md_comp_h5/{p.with_suffix('').name}.h5" for p in md_run_dirs]
concatenated_save_file = "/lambda_stor/homes/abrace/data/3clpro/3clpro_entk_cvae_md_comp.h5"

parallel_preprocess(
    topology_files,
    traj_files,
    ref_topology,
    save_files,
    cutoff=8.0,
    selection="protein and name CA",
    print_every=1000,
    num_workers=5,
)

concatenate_h5(save_files, concatenated_save_file)
