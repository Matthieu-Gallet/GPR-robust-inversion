import h5py
import pprint

pp = pprint.PrettyPrinter()


def examine_h5(h5_full_path, print_i=False):
    # This function prints all of the metadata for a given h5 file
    h5_info = {}
    with h5py.File(h5_full_path.as_posix(), 'r') as f:
        groups = list(f.keys())
        h5_info.update({'H5 Groups': groups})

        for group in groups:
            attrs = list(f[group].attrs.items())
            h5_info.update({f'{group} metadata': attrs})

            datasets = list(f[group].keys())
            h5_info.update({f'{group} datasets': datasets})
            for dataset in datasets:
                attrs = list(f[group][dataset].attrs.items())
                h5_info.update({f'{group} : {dataset} metadata': attrs})
    if print_i:
        pp.pprint(h5_info)
    return h5_info


def get_matrices(h5_full_path):
    with h5py.File(h5_full_path.as_posix(), 'r') as f:
        raw = {}
        procs = {}
        masks = {}
        raw.update({'Raw_Data_Array': f['Raw_Data']['Raw_Data_Array'][()]})

        proc_group = f['Processed_Data']
        for proc_name in proc_group.keys():
            procs.update({proc_name: f['Processed_Data'][proc_name][()]})

        mask_group = f['Masks']
        for mask_name in mask_group.keys():
            masks.update({mask_name: f['Masks'][mask_name][()]})
    return raw, procs, masks
