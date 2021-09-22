from .environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/dc0/SiamBOMB_0831/SiamBOMB/researchmm/Stark/lib/test/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/dc0/SiamBOMB_0831/SiamBOMB/researchmm/Stark/lib/test/result_plots/'
    settings.results_path = '/dc0/SiamBOMB_0831/SiamBOMB/researchmm/Stark/lib/test/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/dc0/SiamBOMB_0831/SiamBOMB/researchmm/Stark/lib/test/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

