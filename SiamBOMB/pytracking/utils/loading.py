from os import path
from ..admin import loading as ltr_loading

pytracking_path = path.dirname(path.dirname(path.abspath(__file__)))
network_path = path.join(pytracking_path, 'networks')

def load_network(net_path, **kwargs):
    """Load network for tracking.
    args:
        net_path - Path to network. If it is not an absolute path, it is relative to the network_path in the local.py.
                   See pytracking.admin.loading.load_network for further details.
        **kwargs - Additional key-word arguments that are sent to pytracking.admin.loading.load_network.
    """
    kwargs['backbone_pretrained'] = False
    if path.isabs(net_path):
        path_full = net_path
        net, _ = ltr_loading.load_network(path_full, **kwargs)
    elif isinstance(network_path, (list, tuple)):
        net = None
        for p in network_path:
            path_full = path.join(p, net_path)
            try:
                net, _ = ltr_loading.load_network(path_full, **kwargs)
                break
            except Exception as e:
                # print(e)
                pass

        assert net is not None, 'Failed to load network'
    else:
        path_full = path.join(network_path, net_path)
        net, _ = ltr_loading.load_network(path_full, **kwargs)

    return net
