import os
import imp
import configs.config as cfg


def get_network_class_constructor(checkpoint: str, use_stripped: bool) -> "NetworkClass":
    if checkpoint is None:
        from network.network_class import NetworkClass

        return NetworkClass

    dir_chekpoint = "%s/chk_%s" % (cfg.DIRS.OUTPUT_MODELS, checkpoint)
    py_path = "%s/file_copies/network/__init__.py" % dir_chekpoint

    if os.path.isfile(py_path):
        module_model = imp.load_package("_network", py_path)

        return module_model.StrippedNetworkClass if use_stripped else module_model.NetworkClass

    from network.network_class import NetworkClass
    
    return NetworkClass


def get_network_index_constructor(checkpoint: str, use_stripped: bool) -> "NetworkIndex":
    if checkpoint is None:
        from network.network_index import NetworkIndex

        return NetworkIndex

    dir_chekpoint = "%s/chk_%s" % (cfg.DIRS.OUTPUT_MODELS, checkpoint)
    py_path = "%s/file_copies/network/__init__.py" % dir_chekpoint

    if os.path.isfile(py_path):
        module_model = imp.load_package("_network", py_path)

        return module_model.StrippedNetworkIndex if use_stripped else module_model.NetworkIndex

    from network.network_index import NetworkIndex
    
    return NetworkIndex
