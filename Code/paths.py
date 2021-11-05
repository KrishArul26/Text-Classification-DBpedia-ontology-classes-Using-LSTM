# Script to centralize different path locations.

class Paths(object):
    """
    Class to hold different paths
    """
    data_path = "data/"
    model = "models"

    def __init__(self, path_file="paths.properties"):
        paths_dict = dict(
            line.strip().split('=')
            for line in open(path_file)
            if not line.startswith('#') and not line.startswith('\n'))

        for k, v in paths_dict.items():
            exec("self.%s=\"%s\"" % (k, v))

# end of class Paths
