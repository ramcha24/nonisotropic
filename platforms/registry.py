from platforms import local, distributed

registered_platforms = {"local": local.Platform, "distributed": distributed.Platform}


def get(name):
    return registered_platforms[name]
