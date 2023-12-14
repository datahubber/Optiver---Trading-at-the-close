from pip._vendor import pkg_resources


def find_dependencies(package_name):
    package = pkg_resources.working_set.by_key[package_name]

    print([str(dependency) for dependency in package.requires()])


find_dependencies('pytorch-tabnet')
