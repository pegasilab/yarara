import os

__all__ = ["cwd", "root"]

cwd = os.getcwd()
root = "/".join(cwd.split("/")[:-1])
