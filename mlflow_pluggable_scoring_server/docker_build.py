"""
Build docker image for a model plugin.
"""

import os
import shutil
import click
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

docker_file = "docker/Dockerfile"

def run_command(cmd):
    from subprocess import Popen, PIPE
    print("Running command:",cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    proc.wait()
    if (proc.stderr):
        output = proc.stderr.read()
        if len(output) > 0:
            raise Exception(f"Failed to execute command '{cmd}'. Error: {output}")

@click.command()
@click.option("--packages", help="PyPI packages (comma delimited).", default=None, type=str)
@click.option("--model-uri", help="Model URI.", required=True, type=str)
@click.option("--plugin-file", help="Python plugin file.", required=True, type=str)
@click.option("--docker-image", help="Docker image name", required=True, type=str)
@click.option("--tmp-model-dir", help="Temporary model directory for docker COPY. Default is './tmp'.", default="tmp", type=str)

def main(packages, model_uri, plugin_file, docker_image, tmp_model_dir):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    dst_model_dir = tmp_model_dir
    if os.path.exists(dst_model_dir):
        shutil.rmtree(dst_model_dir)
    os.makedirs(dst_model_dir)
    _download_artifact_from_uri(model_uri, dst_model_dir)

    packages = packages.replace(","," ")
    packages_cmd = [ "--build-arg", f"PACKAGES={packages}" ] if packages else []

    run_command(["python", "setup.py", "bdist_wheel"])
    run_command([
        "docker", "build", "-t", docker_image,
         "--build-arg", f"MODEL_DIR={dst_model_dir}",
         "--build-arg", f"PLUGIN_FILE={plugin_file}" ]
         + packages_cmd
         + [ "-f", docker_file, "."]
    )

if __name__ == "__main__":
    main()
