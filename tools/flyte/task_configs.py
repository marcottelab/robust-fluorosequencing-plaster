from flytekitplugins.pod import Pod
from kubernetes.client.models import (
    V1Container,
    V1PersistentVolumeClaimVolumeSource,
    V1PodSpec,
    V1Volume,
    V1VolumeMount,
)

import plaster.tools.flyte.remote as remote


def get_job_folder_mount_path() -> str:
    return "/erisyon/jobs_folder"


def generate_efs_task_config() -> Pod:
    # This uses the default flyte container
    primary_container = V1Container(name="primary")

    pvc_name = f"flyte-efs-pvc"

    primary_container.volume_mounts = [
        V1VolumeMount(
            name="persistent-storage",
            # /erisyon here refers to the container's source root
            mount_path=get_job_folder_mount_path(),
        )
    ]

    pod_spec = V1PodSpec(
        containers=[primary_container],
        volumes=[
            V1Volume(
                name="persistent-storage",
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                    claim_name=pvc_name,
                ),
            ),
        ],
    )

    return Pod(pod_spec=pod_spec, primary_container_name="primary")


def generate_secret_requests() -> list:
    secret_requests = [
        remote.UNION_CLIENT_ID_SECRET,
        remote.UNION_CLIENT_SECRET_SECRET,
    ]
    return secret_requests
