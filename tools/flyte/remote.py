import flytekit
from flytekit import Secret
from flytekit.remote.remote import FlyteRemote

# DHW 6/29/22: I couldn't get this to work as a mounted file, but it did work as a env var.
UNION_CLIENT_ID_SECRET = Secret(
    group="controlpanel-union",
    key="clientId",
    mount_requirement=Secret.MountType.ENV_VAR,
)

UNION_CLIENT_SECRET_SECRET = Secret(
    group="controlpanel-union",
    key="clientSecret",
    mount_requirement=Secret.MountType.ENV_VAR,
)


def fetch_flyte_remote() -> FlyteRemote:
    UNION_ENDPOINT = "dns:///erisyon.hosted.unionai.cloud"
    UNION_CLIENT_ID = flytekit.current_context().secrets.get(
        group=UNION_CLIENT_ID_SECRET.group,
        key=UNION_CLIENT_ID_SECRET.key,
    )
    UNION_CLIENT_SECRET = flytekit.current_context().secrets.get(
        group=UNION_CLIENT_SECRET_SECRET.group,
        key=UNION_CLIENT_SECRET_SECRET.key,
    )

    # Configure the remote with the project-domain namespace
    # we are currently operating in
    current_execution_id = flytekit.current_context().execution_id
    remote = FlyteRemote(
        config=flytekit.configuration.Config(
            platform=flytekit.configuration.PlatformConfig(
                endpoint=UNION_ENDPOINT,
                client_id=UNION_CLIENT_ID,
                client_credentials_secret=UNION_CLIENT_SECRET,
                auth_mode=flytekit.configuration.AuthType.CLIENTSECRET,
            )
        ),
        default_domain=current_execution_id.domain,
        default_project=current_execution_id.project,
    )
    return remote
