from plumbum import local

from plaster.tools.ssh import ssh


def p_options(log_file, deed, docker_tag, include_jupyter, extra):
    port_args = "'-p 8080:8080'"  # Note the extra single quoes inside
    return (
        f"DO_NOT_OVERWRITE_PLASTER_SH=1 "  # Because we send it with the launch statement
        f"ON_AWS=1 "
        f"RUN_USER={local.env['RUN_USER']} "
        f"FOLDER_USER={local.env['FOLDER_USER']} "
        f"ROOT=1 "
        f"FORCE_PULL=1 "
        f"ERISYON_HEADLESS=1 "
        f"ERISYON_TMP=/home/ubuntu/erisyon_tmp "
        f"DEBUG=1 "
        f"EXTRA_DOCKER_ARGS={port_args if include_jupyter else ''} "
        f"PLASTER_DATA_FOLDER=/home/ubuntu/jobs_folder/{local.env['FOLDER_USER']} "  # Host namespace
        f"IMAGE=188029688209.dkr.ecr.us-east-1.amazonaws.com/erisyon:{docker_tag} "
        f"DEED={deed} "
        f"LOG_FILE={log_file} "  # LOG_FILE is in the HOST namespace
        f"FORCE_NEW_CONTAINER=1 " + extra
    )


def wrap_with_p_encoding(
    deed,
    p_command,
    log_file,  # host namespace
    send_p=False,
    docker_tag="cloud",
    include_jupyter=False,
    extra="",
):
    assert not p_command.startswith("./p") and not p_command.startswith("p ")
    assert docker_tag is not None and docker_tag != ""

    if deed is None:
        deed = local.env["DEED"]

    if send_p:
        return (
            ssh.encode_file(local_path="./p", remote_path="./p")
            + " && "
            + ssh.bash_encode(
                f"chmod +x ./p "
                f"&& {p_options(log_file, deed, docker_tag, include_jupyter, extra)} ./p {p_command} "
            )
        )
    else:
        return ssh.bash_encode(
            f"{p_options(log_file, deed, docker_tag, include_jupyter, extra)} ./p {p_command} "
        )
