"""Start the Teleimager server with ONLY the head camera (explicit mode).

The submodule and its cam_config_server.yaml stay untouched: this wrapper
derives an in-memory head-only config, writes it to a separate file and points
``teleimager.image_server.CONFIG_PATH`` at it before calling the stock ``main``.
The server then never opens the wrist cameras, and the config it serves on
port 60000 tells ImageClient that they are disabled.
"""
import copy
import os
import sys

BANNER = "TELEIMAGER: modo SOMENTE CABEÇA (pulso esquerdo desativado)"
WRIST_CAMERAS = ("left_wrist_camera", "right_wrist_camera")


def head_only_config(cam_config):
    """Return a copy of ``cam_config`` with every wrist camera disabled."""
    head = (cam_config or {}).get("head_camera")
    if not head or not (head.get("enable_zmq") or head.get("enable_webrtc")):
        raise ValueError("head_camera must be enabled in the camera config for head-only mode")
    result = copy.deepcopy(cam_config)
    for name in WRIST_CAMERAS:
        section = result.setdefault(name, {})
        section["enable_zmq"] = False
        section["enable_webrtc"] = False
    return result


def realsense_serials():
    import pyrealsense2 as rs

    return {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}


def main():
    import yaml
    from teleimager import image_server

    with open(image_server.CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    derived = head_only_config(config)
    head = derived["head_camera"]
    serial = str(head.get("serial_number"))
    if head.get("type") == "realsense" and "--rs" in sys.argv:
        present = realsense_serials()
        if serial not in present:
            print(f"head camera serial {serial} not found (present: {sorted(present)}); refusing to start", file=sys.stderr)
            sys.exit(3)
    out = os.environ.get("TELEIMAGER_HEAD_ONLY_CONFIG") or os.path.join(
        os.path.dirname(os.path.abspath(image_server.CONFIG_PATH)), ".cam_config_server.head_only.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(derived, f, sort_keys=False)
    image_server.CONFIG_PATH = out
    print(BANNER, flush=True)
    image_server.logger_mp.warning(BANNER)
    image_server.main()


if __name__ == "__main__":
    main()
