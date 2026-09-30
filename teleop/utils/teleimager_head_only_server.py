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


HEAD_SERIAL = "243122072230"
LEFT_WRIST_SERIAL = "233622070789"
SOURCE_LABELS = {"head": "cabeça", "left_wrist": "pulso esquerdo"}
WRIST_WARNING = (
    "AVISO: a imagem principal vem da câmera do PULSO esquerdo — o ponto de vista é o da mão, "
    "não o da cabeça, e isso pode confundir a teleoperação."
)


def select_single_camera(present):
    """Pick the one camera to publish: returns (source, serial, both_connected)."""
    has_head = HEAD_SERIAL in present
    has_wrist = LEFT_WRIST_SERIAL in present
    if not has_head and not has_wrist:
        raise RuntimeError(
            f"nenhuma câmera conectada (esperado {HEAD_SERIAL} cabeça ou {LEFT_WRIST_SERIAL} pulso esquerdo; "
            f"presentes: {sorted(present)})"
        )
    if has_head:
        return "head", HEAD_SERIAL, has_wrist
    return "left_wrist", LEFT_WRIST_SERIAL, False


def single_banner(source, serial):
    return f"TELEIMAGER: modo CÂMERA ÚNICA ({SOURCE_LABELS[source]} serial {serial} publicada como imagem principal)"


def single_camera_config(cam_config, source, serial):
    """Copy of ``cam_config`` where the only present camera is published as head_camera."""
    if source not in SOURCE_LABELS:
        raise ValueError(f"unknown camera source {source!r}")
    result = copy.deepcopy(cam_config or {})
    head = result.setdefault("head_camera", {})
    head.update({
        "enable_zmq": True, "zmq_port": 55555, "enable_webrtc": False, "type": "realsense",
        "image_shape": [720, 1280], "binocular": False, "fps": 15,
        "video_id": None, "serial_number": serial, "physical_path": None,
    })
    for name in WRIST_CAMERAS:
        section = result.setdefault(name, {})
        section["enable_zmq"] = False
        section["enable_webrtc"] = False
    return result


def realsense_serials():
    import pyrealsense2 as rs

    return {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}


def detect_main():
    """``--detect``: print '<source> <serial>' of the camera to use; exit 3 if none."""
    try:
        source, serial, both = select_single_camera(realsense_serials())
    except RuntimeError as exc:
        print(f"TELEIMAGER: {exc}", file=sys.stderr)
        sys.exit(3)
    print(f"{source} {serial} {'both' if both else 'single'}")


def main_any():
    import yaml
    from teleimager import image_server

    try:
        source, serial, both = select_single_camera(realsense_serials())
    except RuntimeError as exc:
        print(f"TELEIMAGER: {exc}; refusing to start", file=sys.stderr)
        sys.exit(3)
    with open(image_server.CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    derived = single_camera_config(config, source, serial)
    out = os.environ.get("TELEIMAGER_HEAD_ONLY_CONFIG") or os.path.join(
        os.path.dirname(os.path.abspath(image_server.CONFIG_PATH)), ".cam_config_server.any.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(derived, f, sort_keys=False)
    image_server.CONFIG_PATH = out
    msgs = [single_banner(source, serial)]
    if both:
        msgs.append("AVISO: cabeça e pulso conectados; modo any usa apenas a cabeça.")
    if source == "left_wrist":
        msgs.append(WRIST_WARNING)
    for m in msgs:
        print(m, flush=True)
        image_server.logger_mp.warning(m)
    image_server.main()


def main():
    if "--detect" in sys.argv:
        return detect_main()
    if os.environ.get("TELEIMAGER_CAMERA_MODE") == "any":
        return main_any()
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
