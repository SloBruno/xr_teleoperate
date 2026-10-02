"""Read-only validation of a requested RealSense RGB stream profile.

This module intentionally uses ``config.resolve(pipeline_wrapper)`` rather than
``pipeline.start``: it asks librealsense which profile it would use and reads
that profile's intrinsics, but never opens a streaming pipeline or changes a
sensor option.  It is suitable for checking the 1280x720@15 Teleimager request
only while Teleimager/teleop are stopped.
"""
import argparse
import math


def fov_from_intrinsics(intrinsics):
    """Return RGB (horizontal_deg, vertical_deg) from pinhole intrinsics."""
    horizontal = 2 * math.degrees(math.atan(intrinsics.width / (2 * intrinsics.fx)))
    vertical = 2 * math.degrees(math.atan(intrinsics.height / (2 * intrinsics.fy)))
    return horizontal, vertical


def resolve_color_profile(rs, *, pipeline, serial, width=1280, height=720, fps=15):
    """Resolve RGB profile/intrinsics without calling ``pipeline.start()``."""
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    resolved = config.resolve(rs.pipeline_wrapper(pipeline))
    profile = resolved.get_stream(rs.stream.color).as_video_stream_profile()
    return profile, profile.get_intrinsics()


def describe_profile(serial, profile, intrinsics):
    """Return a human-readable report that distinguishes sensor and XR FOV."""
    horizontal, vertical = fov_from_intrinsics(intrinsics)
    return (
        f"RealSense {serial or '(dispositivo padrão)'} RGB resolvido: "
        f"{intrinsics.width}x{intrinsics.height} @ {profile.fps()} "
        f"({profile.format()}); intrínsecos fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}; "
        f"FOV pinhole {horizontal:.1f}° H x {vertical:.1f}° V. "
        "Esta consulta não altera o FOV óptico; XR_VIDEO_PLANE_HEIGHT=auto também não: "
        "ele só ajusta quanto da imagem já capturada ocupa no Quest."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspeciona (sem iniciar stream) o perfil RGB RealSense solicitado pelo Teleimager."
    )
    parser.add_argument("--serial", default=None, help="Serial RealSense; omita para o dispositivo padrão.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("--width, --height e --fps devem ser positivos")

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    profile, intrinsics = resolve_color_profile(
        rs,
        pipeline=pipeline,
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(describe_profile(args.serial, profile, intrinsics))


if __name__ == "__main__":
    main()
