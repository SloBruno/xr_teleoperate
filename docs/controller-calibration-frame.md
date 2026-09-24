# Controller calibration frame contract

`G1_29_ArmIK.forward_kinematics()` calls Pinocchio `framesForwardKinematics`
on the reduced model and returns `reduced_robot.data.oMf[L_ee]` and
`oMf[R_ee]`. `solve_ik()` puts the calibration output directly into the
CasADi targets compared with those same `oMf` frames. The reduced model's
fixed zero waist chain starts at the URDF `pelvis`; therefore the calibration
workspace is documented as pelvis-root, not “waist”.

`TeleVuerWrapper` first converts controller poses from OpenXR world into the
robot convention, makes them head-relative (yaw-only by default), and then
adds the fixed synthetic translation `[+0.15, 0, +0.45]` m. That returned
`TeleData.left/right_wrist_pose` is the controller input to calibration; it is
not assumed to share the Pinocchio root origin. Calibration therefore keeps
translation and rotation deltas separate: controller translation is added to
the measured wrist position, while controller relative rotation is composed
with the measured wrist orientation. It must not use one rigid
`inv(controller) @ wrist` offset, because rotating that cross-origin offset
would make an in-place controller rotation translate the wrist target.
Calibration returns the exact measured FK pose once and validates every later
mapped target in pelvis-root.

The workspace gate uses the URDF shoulder origins after the fixed pelvis →
waist-yaw → waist-roll → torso chain: left `[-0.0000072, +0.10022,
0.29178]`, right `[-0.0000072, -0.10021, 0.29178]` metres. Targets must be
0.18–0.50 m from their own shoulder origin and remain in their shoulder's
pelvis side half-space (`y >= 0` left, `y <= 0` right). The lower radius
keeps targets out of the torso/shoulder neighborhood; the upper radius is a
conservative rejection bound below the loose URDF link-length sum. It is not
a complete joint-limit reachability proof, so IK remains the final feasibility
check and rejected/stale/invalid targets hold measured joints with zero torque.
