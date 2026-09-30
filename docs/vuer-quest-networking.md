# Vuer/Quest networking and TLS

`teleop/run_g1_quest_dex3.sh` selects the Quest endpoint before it starts
Teleimager or teleoperation. It prints the exact URL to open:

```text
https://vuer.ai?ws=wss://<selected-ip>:8012&grid=False
```

Selection order is intentionally runtime-based so DHCP changes do not require a
launcher edit:

1. `XR_TELEOP_VUER_IP`, when explicitly set and a usable IPv4 address;
2. the global IPv4 address on the default-route interface (the current robot
   Wi-Fi interface is `wlxfc23cd929ddc`, currently `10.22.16.110/20`);
3. the global IPv4 address on `tailscale0` (currently `100.126.188.19`).

`XR_TELEOP_VUER_INTERFACE` can explicitly select an interface when the default
route is not the Wi-Fi path. The internal robot address (`192.168.123.164`) is
not rewritten or removed; it remains a valid TLS SAN when the certificate is
renewed, but it is not used as the Quest endpoint unless explicitly selected.

## TLS preflight

The launcher checks the selected IP against `XR_TELEOP_CERT` with
`openssl x509 -checkip` and stops before starting services if that IP is absent
from the certificate SAN. This is deliberate: advertising an IP without its SAN
would be a no-ship TLS configuration for Quest.

Inspect the deployed certificate without changing it:

```bash
openssl x509 -in /home/unitree/.config/xr_teleoperate/cert.pem -noout -text \
  | grep -A1 'Subject Alternative Name'
openssl x509 -in /home/unitree/.config/xr_teleoperate/cert.pem -noout -checkip 10.22.16.110
```

## Explicit certificate renewal only

Do **not** generate or overwrite certificates from the launcher. If the
preflight reports a missing SAN, use the existing approved CA/key flow from the
repository README, with a reviewed extension file containing every connection
address that must remain valid:

```ini
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
IP.1 = 10.22.16.110
IP.2 = 100.126.188.19
IP.3 = 192.168.123.164
```

Before replacing the deployed certificate, verify the new artifact without
copying it into the robot configuration:

```bash
openssl x509 -in /path/to/new-cert.pem -noout -checkip 10.22.16.110
openssl x509 -in /path/to/new-cert.pem -noout -checkip 100.126.188.19
openssl x509 -in /path/to/new-cert.pem -noout -checkip 192.168.123.164
```

Certificate replacement and Quest trust-store updates are operational changes
and are intentionally outside this launcher change.
