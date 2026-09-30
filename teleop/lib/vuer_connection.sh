#!/usr/bin/env bash
# Shared Vuer/Quest connection selection. Safe to source from launchers and tests.

xr_teleop_is_usable_ipv4() {
    local address=$1 octet
    local -a octets

    [[ $address =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || return 1
    IFS=. read -r -a octets <<< "$address"
    ((${#octets[@]} == 4)) || return 1
    for octet in "${octets[@]}"; do
        [[ $octet == 0 || $octet != 0* ]] || return 1
        ((10#$octet <= 255)) || return 1
    done

    # Reject addresses that cannot be used by a remote Quest client.
    ((10#${octets[0]} != 0 && 10#${octets[0]} != 127 && 10#${octets[0]} < 224)) || return 1
    ! ((10#${octets[0]} == 169 && 10#${octets[1]} == 254))
}

xr_teleop_ipv4_on_interface() {
    local interface=$1 address
    [[ -n $interface ]] || return 1
    address=$(ip -4 -o addr show dev "$interface" scope global 2>/dev/null |
        awk '$3 == "inet" {split($4, cidr, "/"); print cidr[1]; exit}')
    xr_teleop_is_usable_ipv4 "$address" || return 1
    printf '%s\n' "$address"
}

xr_teleop_default_route_interface() {
    ip -4 route show default 2>/dev/null |
        awk '$1 == "default" {for (i = 1; i <= NF; i++) if ($i == "dev" && i < NF) {print $(i + 1); exit}}'
}

xr_teleop_select_vuer_ip() {
    local route_interface address

    if [[ -n ${XR_TELEOP_VUER_IP:-} ]]; then
        if ! xr_teleop_is_usable_ipv4 "$XR_TELEOP_VUER_IP"; then
            echo "XR_TELEOP_VUER_IP must be a valid IPv4 address reachable by the Quest: ${XR_TELEOP_VUER_IP}" >&2
            return 1
        fi
        export XR_TELEOP_VUER_IP
        return 0
    fi

    route_interface=${XR_TELEOP_VUER_INTERFACE:-$(xr_teleop_default_route_interface)}
    if address=$(xr_teleop_ipv4_on_interface "$route_interface"); then
        export XR_TELEOP_VUER_IP=$address
        return 0
    fi

    # A route-less robot can still be reached through the Tailscale interface.
    if address=$(xr_teleop_ipv4_on_interface tailscale0); then
        export XR_TELEOP_VUER_IP=$address
        return 0
    fi

    echo "No valid Wi-Fi or Tailscale IPv4 address found. Set XR_TELEOP_VUER_IP explicitly or restore the network route." >&2
    return 1
}

xr_teleop_quest_url() {
    local address=$1
    xr_teleop_is_usable_ipv4 "$address" || return 1
    printf 'https://vuer.ai?ws=wss://%s:8012&grid=False\n' "$address"
}

xr_teleop_verify_cert_san() {
    local cert=$1 address=$2
    if [[ ! -r $cert ]]; then
        echo "Vuer TLS certificate is unreadable: $cert" >&2
        return 1
    fi
    if ! command -v openssl >/dev/null 2>&1; then
        echo "openssl is required to verify the Vuer TLS certificate SAN." >&2
        return 1
    fi
    if ! openssl x509 -in "$cert" -noout -checkip "$address" >/dev/null 2>&1; then
        echo "Vuer TLS certificate $cert does not contain IP SAN $address. Renew it explicitly; see docs/vuer-quest-networking.md." >&2
        return 1
    fi
}
