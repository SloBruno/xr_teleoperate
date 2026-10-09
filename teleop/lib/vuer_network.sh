#!/usr/bin/env bash
# Launcher helper: show the robot's current network IPs, the exact Quest URL,
# and keep the TLS certificate SAN in sync with the Wi-Fi IP.
# Never blocks the launch: xr_net_announce always returns 0. Runs once, before
# any actuator process starts (no I/O in the control loop).

XR_NET_PORT=${XR_NET_PORT:-8012}

xr_net_valid_ipv4() {
    local a=$1 o
    [[ $a =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || return 1
    local IFS=.
    for o in $a; do ((10#$o <= 255)) || return 1; done
}

# "<iface> <ip>" of the default route (empty when there is none).
xr_net_default_route() {
    ip -4 route get 1.1.1.1 2>/dev/null | awk '
        {for (i = 1; i <= NF; i++) { if ($i == "dev") d = $(i+1); if ($i == "src") s = $(i+1) }}
        END { if (d != "" && s != "") print d, s }'
}

# "<iface> <ip>" per line, skipping loopback/docker/bridge/veth.
xr_net_list_ipv4() {
    ip -4 -o addr show 2>/dev/null | awk '
        {for (i = 1; i <= NF; i++) if ($i == "inet") { split($(i+1), c, "/"); print $2, c[1] }}' |
        awk '$1 != "lo" && $1 !~ /^(docker|br-|veth|virbr)/'
}

xr_net_cert_sans() {  # prints "IP:x" / "DNS:y" per line
    openssl x509 -in "$1" -noout -ext subjectAltName 2>/dev/null |
        grep -oE '(IP Address|DNS):[^,[:space:]]+' | sed 's/^IP Address:/IP:/'
}

xr_net_label() {  # iface -> human label
    case $1 in
        tailscale*) echo "Tailscale - o Quest NÃO alcança" ;;
        enP8p1s0) echo "rede interna G1" ;;
        *) echo "$1" ;;
    esac
}

xr_net_regen_cert() {  # $1 ip, $2 cert, $3 key, $4.. current SAN entries
    local ip=$1 cert=$2 key=$3; shift 3
    local -a entries=("IP:$ip")
    local e seen=" IP:$ip "
    for e in "$@" IP:100.126.188.19 IP:192.168.123.164 DNS:unitree-g1-nx; do
        [[ $seen == *" $e "* ]] && continue
        seen+="$e "; entries+=("$e")
    done
    local san ts dir tmpc tmpk
    san=$(IFS=,; echo "${entries[*]}")
    ts=$(date +%Y%m%d-%H%M%S)
    dir=$(dirname -- "$cert")
    mkdir -p "$dir" || return 1
    tmpc=$(mktemp "$dir/.cert.XXXXXX") || return 1
    tmpk=$(mktemp "$dir/.key.XXXXXX") || { rm -f "$tmpc"; return 1; }
    if ! openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=unitree-g1-nx" \
            -addext "subjectAltName=$san" -keyout "$tmpk" -out "$tmpc" >/dev/null 2>&1; then
        rm -f "$tmpc" "$tmpk"; return 1
    fi
    chmod 600 "$tmpk" && chmod 644 "$tmpc" || { rm -f "$tmpc" "$tmpk"; return 1; }
    [[ -e $cert ]] && { cp -p "$cert" "$cert.bak.$ts" || { rm -f "$tmpc" "$tmpk"; return 1; }; }
    [[ -e $key ]] && { cp -p "$key" "$key.bak.$ts" || { rm -f "$tmpc" "$tmpk"; return 1; }; }
    mv -f "$tmpk" "$key" && mv -f "$tmpc" "$cert" || return 1
}

xr_net_announce() {
    local cert=${XR_TELEOP_CERT:-/home/unitree/.config/xr_teleoperate/cert.pem}
    local key=${XR_TELEOP_KEY:-${cert%cert.pem}key.pem}
    local state_dir=${XR_TELEOP_STATE_DIR:-/home/unitree/.local/state/xr_teleoperate}
    local route iface="" ip="" label="Wi-Fi" line sans
    local bar="================================================================"

    route=$(xr_net_default_route)
    if [[ -n ${XR_TELEOP_VUER_IP:-} ]]; then
        if xr_net_valid_ipv4 "$XR_TELEOP_VUER_IP"; then
            ip=$XR_TELEOP_VUER_IP; label="definido por XR_TELEOP_VUER_IP"
        else
            echo "AVISO: XR_TELEOP_VUER_IP='$XR_TELEOP_VUER_IP' inválido; ignorando e detectando." >&2
        fi
    fi
    if [[ -z $ip && -n $route ]]; then
        read -r iface ip <<<"$route"
        [[ $iface == tailscale* || $iface == enP8p1s0 ]] && label="rota default via $iface (NÃO é o Wi-Fi)"
    fi

    echo "$bar"
    if [[ -n $ip ]]; then
        echo "IP do robô na rede ($label): $ip"
        echo "URL recomendada (cliente local 0.0.60): https://$ip:$XR_NET_PORT?grid=False"
        echo "Alternativa hospedada: https://vuer.ai?ws=wss://$ip:$XR_NET_PORT&grid=False (versão não controlada)"
    else
        echo "AVISO: SEM rota default / sem IP Wi-Fi detectado. O Quest não vai conseguir conectar."
        echo "       Conecte o Wi-Fi ou defina XR_TELEOP_VUER_IP=<ip>. Seguindo mesmo assim."
    fi
    local others=""
    while read -r line; do
        [[ -z $line ]] && continue
        local n=${line%% *} a=${line#* }
        [[ $a == "$ip" ]] && continue
        others+="  - $a  ($(xr_net_label "$n"))"$'\n'
    done < <(xr_net_list_ipv4)
    [[ -n $others ]] && { echo "Outras interfaces:"; printf '%s' "$others"; }

    if [[ -n $ip ]]; then
        if ! command -v openssl >/dev/null 2>&1; then
            echo "AVISO: openssl ausente; não verifiquei o certificado TLS."
        else
            sans=$(xr_net_cert_sans "$cert")
            if [[ -e $cert && $'\n'$sans$'\n' == *$'\n'"IP:$ip"$'\n'* ]]; then
                echo "Certificado TLS: OK (SAN contém $ip)"
            else
                echo "Certificado TLS não contém $ip; regenerando (com backup)..."
                # shellcheck disable=SC2046
                if xr_net_regen_cert "$ip" "$cert" "$key" $sans; then
                    echo "*** certificado regenerado; abra https://$ip:$XR_NET_PORT no Quest e aceite uma vez ***"
                    echo "    (Teleimager/TeleVuer já em execução precisam ser reiniciados para usar o novo certificado)"
                else
                    echo "AVISO: FALHA ao regenerar o certificado TLS; seguindo sem bloquear (Quest pode rejeitar o TLS)."
                fi
            fi
        fi
        if mkdir -p "$state_dir" 2>/dev/null; then
            printf 'https://vuer.ai?ws=wss://%s:%s&grid=False\n' "$ip" "$XR_NET_PORT" >"$state_dir/quest_url" 2>/dev/null || true
        fi
        export XR_TELEOP_VUER_IP=$ip
    fi
    echo "$bar"
    return 0
}
