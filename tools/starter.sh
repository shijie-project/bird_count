#!/usr/bin/env bash
# starter.sh - launches one of the dev/data tools.
#
# Usage:
#   ./tools/starter.sh streamlit                          # dataset viewer
#   ./tools/starter.sh label-studio                       # annotation server (local)
#   ./tools/starter.sh label-studio <ngrok-domain>        # annotation server (via ngrok)
#
# Examples:
#   ./tools/starter.sh label-studio
#   ./tools/starter.sh label-studio obliging-maggot-frank.ngrok-free.app
#
# Always runs from the project root (bird_count/), regardless of CWD.

set -euo pipefail

# Jump to project root (parent of tools/)
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# --- config ---------------------------------------------------------------
LS_PORT="${LS_PORT:-8080}"                                 # Label Studio's own default; override with LS_PORT=... ./starter.sh
LS_DATA_DIR="${LS_DATA_DIR:-$PROJECT_ROOT/../data}"        # ../data relative to project root
LS_LOCAL_FILES_ROOT="${LS_LOCAL_FILES_ROOT:-C:/Shijie_Li/bird_pileup/code/data}"
# --------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $0 {streamlit|label-studio} [ngrok-domain]

Commands:
  streamlit                    Start the dataset viewer (Streamlit UI)
  label-studio [domain]        Start Label Studio. Pass an ngrok domain
                               (e.g. foo.ngrok-free.app) to enable public access.

Env overrides:
  LS_PORT                      Port for Label Studio (default: 8080)
  LS_DATA_DIR                  Data directory (default: ../data)
  LS_LOCAL_FILES_ROOT          Local files document root for LS
EOF
}

cmd="${1:-}"
ngrok_domain="${2:-}"

case "$cmd" in
    streamlit)
        exec streamlit run tools/visualize_data.py
        ;;

    label-studio | label_studio)
        export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
        export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$LS_LOCAL_FILES_ROOT"

        if [[ -n "$ngrok_domain" ]]; then
            # strip any accidental https:// the user pasted in
            ngrok_domain="${ngrok_domain#https://}"
            ngrok_domain="${ngrok_domain#http://}"
            ngrok_domain="${ngrok_domain%/}"

            export CSRF_TRUSTED_ORIGINS="https://${ngrok_domain}"
            export LABEL_STUDIO_HOST="https://${ngrok_domain}"

            echo "==> Label Studio will trust origin: https://${ngrok_domain}"
            echo "==> Start ngrok in another terminal:"
            echo "    ngrok http --url=${ngrok_domain} ${LS_PORT}"
            echo
        fi

        exec label-studio start --data-dir "$LS_DATA_DIR" -p "$LS_PORT"
        ;;

    -h | --help)
        usage
        exit 0
        ;;

    "")
        usage
        exit 1
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        usage >&2
        exit 2
        ;;
esac