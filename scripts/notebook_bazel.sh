#!/usr/bin/env bash

# Shared Bazel/Bazelisk discovery for fresh Kaggle and Colab runtimes.
clqr_notebook_bazel() {
  local repo_dir="$1"
  local notebook_work_dir="$2"
  local bazel_command="${CLQR_BAZEL:-}"
  local required_bazel_version
  required_bazel_version="$(<"${repo_dir}/.bazelversion")"

  if [[ -z "${bazel_command}" ]] && command -v bazelisk >/dev/null 2>&1; then
    bazel_command="$(command -v bazelisk)"
  fi
  if [[ -z "${bazel_command}" ]] && command -v bazel >/dev/null 2>&1 &&
     [[ "$(bazel --version 2>/dev/null)" == \
        "bazel ${required_bazel_version}" ]]; then
    bazel_command="$(command -v bazel)"
  fi
  if [[ -z "${bazel_command}" ]]; then
    local bazelisk_version=1.29.0
    local bazelisk_sha256=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
    local bazelisk_asset
    case "$(uname -m)" in
      x86_64|amd64) bazelisk_asset=bazelisk-linux-amd64 ;;
      *)
        echo "automatic Bazelisk bootstrap supports x86-64 Linux only; set CLQR_BAZEL" >&2
        return 2
        ;;
    esac
    local tool_dir="${notebook_work_dir}/clqr_tools"
    bazel_command="${tool_dir}/bazelisk-${bazelisk_version}"
    if [[ ! -x "${bazel_command}" ]] ||
       ! printf '%s  %s\n' "${bazelisk_sha256}" "${bazel_command}" |
         sha256sum --check --status; then
      mkdir -p "${tool_dir}"
      local bazelisk_tmp="${bazel_command}.tmp"
      local bazelisk_url
      bazelisk_url="https://github.com/bazelbuild/bazelisk/releases/download"
      curl --fail --location --silent --show-error \
        "${bazelisk_url}/v${bazelisk_version}/${bazelisk_asset}" \
        --output "${bazelisk_tmp}"
      printf '%s  %s\n' "${bazelisk_sha256}" "${bazelisk_tmp}" |
        sha256sum --check --status
      chmod +x "${bazelisk_tmp}"
      mv "${bazelisk_tmp}" "${bazel_command}"
    fi
  fi
  printf '%s\n' "${bazel_command}"
}
