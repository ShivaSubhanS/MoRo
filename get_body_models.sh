#!/usr/bin/env bash

set -euo pipefail
source functions.sh

# Download body models directly into the MoRo project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
subdir="$SCRIPT_DIR/body_models"
mkdircd "$subdir"
mkdir -p smpl smplx smplh

read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")

download() {
  local domain=$1
  local filename=$2
  local filepath=$filename
  wget --post-data "username=$encoded_email&password=$password" \
    "https://download.is.tue.mpg.de/download.php?domain=${domain}&resume=1&sfile=${filename}" \
    -O "$filepath" --no-check-certificate --continue
}

download_and_extract() {
  local domain=$1
  local filename=$2
  local filepath=$filename
  download "$domain" "$filename"
  extractrm "$filepath"
}

# SMPL
if [[ ! -f smpl/SMPL_NEUTRAL.pkl ]]; then
  download_and_extract smpl SMPL_python_v.1.1.0.zip
  mv SMPL_python_v.1.1.0/smpl/models/basicmodel_*_lbs_10_207_0_v1.1.0.pkl smpl/
  rm -rf SMPL_python_v.1.1.0
  pushd smpl
  ln -sf basicmodel_m_lbs_10_207_0_v1.1.0.pkl SMPL_MALE.pkl
  ln -sf basicmodel_f_lbs_10_207_0_v1.1.0.pkl SMPL_FEMALE.pkl
  ln -sf basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl SMPL_NEUTRAL.pkl
  popd
else
  echo "[skip] SMPL already downloaded"
fi

if [[ ! -f smpl/J_regressor_h36m.npy ]]; then
  wget -c http://visiondata.cis.upenn.edu/spin/data.tar.gz
  extractrm data.tar.gz
  mv data/J_regressor_{extra,h36m}.npy smpl/
  rm -rf data
else
  echo "[skip] J_regressor_h36m.npy already downloaded"
fi

# SMPL-X
if [[ ! -f smplx/SMPLX_NEUTRAL.npz ]]; then
  download_and_extract smplx models_smplx_v1_1.zip
  mv models/smplx/* smplx/
  rm -rf models
else
  echo "[skip] SMPL-X models already downloaded"
fi

# SMPL+H
if [[ ! "$(ls -A smplh)" ]]; then
  download_and_extract mano mano_v1_2.zip
  mv mano_v1_2/models/SMPLH_* smplh/
  rm -rf mano_v1_2
else
  echo "[skip] SMPL+H already downloaded"
fi

