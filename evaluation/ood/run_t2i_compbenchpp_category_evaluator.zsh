#!/usr/bin/env zsh
set -euo pipefail
set +x

if (( $# != 2 )); then
  print -u2 'usage: run_t2i_compbenchpp_category_evaluator.zsh <category> <official-dir>'
  exit 2
fi
repo_root=${0:A:h:h:h}
repo=${T2ICBPP_EVALUATOR_ROOT:-${repo_root}/evaluation/.assets/t2i-compbench-plusplus}
python=${T2ICBPP_PYTHON:-${repo}/.venv/bin/python}
models=${T2ICBPP_MODEL_ROOT:-${repo}/models}
category=$1
official_dir=${2:A}
case ${category} in
  color|shape|texture|spatial|3d_spatial|numeracy|non_spatial|complex) ;;
  *) print -u2 -- "unsupported category: ${category}"; exit 2 ;;
esac
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export T2ICBPP_BLIP_VQA_MODEL=${models}/model_base_vqa_capfilt_large.pth
export T2ICBPP_CLIP_MODEL=${models}/ViT-B-32.pt
export T2ICBPP_BERT_TOKENIZER=${models}/bert-base-uncased
export HF_HOME=${official_dir}/runtime-cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export XDG_CACHE_HOME=${official_dir}/runtime-cache
export TORCH_HOME=${official_dir}/runtime-cache/torch
# multiprocessing adds pymp-*/listener-* below TMPDIR (AF_UNIX limit: 108 bytes).
scratch_root=${T2ICBPP_SCRATCH_ROOT:-/tmp}
mkdir -p ${scratch_root}
export TMPDIR=$(mktemp -d ${scratch_root}/cb.XXXXXX)
export PYTHONDONTWRITEBYTECODE=1
mkdir -p ${TMPDIR} ${HF_HOME}
run_blip() {
  cd ${repo}/BLIPvqa_eval
  ${python} BLIP_vqa.py --out_dir=${official_dir}
}
run_2d() {
  cd ${repo}/UniDet_eval
  if [[ ${category} == complex ]]; then
    ${python} 2D_spatial_eval.py --outpath=${official_dir} --complex True
  else
    ${python} 2D_spatial_eval.py --outpath=${official_dir}
  fi
}
run_clip() {
  cd ${repo}
  if [[ ${category} == complex ]]; then
    ${python} CLIPScore_eval/CLIP_similarity.py --outpath=${official_dir} --complex True
  else
    ${python} CLIPScore_eval/CLIP_similarity.py --outpath=${official_dir}
  fi
}

case ${category} in
  color|shape|texture) run_blip ;;
  spatial) run_2d ;;
  3d_spatial)
    cd ${repo}/UniDet_eval
    ${python} 3D_spatial_eval.py --outpath=${official_dir}
    ;;
  numeracy)
    cd ${repo}/UniDet_eval
    ${python} numeracy_eval.py --outpath=${official_dir}
    ;;
  non_spatial) run_clip ;;
  complex)
    run_blip
    run_2d
    run_clip
    cd ${repo}
    ${python} 3_in_1_eval/3_in_1.py \
      --outpath=${official_dir} --data_path=${T2ICBPP_COMPLEX_DATA_PATH:-${repo}/examples/dataset}
    ;;
esac
