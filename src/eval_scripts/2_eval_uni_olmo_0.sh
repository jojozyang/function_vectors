#!/bin/bash
#datasets=('antonym')
datasets=(
    # 'person-instrument'
    # 'prev_item'
    # 'person-sport' 
    # 'national_parks'
    # 'product-company'
    # 'singular-plural'
    # 'person-occupation'
    # 'present-past'
    # 'next_item'
    # 'landmark-country'
    # 'lowercase_last_letter'
    'english-french'
    # 'english-german'
    # 'country-currency'
    # 'lowercase_first_letter'
    # 'capitalize_first_letter'
    # 'park-country'
    # 'antonym'
    # 'capitalize'
    # 'sentiment'
    # 'english-spanish'
    # 'synonym'
    # 'country-capital'
)
cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python evaluate_function_vector.py --dataset_name="${d_name}" \
    --save_path_root="/oscar/data/epavlick/zyang220/results/function_vector" \
    --model_name='allenai/OLMo-2-1124-7B-Instruct' \
    --n_top_heads=20 --device='cuda:0' \
    --universal_set="True" --compute_baseline="False"
done
sleep 5
