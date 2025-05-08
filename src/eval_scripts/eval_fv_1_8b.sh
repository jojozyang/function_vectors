datasets=(
    'person-instrument'
    'prev_item'
    'person-sport' 
    'product-company'
    'person-occupation'
    'present-past' 
    
)

cd ../

for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    python evaluate_inst_function_vector.py --dataset_name="${d_name}" \
    --save_path_root="/oscar/data/epavlick/zyang220/results/function_vector_instructions" \
    --model_name='meta-llama/Llama-3.1-8B-Instruct' \
    --n_top_heads=20 --device='cuda:0' \
    --universal_set="True"
done
sleep 5