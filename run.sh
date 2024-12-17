# python llm_api.py --model "text-davinci-003"
# python llm_api.py --model "text-davinci-003" --small_model CNN --task charge --use_split_fact True
# python llm_api.py --model "text-davinci-003" --small_model CNN --task article --use_split_fact True --dataset "cjo22"
# python llm_api.py --model "text-davinci-003" --small_model CNN --task penalty --dataset "cail18"
# python llm_api.py --model "gpt-3.5-turbo" --small_model CNN --task penalty --use_split_fact True --dataset "cail18"

# python llm_api.py --model "gpt-3.5-turbo" --small_model BERT --task article --use_split_fact True --dataset "cjo22"


# TASK: article/charge/penalty, TRAIN_SIZE: 6w/20w/50w
TASK=article
TRAIN_SIZE=6w

nohup python llm_api.py \
    --model "qwen-2.5-32b" \
    --small_model BERTC \
    --task ${TASK}  \
    --use_split_fact True \
    --dataset "cjo22" \
    --train_size ${TRAIN_SIZE} > run_${TASK}_${TRAIN_SIZE}_popt.log 2>&1 &

# python llm_api.py \
#     --model "qwen-2.5-32b" \
#     --small_model BERTC \
#     --task ${TASK}  \
#     --use_split_fact True \
#     --dataset "cjo22" \
#     --train_size ${TRAIN_SIZE}
