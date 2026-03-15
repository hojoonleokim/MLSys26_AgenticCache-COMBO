port=12077
pkill -f -9 "port\ $port"
python3 challenge.py \
--port $port \
--experiment_name combo-cook-agent1 \
--num_propose 3 \
--plan_horizon 3 \
--plan_beam 3 \
--task cook \
--data_prefix dataset/ \
--data_path test.json \
--agents_algo genco_agent genco_agent \
--screen_size 336 \
--start_id 0 \
--num_runs 5 \
--max_steps 60 \
--lm_source llava_server \
--temperature 0.7 \
--only_propose \
--proposer_lm_id gpt-5 \
--guidance_weight 5