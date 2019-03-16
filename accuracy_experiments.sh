########################################
################ LinUCB ################
########################################
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
python src/main.py --algo=lin_ucb >> lin_ucb_stats.txt
########################################
############### Thompson ###############
########################################
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt
python src/main.py --algo=thompson >> thompson_stats.txt

########################################
################ LASSO #################
########################################
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
python src/main.py --algo=lasso >> lasso_stats.txt
########################################
############ MWU (Thompson) ############
########################################
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
python src/main.py --algo=mwu --expert_type=thompson >> mwu_thompson_stats.txt
########################################
############## MWU (LASSO) #############
########################################
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt
python src/main.py --algo=mwu --expert_type=lasso >> mwu_lasso_stats.txt

# Output stats
python src/stats.py