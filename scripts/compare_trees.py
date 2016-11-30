tree_dir_rl   = "./checkpoints/graphs/summaries.txt"
tree_dir_norl = "./checkpoints/graphs-norl/summaries.txt"

with open(tree_dir_rl) as f_rl, open(tree_dir_norl) as f_norl:
    total_rl_hamm, total_norl_hamm = 0, 0
    for i, (line_rl, line_norl) in enumerate(zip(f_rl, f_norl)):
        if i == 0: continue
        rl_id, rl_hamm = line_rl.split(',')[:2]
        norl_id, norl_hamm = line_norl.split(',')[:2]
        print(rl_id, rl_hamm, norl_hamm, int(rl_hamm) - int(norl_hamm))
        total_rl_hamm += int(rl_hamm)
        total_norl_hamm += int(norl_hamm)

    print("RL, NORL")
    print(total_rl_hamm, total_norl_hamm)
    print(total_rl_hamm/float(i), total_norl_hamm/float(i))
