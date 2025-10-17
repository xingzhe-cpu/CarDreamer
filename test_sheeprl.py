from sheeprl.envs.cardreamer import CarDreamerWrapper
w = CarDreamerWrapper(task_name="carla_four_lane")
print(w.action_space)    # 也应是 Discrete(n)
