import pandas as pd

read_file = pd.read_csv('aws_final_repo_3.csv')
total = len(read_file)

new_file = read_file[0:total//2]
new_file.to_csv('aws_final_repo_3_0.csv', index=False)

new_file = read_file[total//2:]
new_file.to_csv('aws_final_repo_3_1.csv', index=False)


read_file = pd.read_csv('aws_final_repo_4.csv')
total     = len(read_file)

new_file = read_file[0:total//2]
new_file.to_csv('aws_final_repo_4_0.csv', index=False)

new_file = read_file[total//2:]
new_file.to_csv('aws_final_repo_4_1.csv', index=False)
