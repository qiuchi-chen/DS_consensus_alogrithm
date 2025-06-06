from graphviz import Digraph

# 创建流程图对象
dot = Digraph(comment='Raft Algorithm Flowchart')

# 添加各个节点，指定形状（矩形或菱形）
dot.node('Initialize', 'Initialize Nodes', shape='rectangle')
dot.node('Heartbeat', 'Receive Heartbeat?', shape='diamond')
dot.node('Follower', 'Maintain Follower State', shape='rectangle')
dot.node('Timeout', 'Election Timeout?', shape='diamond')
dot.node('Candidate', 'Become Candidate', shape='rectangle')
dot.node('Initiate', 'Initiate Election', shape='rectangle')
dot.node('Votes', 'Obtain Majority Votes?', shape='diamond')
dot.node('Leader', 'Become Leader', shape='rectangle')
dot.node('Return', 'Return to Follower State', shape='rectangle')
dot.node('Client', 'Receive Client Request', shape='rectangle')
dot.node('Replicate', 'Replicate Logs to Other Nodes', shape='rectangle')
dot.node('Confirm', 'Majority Nodes Confirm?', shape='diamond')
dot.node('Commit', 'Commit Logs', shape='rectangle')

# 定义节点间的连接关系
dot.edge('Initialize', 'Heartbeat')
dot.edge('Heartbeat', 'Follower', label='Yes')
dot.edge('Heartbeat', 'Timeout', label='No')
dot.edge('Timeout', 'Candidate', label='Yes')
dot.edge('Candidate', 'Initiate')
dot.edge('Initiate', 'Votes')
dot.edge('Votes', 'Leader', label='Yes')
dot.edge('Votes', 'Return', label='No')
dot.edge('Return', 'Heartbeat')
dot.edge('Leader', 'Client')
dot.edge('Client', 'Replicate')
dot.edge('Replicate', 'Confirm')
dot.edge('Confirm', 'Commit', label='Yes')
dot.edge('Confirm', 'Replicate', label='No')
dot.edge('Commit', 'Client')

# 渲染并生成图片（会自动打开查看）
dot.render('raft_flowchart', view=True)