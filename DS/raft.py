import random
import time
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 'Agg'后端适合在非交互式环境下保存图片，如在PyCharm的终端运行脚本时
import matplotlib.pyplot as plt
import os


class RaftNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.state = "Follower"
        self.current_term = 0
        self.voted_for = None
        self.log = []  # List of (term, command)
        self.election_timeout = random.uniform(150, 300) / 1000  # 150-300 ms
        self.last_heartbeat_time = time.time()
        self.is_failed = False

        # 新增日志提交相关属性
        self.commit_index = 0  # 已提交的日志索引
        self.last_applied = 0  # 已应用到状态机的日志索引
        self.next_index = {}  # 领导者：下一个要发送给每个节点的日志索引
        self.match_index = {}  # 领导者：每个节点复制的最后日志索引

    def handle_heartbeat(self, leader_id, term):
        if term >= self.current_term:
            self.state = "Follower"
            self.leader_id = leader_id
            self.last_heartbeat_time = time.time()
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None

    def start_election(self, all_nodes):
        if self.state == "Follower" and (time.time() - self.last_heartbeat_time) > self.election_timeout:
            self.state = "Candidate"
            self.current_term += 1
            self.voted_for = self.node_id
            vote_count = 1
            for other_node in all_nodes:
                if other_node.node_id != self.node_id and not other_node.is_failed:
                    granted, _ = other_node.handle_vote_request(self.current_term, self.node_id,
                                                                len(self.log),
                                                                self.log[-1][0] if self.log else 0)
                    if granted:
                        vote_count += 1
            if vote_count > len(all_nodes) / 2:
                self.become_leader(all_nodes)

    def handle_vote_request(self, candidate_term, candidate_id, last_log_index, last_log_term):
        if candidate_term < self.current_term:
            return False, self.current_term
        if (self.voted_for is None or self.voted_for == candidate_id) and self.log_is_up_to_date(last_log_index,
                                                                                                 last_log_term):
            self.voted_for = candidate_id
            if candidate_term > self.current_term:
                self.current_term = candidate_term
            return True, self.current_term
        return False, self.current_term

    def log_is_up_to_date(self, candidate_index, candidate_term):
        my_last_index = len(self.log)
        my_last_term = self.log[-1][0] if self.log else 0
        return (candidate_term > my_last_term) or (candidate_term == my_last_term and candidate_index >= my_last_index)

    def become_leader(self, all_nodes):
        self.state = "Leader"
        for other_node in all_nodes:
            if other_node.node_id != self.node_id and not other_node.is_failed:
                other_node.handle_heartbeat(self.node_id, self.current_term)
        # 初始化日志复制索引（新增）
        self.next_index = {node.node_id: len(self.log) + 1 for node in all_nodes if node != self}
        self.match_index = {node.node_id: 0 for node in all_nodes if node != self}

    def send_heartbeat(self, all_nodes):
        if self.state == "Leader":
            for other_node in all_nodes:
                if other_node.node_id != self.node_id and not other_node.is_failed:
                    other_node.handle_heartbeat(self.node_id, self.current_term)

    def handle_client_request(self, request, all_nodes):
        if self.state == "Leader":
            term = self.current_term
            self.log.append((term, request))
            # 触发日志复制（新增）
            self.replicate_logs(all_nodes)

    # 新增日志复制与提交方法
    def replicate_logs(self, all_nodes):
        for node in all_nodes:
            if node == self or node.is_failed:
                continue
            idx = self.next_index.get(node.node_id, len(self.log))  # 获取下一个要发送的索引
            if idx <= len(self.log):
                prev_log_index = idx - 1
                prev_log_term = self.log[prev_log_index][0] if prev_log_index > 0 else 0
                log_entry = self.log[prev_log_index]
                # 跟随者追加日志（返回是否成功）
                success = node.append_log(prev_log_index, prev_log_term, log_entry)
                if success:
                    self.next_index[node.node_id] = idx + 1  # 成功则递增下一个索引
                    self.match_index[node.node_id] = idx  # 更新已复制的最后索引
                    self.commit_log(all_nodes)  # 检查是否可提交
                else:
                    self.next_index[node.node_id] -= 1  # 失败则回退索引，下次重试

    # 跟随者追加日志（支持前驱日志检查，处理冲突）
    def append_log(self, prev_log_index, prev_log_term, log_entry):
        if prev_log_index > len(self.log):
            return False  # 索引越界，拒绝
        # 检查前驱日志一致性，不一致则截断
        if prev_log_index > 0 and (len(self.log) <= prev_log_index or self.log[prev_log_index - 1][0] != prev_log_term):
            self.log = self.log[:prev_log_index]  # 截断冲突日志
        if prev_log_index == len(self.log):
            self.log.append(log_entry)  # 追加新日志
            return True
        return False

    # 多数确认后提交日志
    def commit_log(self, all_nodes):
        for idx in range(self.commit_index + 1, len(self.log) + 1):
            # 计算确认数（包括领导者自己）
            count = 1  # 领导者自己已确认
            for node_id, mi in self.match_index.items():
                if mi >= idx:
                    count += 1
            if count > len(all_nodes) / 2:  # 超过半数确认
                self.commit_index = idx  # 更新提交索引
                self.last_applied = idx  # 应用到状态机（简化处理）


def simulate_network_delay(delay):
    time.sleep(delay / 1000)


def simulate_node_failure(nodes, failure_rate):
    num_failures = int(len(nodes) * failure_rate)
    failed_nodes = random.sample([n for n in nodes if not n.is_failed], num_failures)
    for node in failed_nodes:
        node.is_failed = True


def run_experiment(network_delay, node_failure_rate):
    start_time = time.time()
    node_count = 10
    all_nodes = [RaftNode(i) for i in range(node_count)]
    simulate_network_delay(network_delay)
    simulate_node_failure(all_nodes, node_failure_rate)

    # 触发选举
    for node in all_nodes:
        node.last_heartbeat_time = time.time() - 1
        node.start_election(all_nodes)

    leader = next((n for n in all_nodes if n.state == "Leader"), None)
    if leader:
        for _ in range(10):
            leader.handle_client_request("Test Request", all_nodes)
            time.sleep(0.01)  # 等待复制

        # 等待所有日志提交（新增）
        while leader.commit_index < len(leader.log):
            time.sleep(0.01)

    end_time = time.time()
    consensus_time = end_time - start_time
    throughput = 10 / consensus_time if consensus_time > 0 else 0
    data_consistency = check_data_consistency(all_nodes)
    return consensus_time, throughput, data_consistency


def check_data_consistency(all_nodes):
    non_failed_nodes = [n for n in all_nodes if not n.is_failed]
    if not non_failed_nodes:
        return True
    reference_log = non_failed_nodes[0].log
    for node in non_failed_nodes[1:]:
        if node.log != reference_log:
            return False
    return True


def analyze_data(raft_data):
    raft_consensus_times = np.array([data[0] for data in raft_data])
    raft_throughputs = np.array([data[1] for data in raft_data])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(raft_consensus_times, label='Raft')
    plt.title('Consensus Time')
    plt.xlabel('Experiment')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(raft_throughputs, label='Raft')
    plt.title('Throughput')
    plt.xlabel('Experiment')
    plt.ylabel('Requests/s')
    plt.legend()

    plt.tight_layout()

    save_path = "D:/Code/CCBDA/DS/raft_analysis.png"
    # 检查路径是否存在，如果不存在则创建
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Raft图表已保存至: {save_path}')
    except Exception as e:
        print(f"保存图片时出错: {e}")


# plt.show()


class TestRaft(unittest.TestCase):
    def test_raft_election(self):
        node = RaftNode(0)
        all_nodes = [node]
        node.last_heartbeat_time = time.time() - 1
        node.start_election(all_nodes)
        self.assertEqual(node.state, "Leader")

    # 新增日志提交测试（示例）
    def test_log_commit(self):
        leader = RaftNode(0)
        follower = RaftNode(1)
        all_nodes = [leader, follower]

        leader.become_leader(all_nodes)
        leader.handle_client_request("a", all_nodes)

        # 模拟跟随者确认日志
        follower.append_log(1, leader.current_term, (leader.current_term, "a"))
        leader.match_index[follower.node_id] = 1
        leader.commit_log(all_nodes)

        self.assertEqual(leader.commit_index, 1, "日志应被提交")


if __name__ == '__main__':
    raft_results = []

    for _ in range(50):
        network_delay = random.randint(10, 500)
        failure_rate = random.uniform(0.1, 0.5)

        raft_res = run_experiment(network_delay, failure_rate)
        raft_results.append(raft_res)

    analyze_data(raft_results)

    unittest.main(argv=[''], exit=False)