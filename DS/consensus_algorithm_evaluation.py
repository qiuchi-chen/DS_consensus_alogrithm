import random
import time
import unittest
import matplotlib.pyplot as plt
import numpy as np



class RaftNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.state = "Follower"
        self.leader_id = None
        self.log = []
        self.election_timeout = random.uniform(1, 3)
        self.last_heartbeat_time = time.time()
        self.voted_for = None
        self.vote_count = 0

    def handle_heartbeat(self, leader_id):
        if self.state != "Leader":
            self.state = "Follower"
            self.leader_id = leader_id
            self.last_heartbeat_time = time.time()
            self.voted_for = None

    def start_election(self, all_nodes):
        if self.state == "Follower" and (time.time() - self.last_heartbeat_time) > self.election_timeout:
            self.state = "Candidate"
            self.vote_count = 1
            self.voted_for = self.node_id
            for other_node in all_nodes:
                if other_node.node_id != self.node_id:
                    self.send_vote_request(other_node, all_nodes)

    def send_vote_request(self, other_node, all_nodes):
        if other_node.handle_vote_request(self.node_id):
            self.vote_count += 1
            if self.vote_count > len(all_nodes) / 2:
                self.become_leader(all_nodes)

    def handle_vote_request(self, candidate_id):
        if self.state == "Follower" and self.voted_for is None:
            self.voted_for = candidate_id
            return True
        return False

    def become_leader(self, all_nodes):
        self.state = "Leader"
        self.voted_for = None
        for other_node in all_nodes:
            if other_node.node_id != self.node_id:
                self.send_heartbeat(other_node)

    def send_heartbeat(self, other_node):
        other_node.handle_heartbeat(self.node_id)

    def handle_client_request(self, request, all_nodes):
        if self.state == "Leader":
            self.log.append(request)
            for other_node in all_nodes:
                if other_node.node_id != self.node_id:
                    self.replicate_log(other_node, request)

    def replicate_log(self, other_node, log_entry):
        other_node.append_log(log_entry)

    def append_log(self, log_entry):
        self.log.append(log_entry)


class PBFTClient:
    def __init__(self):
        self.request_id = 0

    def send_request(self, request, primary_node, all_pbft_nodes):
        self.request_id += 1
        primary_node.handle_request(request, self.request_id, all_pbft_nodes)


class PBFTNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.role = "Backup"
        self.requests = {}
        self.prepared = {}
        self.confirmed = {}
        self.executed_requests = set()  # 新增执行记录集合
        self.f = 1
        self.is_failed = False

    def handle_request(self, request, request_id, all_pbft_nodes):
        if self.is_failed:
            return
        print(f"Node {self.node_id} received request {request_id}")

        # 主节点立即记录请求（关键修复）
        if self.role == "Primary":
            self.requests[request_id] = request

        if self.role == "Primary":
            self.broadcast_pre_prepare(request, request_id, all_pbft_nodes)

    def broadcast_pre_prepare(self, request, request_id, all_pbft_nodes):
        if self.is_failed:
            return
        # 包含主节点自己的处理（重要修复）
        for node in all_pbft_nodes:
            if not node.is_failed:
                node.handle_pre_prepare(request, request_id, self.node_id, all_pbft_nodes)

    def handle_pre_prepare(self, request, request_id, primary_id, all_pbft_nodes):
        if self.is_failed:
            return
        if self.role in ["Backup", "Primary"] and not self.is_failed:
            if request_id not in self.requests:
                self.requests[request_id] = request
                print(f"Node {self.node_id} stored request {request_id}")
            self.broadcast_prepare(request_id, primary_id, all_pbft_nodes)

    def broadcast_prepare(self, request_id, primary_id, all_pbft_nodes):
        if self.is_failed:
            return
        for node in all_pbft_nodes:
            if not node.is_failed:
                node.handle_prepare(request_id, primary_id, all_pbft_nodes)

    def handle_prepare(self, request_id, primary_id, all_pbft_nodes):
        if self.is_failed:
            return
        if self.role in ["Backup", "Primary"] and not self.is_failed:
            self.prepared[request_id] = self.prepared.get(request_id, 0) + 1
            if self.prepared[request_id] > 2 * self.f:
                self.broadcast_commit(request_id, all_pbft_nodes)

    def broadcast_commit(self, request_id, all_pbft_nodes):
        if self.is_failed:
            return
        for node in all_pbft_nodes:
            if not node.is_failed:
                node.handle_commit(request_id, all_pbft_nodes)

    def handle_commit(self, request_id, all_pbft_nodes):
        if self.is_failed:
            return
        if self.role in ["Backup", "Primary"] and not self.is_failed:
            self.confirmed[request_id] = self.confirmed.get(request_id, 0) + 1
            if self.confirmed[request_id] > 2 * self.f:
                self._execute_request(request_id)

    def _execute_request(self, request_id):  # 私有方法防止重复执行
        if request_id in self.executed_requests:
            return
        try:
            request = self.requests[request_id]
            print(f"Node {self.node_id} executed request: {request}")
            self.executed_requests.add(request_id)
        except KeyError:
            print(f"Node {self.node_id} failed to execute unknown request {request_id}")


def simulate_network_delay(delay):
    time.sleep(delay / 1000)


def simulate_node_failure(nodes, failure_rate):
    num_failures = int(len(nodes) * failure_rate)
    failed_nodes = random.sample(nodes, num_failures)
    for node in failed_nodes:
        node.is_failed = True


def simulate_malicious_nodes(nodes, malicious_rate):
    num_malicious = int(len(nodes) * malicious_rate)
    malicious_nodes = random.sample(nodes, num_malicious)
    for node in malicious_nodes:
        def malicious_send(*args, **kwargs):
            if random.random() < 0.5:
                print(f"Malicious node {node.node_id} sending wrong message")
                fake_request = "FAKE_" + str(random.randint(1000, 9999))
                node.broadcast_pre_prepare(fake_request, kwargs['request_id'], kwargs['all_pbft_nodes'])

        node.broadcast_pre_prepare = malicious_send


def run_experiment(algorithm, network_delay, node_failure_rate, malicious_node_rate):
    start_time = time.time()

    if algorithm == "Raft":
        all_nodes = [RaftNode(i) for i in range(5)]
        simulate_network_delay(network_delay)
        simulate_node_failure(all_nodes, node_failure_rate)

        for node in all_nodes:
            node.election_timeout = 0.1
            node.last_heartbeat_time = time.time() - 1
            node.start_election(all_nodes)

        leader = next((n for n in all_nodes if n.state == "Leader"), None)
        if leader:
            for _ in range(10):
                leader.handle_client_request("Test Request", all_nodes)

    elif algorithm == "PBFT":
        all_pbft_nodes = [PBFTNode(i) for i in range(5)]
        all_pbft_nodes[0].role = "Primary"
        client = PBFTClient()
        simulate_network_delay(network_delay)
        simulate_node_failure(all_pbft_nodes, node_failure_rate)
        simulate_malicious_nodes(all_pbft_nodes, malicious_node_rate)

        for _ in range(10):
            client.send_request("Test Request", all_pbft_nodes[0], all_pbft_nodes)
            time.sleep(0.01)  # 添加微小延迟确保消息处理

    end_time = time.time()
    consensus_time = end_time - start_time
    throughput = 10 / consensus_time if consensus_time > 0 else 0
    data_consistency = True
    return consensus_time, throughput, data_consistency


def analyze_data(raft_data, pbft_data):
    raft_consensus_times = np.array([data[0] for data in raft_data])
    raft_throughputs = np.array([data[1] for data in raft_data])
    pbft_consensus_times = np.array([data[0] for data in pbft_data])
    pbft_throughputs = np.array([data[1] for data in pbft_data])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(raft_consensus_times, label='Raft')
    plt.plot(pbft_consensus_times, label='PBFT')
    plt.title('Consensus Time Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(raft_throughputs, label='Raft')
    plt.plot(pbft_throughputs, label='PBFT')
    plt.title('Throughput Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Requests/s')
    plt.legend()

    plt.tight_layout()
    plt.show()


class TestRaft(unittest.TestCase):
    def test_raft_election(self):
        node = RaftNode(0)
        all_nodes = [node]
        node.election_timeout = 0.1
        node.last_heartbeat_time = time.time() - 1
        node.start_election(all_nodes)
        self.assertEqual(node.state, "Candidate")


class TestPBFT(unittest.TestCase):
    def setUp(self):
        self.all_pbft_nodes = [PBFTNode(i) for i in range(5)]
        self.all_pbft_nodes[0].role = "Primary"
        self.client = PBFTClient()
        # 确保测试稳定性
        for node in self.all_pbft_nodes:
            node.is_failed = False

    def test_pbft_request_processing(self):
        self.client.send_request("Sample Request", self.all_pbft_nodes[0], self.all_pbft_nodes)
        # 等待消息处理完成
        time.sleep(0.1)
        self.assertIn(1, self.all_pbft_nodes[0].requests)
        # 验证至少3个节点执行了请求（容错要求）
        executed_count = sum(1 for node in self.all_pbft_nodes if 1 in node.executed_requests)
        self.assertGreaterEqual(executed_count, 3)


if __name__ == '__main__':
    raft_results = []
    pbft_results = []

    for _ in range(50):
        network_delay = random.randint(10, 500)
        failure_rate = random.uniform(0.1, 0.5)
        malicious_rate = random.uniform(0.05, 0.2)

        raft_res = run_experiment("Raft", network_delay, failure_rate, 0)
        pbft_res = run_experiment("PBFT", network_delay, failure_rate, malicious_rate)

        raft_results.append(raft_res)
        pbft_results.append(pbft_res)

    analyze_data(raft_results, pbft_results)

    unittest.main(argv=[''], exit=False)