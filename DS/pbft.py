import random
import time
import os
import unittest
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端，用于保存图片
import matplotlib.pyplot as plt
import numpy as np


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
        self.executed_requests = set()
        self.f = 1
        self.is_failed = False

    def handle_request(self, request, request_id, all_pbft_nodes):
        if self.is_failed:
            return
        print(f"Node {self.node_id} received request {request_id}")

        if self.role == "Primary":
            self.requests[request_id] = request

        if self.role == "Primary":
            self.broadcast_pre_prepare(request, request_id, all_pbft_nodes)

    def broadcast_pre_prepare(self, request, request_id, all_pbft_nodes):
        if self.is_failed:
            return
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

    def _execute_request(self, request_id):
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


def run_experiment(network_delay, node_failure_rate, malicious_node_rate):
    start_time = time.time()
    node_count = 10  # 设置更多节点
    all_pbft_nodes = [PBFTNode(i) for i in range(node_count)]
    all_pbft_nodes[0].role = "Primary"
    client = PBFTClient()
    simulate_network_delay(network_delay)
    simulate_node_failure(all_pbft_nodes, node_failure_rate)
    simulate_malicious_nodes(all_pbft_nodes, malicious_node_rate)

    for _ in range(10):
        client.send_request("Test Request", all_pbft_nodes[0], all_pbft_nodes)
        time.sleep(0.01)

    end_time = time.time()
    consensus_time = end_time - start_time
    throughput = 10 / consensus_time if consensus_time > 0 else 0
    data_consistency = True
    return consensus_time, throughput, data_consistency


def analyze_data(pbft_data):
    try:
        pbft_consensus_times = np.array([data[0] for data in pbft_data])
        pbft_throughputs = np.array([data[1] for data in pbft_data])

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(pbft_consensus_times, label='PBFT')
        plt.title('Consensus Time')
        plt.xlabel('Experiment')
        plt.ylabel('Time (s)')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(pbft_throughputs, label='PBFT')
        plt.title('Throughput')
        plt.xlabel('Experiment')
        plt.ylabel('Requests/s')
        plt.legend()

        plt.tight_layout()

        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, 'pbft_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

        print("Current directory:", current_dir)
    except Exception as e:
        print(f"保存图表时出现错误: {e}")


class TestPBFT(unittest.TestCase):
    def setUp(self):
        self.all_pbft_nodes = [PBFTNode(i) for i in range(10)]
        self.all_pbft_nodes[0].role = "Primary"
        self.client = PBFTClient()
        for node in self.all_pbft_nodes:
            node.is_failed = False

    def test_pbft_request_processing(self):
        self.client.send_request("Sample Request", self.all_pbft_nodes[0], self.all_pbft_nodes)
        time.sleep(0.1)
        self.assertIn(1, self.all_pbft_nodes[0].requests)
        executed_count = sum(1 for node in self.all_pbft_nodes if 1 in node.executed_requests)
        self.assertGreaterEqual(executed_count, 3)


if __name__ == '__main__':
    pbft_results = []

    for _ in range(50):
        network_delay = random.randint(10, 500)
        failure_rate = random.uniform(0.1, 0.5)
        malicious_rate = random.uniform(0.05, 0.2)

        pbft_res = run_experiment(network_delay, failure_rate, malicious_node_rate)
        pbft_results.append(pbft_res)

    analyze_data(pbft_results)

    unittest.main(argv=[''], exit=False)