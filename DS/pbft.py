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
    def __init__(self, node_id, total_nodes):
        self.node_id = node_id
        self.role = "Backup"
        self.requests = {}
        self.prepared = {}
        self.confirmed = {}
        self.executed_requests = set()
        self.f = (total_nodes - 1) // 3  # 动态计算容错数
        self.is_failed = False

    def handle_request(self, request, request_id, all_pbft_nodes):
        if self.is_failed:
            return
        print(f"Node {self.node_id} received request {request_id}")

        if self.role == "Primary":
            self.requests[request_id] = request
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
            n = len(all_pbft_nodes)
            if self.prepared[request_id] >= (n - self.f):  # 使用 n - f 作为法定人数
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
            n = len(all_pbft_nodes)
            if self.confirmed[request_id] >= (n - self.f):  # 使用 n - f 作为法定人数
                self._execute_request(request_id)

    def _execute_request(self, request_id):
        if request_id in self.executed_requests:
            return
        if request_id not in self.requests:
            print(f"Node {self.node_id} 未找到请求 {request_id}，无法执行")
            return
        request = self.requests[request_id]
        print(f"Node {self.node_id} executed request: {request}")
        self.executed_requests.add(request_id)


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
        original_broadcast = node.broadcast_pre_prepare

        def malicious_send(self, request, request_id, all_pbft_nodes):
            if random.random() < 0.5:
                print(f"Malicious node {self.node_id} sending wrong message")
                fake_request = "FAKE_" + str(random.randint(1000, 9999))
                original_broadcast(fake_request, request_id, all_pbft_nodes)
            else:
                original_broadcast(request, request_id, all_pbft_nodes)

        import types
        node.broadcast_pre_prepare = types.MethodType(malicious_send, node)


def run_experiment(network_delay, node_failure_rate, malicious_node_rate):
    start_time = time.time()
    node_count = 10
    all_pbft_nodes = [PBFTNode(i, node_count) for i in range(node_count)]
    all_pbft_nodes[0].role = "Primary"
    client = PBFTClient()
    simulate_network_delay(network_delay)
    simulate_node_failure(all_pbft_nodes, node_failure_rate)
    simulate_malicious_nodes(all_pbft_nodes, malicious_node_rate)

    for _ in range(10):
        client.send_request("Test Request", all_pbft_nodes[0], all_pbft_nodes)
        time.sleep(0.1)  # 原 0.01，增加等待时间
    # 其他代码不变...

    end_time = time.time()
    consensus_time = end_time - start_time
    throughput = 10 / consensus_time if consensus_time > 0 else 0

    # 检查数据一致性
    non_failed_nodes = [n for n in all_pbft_nodes if not n.is_failed]
    if non_failed_nodes:
        reference_log = non_failed_nodes[0].executed_requests
        data_consistency = all(node.executed_requests == reference_log for node in non_failed_nodes)
    else:
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
        self.node_count = 10
        self.all_pbft_nodes = [PBFTNode(i, self.node_count) for i in range(self.node_count)]
        self.all_pbft_nodes[0].role = "Primary"
        self.client = PBFTClient()
        for node in self.all_pbft_nodes:
            node.is_failed = False

    def test_pbft_request_processing(self):
        self.client.send_request("Sample Request", self.all_pbft_nodes[0], self.all_pbft_nodes)
        time.sleep(0.1)
        self.assertIn(1, self.all_pbft_nodes[0].requests)
        executed_count = sum(1 for node in self.all_pbft_nodes if 1 in node.executed_requests)
        n = self.node_count
        f = (n - 1) // 3
        expected_min = n - f
        self.assertGreaterEqual(executed_count, expected_min,
                                f"至少应有 {expected_min} 个节点执行请求，但实际只有 {executed_count} 个")


if __name__ == '__main__':
    pbft_results = []

    for _ in range(50):
        network_delay = random.randint(10, 500)
        failure_rate = random.uniform(0.1, 0.5)
        malicious_rate = random.uniform(0.05, 0.2)

        pbft_res = run_experiment(network_delay, failure_rate, malicious_rate)
        pbft_results.append(pbft_res)

        # 输出每次实验的结果
        print(
            f"实验 {_ + 1}/{50}: 共识时间={pbft_res[0]:.4f}s, 吞吐量={pbft_res[1]:.2f}req/s, 数据一致性={pbft_res[2]}")

    analyze_data(pbft_results)

    # 计算并输出统计信息
    print("\n=== 实验统计 ===")
    success_rate = sum(1 for res in pbft_results if res[2]) / len(pbft_results) * 100
    avg_time = np.mean([res[0] for res in pbft_results])
    avg_throughput = np.mean([res[1] for res in pbft_results])
    print(f"成功率: {success_rate:.2f}%")
    print(f"平均共识时间: {avg_time:.4f}s")
    print(f"平均吞吐量: {avg_throughput:.2f}req/s")

    unittest.main(argv=[''], exit=False)