import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def generate_comparison_chart():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axis_off()

    # Table data
    columns = ['Algorithm', 'Fault Tolerance', 'Message Complexity', 'Consensus Efficiency', 'Typical Scenarios']
    rows = ['Raft', 'PBFT']

    data = [
        ['Raft', 'Crash Fault Tolerance', 'Low', 'High', 'Internal Distributed Systems (e.g., Etcd)'],
        ['PBFT', 'Byzantine Fault Tolerance', 'High', 'Moderate',
         'Blockchain Consortium Networks (e.g., Hyperledger Fabric)']
    ]

    # Colors
    header_color = '#40466e'
    row_colors = ['#f1f1f2', '#eaeaf2']
    edge_color = 'w'

    # Create table
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add headers
    for col_idx, col_label in enumerate(columns):
        table.add_cell(0, col_idx, 0.2, 0.1, text=col_label,
                       loc='center', facecolor=header_color, textprops={'color': 'w', 'weight': 'bold'})

    # Add data rows
    for row_idx, row_data in enumerate(data):
        for col_idx, cell_data in enumerate(row_data):
            table.add_cell(row_idx + 1, col_idx, 0.2, 0.1, text=cell_data,
                           loc='center', facecolor=row_colors[row_idx % len(row_colors)],
                           edgecolor=edge_color)

    # Add row headers
    for row_idx, row_label in enumerate(rows):
        table.add_cell(row_idx + 1, -1, 0.2, 0.1, text=row_label,
                       loc='center', facecolor=header_color, textprops={'color': 'w', 'weight': 'bold'})

    # Set table properties
    ax.add_table(table)

    # Add title
    plt.title('Comparison of Raft and PBFT Consensus Algorithms', fontsize=16, pad=20)

    # Save the chart
    plt.savefig('consensus_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    print("Chart saved successfully as 'consensus_algorithms_comparison.png'")

    # Show the plot
    plt.show()


# Run the function to generate the chart
generate_comparison_chart()