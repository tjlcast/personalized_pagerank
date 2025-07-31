import networkx as nx


def test_personalization_with_missing_node():
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)

    # Personalization vector doesn't include "C"
    personalization = {
        "A": 0.7,
        "B": 0.3
        # "C" is intentionally missing
    }

    # networkx 会自动补全缺失节点，使它们分配为 0（然后自动归一化）
    result = nx.pagerank(G, alpha=0.85, personalization=personalization,
                         max_iter=100, tol=1e-6, weight="weight")
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    print("result:", result)

    # C should still have some rank (from teleportation)
    # assert result["C"] > 0.0, "Node C should have some rank due to teleportation"

    # A should have higher rank than B due to personalization
    # assert result["A"] > result["B"], "Node A should have higher rank than B due to personalization"


def test_personalization_with_extra_node():
    G = nx.DiGraph()
    G.add_edge("B", "A", weight=1.0)

    # Personalization vector includes a node not in graph
    personalization = {
        "A": 0.5,
        "B": 0.3,
        "C": 0.2  # C not in graph
    }

    # networkx 会自动忽略 personalization 中不存在的节点
    result = nx.pagerank(G, alpha=0.85, personalization=personalization,
                         max_iter=100, tol=1e-6, weight='weight')
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    print("result:", result)



def test_personalized_pagerank():
    # 创建一个 MultiDiGraph，并添加边: A -> B -> C -> A
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)

    # 将其转成 DiGraph 进行 pagerank 计算（因为 MultiDiGraph 不支持 pagerank）
    G_simple = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        if G_simple.has_edge(u, v):
            G_simple[u][v]['weight'] += w
        else:
            G_simple.add_edge(u, v, weight=w)

    # 默认 PageRank
    pr1 = nx.pagerank(G_simple, alpha=0.85, max_iter=100, tol=1e-6, weight='weight')
    print("默认PageRank:", pr1)

    # 个性化 PageRank，偏向 A
    personalization = {
        "A": 1.0,
        "B": 0.0,
        "C": 0.0
    }
    pr2 = nx.pagerank(G_simple, alpha=0.85, personalization=personalization, max_iter=100, tol=1e-6, weight='weight')
    print("个性化PageRank (偏向A):", pr2)

    # A 应该有最高的 PageRank 值
    assert pr2["A"] > pr2["B"]
    assert pr2["A"] > pr2["C"]

    

if __name__ == "__main__":
    test_personalization_with_missing_node()
    test_personalization_with_extra_node()
    test_personalized_pagerank()
