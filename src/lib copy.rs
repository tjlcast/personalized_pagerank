use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

pub fn personalized_pagerank(
    edges: &[(usize, usize, f64)], // (source, target, weight) 形式的边列表
    personalization: Option<&HashMap<usize, f64>>,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> HashMap<usize, f64> {
    // 收集所有节点
    let mut nodes = HashSet::new();
    for &(src, dst, _) in edges {
        nodes.insert(src);
        nodes.insert(dst);
    }
    let nodes: Vec<usize> = nodes.into_iter().collect();
    let n = nodes.len();

    // 创建节点到索引的映射
    let node_to_idx: HashMap<usize, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, i))
        .collect();

    // 构建出边和入边的邻接表（带权重）
    let mut out_edges: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let mut out_weights: HashMap<usize, f64> = HashMap::new();

    for &(src, dst, weight) in edges {
        out_edges
            .entry(src)
            .or_insert_with(Vec::new)
            .push((dst, weight));
        *out_weights.entry(src).or_insert(0.0) += weight;
    }

    // 初始化个性化向量
    let mut personalization_vec = vec![0.0; n];
    if let Some(personalization) = personalization {
        let total: f64 = personalization.values().sum();
        for (&node, &weight) in personalization {
            if let Some(&idx) = node_to_idx.get(&node) {
                personalization_vec[idx] = weight / total;
            }
        }
    } else {
        // 如果没有提供个性化向量，则使用均匀分布
        let uniform_weight = 1.0 / n as f64;
        personalization_vec = vec![uniform_weight; n];
    }

    // 初始化 PageRank 向量
    let mut pagerank = vec![1.0 / n as f64; n];

    // 迭代计算
    for _ in 0..max_iter {
        let mut new_pagerank = vec![0.0; n];
        let mut dangling = 0.0;

        // 处理悬挂节点（没有出边的节点）
        for (i, &node) in nodes.iter().enumerate() {
            if !out_edges.contains_key(&node) {
                dangling += pagerank[i];
            }
        }

        // 分配悬挂节点的权重
        let dangling_contribution = dangling / n as f64;

        // 计算转移概率
        for (i, &node) in nodes.iter().enumerate() {
            let mut sum = 0.0;

            if let Some(edges) = out_edges.get(&node) {
                let total_weight = out_weights[&node];
                for &(dst, weight) in edges {
                    if let Some(&j) = node_to_idx.get(&dst) {
                        sum += pagerank[i] * weight / total_weight;
                    }
                }
            }

            new_pagerank[i] = alpha * sum + (1.0 - alpha) * personalization_vec[i];
        }

        // 添加悬挂节点的贡献
        for i in 0..n {
            new_pagerank[i] += alpha * dangling_contribution;
        }

        // 检查收敛
        let diff: f64 = pagerank
            .iter()
            .zip(&new_pagerank)
            .map(|(a, b)| (a - b).abs())
            .sum();

        pagerank = new_pagerank;

        if diff < tol {
            break;
        }
    }

    // 将结果转换回节点ID到PageRank值的映射
    nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| (node, pagerank[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_personalized_pagerank() {
        // 创建一个简单的图：0 -> 1 -> 2 -> 0
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];

        // 测试非个性化 PageRank
        let result = personalized_pagerank(&edges, None, 0.85, 100, 1e-6);

        // 在这个环形图中，所有节点的PageRank应该相等
        for &node in result.keys() {
            assert_abs_diff_eq!(result[&node], 1.0 / 3.0, epsilon = 1e-4);
        }

        // 测试个性化 PageRank
        let mut personalization = HashMap::new();
        personalization.insert(0, 200.0);
        let result = personalized_pagerank(&edges, Some(&personalization), 0.85, 100, 1e-6);

        // 节点0应该有最高的PageRank
        println!(
            "result[0]: {:?} result[1]: {:?} result[2]: {:?}",
            result[&0], result[&1], result[&2]
        );
        assert!(result[&0] > result[&1] && result[&1] > result[&2]);
    }
}
