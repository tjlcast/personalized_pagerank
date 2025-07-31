use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct WeightedGraph<T> {
    /// 邻接表表示，存储 (节点, 权重) 对
    adjacency: HashMap<T, Vec<(T, f64)>>,
    /// 所有节点的集合
    nodes: Vec<T>,
}

impl<T> WeightedGraph<T>
where
    T: Clone + Hash + Eq + std::fmt::Debug,
{
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            nodes: Vec::new(),
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, node: T) {
        if !self.adjacency.contains_key(&node) {
            self.adjacency.insert(node.clone(), Vec::new());
            self.nodes.push(node);
        }
    }

    /// 添加带权重的边
    pub fn add_edge(&mut self, from: T, to: T, weight: f64) {
        // 确保节点存在
        self.add_node(from.clone());
        self.add_node(to.clone());

        // 添加边
        if let Some(edges) = self.adjacency.get_mut(&from) {
            edges.push((to, weight));
        }
    }

    /// 获取节点的出边
    pub fn get_edges(&self, node: &T) -> Option<&Vec<(T, f64)>> {
        self.adjacency.get(node)
    }

    /// 获取所有节点
    pub fn nodes(&self) -> &Vec<T> {
        &self.nodes
    }

    /// 计算每个节点的出度权重和
    fn out_degree_weights(&self) -> HashMap<T, f64> {
        let mut out_weights = HashMap::new();

        for (node, edges) in &self.adjacency {
            let total_weight: f64 = edges.iter().map(|(_, weight)| weight).sum();
            out_weights.insert(node.clone(), total_weight);
        }

        out_weights
    }

    /// Personalized PageRank 计算
    ///
    /// # 参数
    /// - `personalization`: 个性化向量，指定哪些节点作为"起始点"及其权重
    /// - `alpha`: 阻尼系数，通常为 0.85
    /// - `max_iter`: 最大迭代次数
    /// - `tolerance`: 收敛容忍度
    pub fn personalized_pagerank(
        &self,
        personalization: Option<HashMap<T, f64>>,
        alpha: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> HashMap<T, f64> {
        let n = self.nodes.len();
        if n == 0 {
            return HashMap::new();
        }

        // 初始化个性化向量
        let personalization = personalization.unwrap_or_else(|| {
            // 默认均匀分布
            let uniform_weight = 1.0 / n as f64;
            self.nodes
                .iter()
                .map(|node| (node.clone(), uniform_weight))
                .collect()
        });

        // 归一化个性化向量
        let total_personalization: f64 = personalization.values().sum();
        let normalized_personalization: HashMap<T, f64> = personalization
            .iter()
            .map(|(k, v)| (k.clone(), v / total_personalization))
            .collect();

        // 计算出度权重
        let out_weights = self.out_degree_weights();

        // 初始化 PageRank 值
        let mut pagerank: HashMap<T, f64> = self
            .nodes
            .iter()
            .map(|node| (node.clone(), 1.0 / n as f64))
            .collect();

        // 迭代计算
        for _ in 0..max_iter {
            let mut new_pagerank = HashMap::new();

            // 初始化新的 PageRank 值为个性化向量的贡献
            for node in &self.nodes {
                let personalization_contrib =
                    (1.0 - alpha) * normalized_personalization.get(node).unwrap_or(&0.0);
                new_pagerank.insert(node.clone(), personalization_contrib);
            }

            // 计算来自其他节点的贡献
            for (from_node, edges) in &self.adjacency {
                let from_pagerank = pagerank.get(from_node).unwrap_or(&0.0);
                let out_weight = out_weights.get(from_node).unwrap_or(&0.0);

                if *out_weight > 0.0 {
                    for (to_node, edge_weight) in edges {
                        let contribution = alpha * from_pagerank * (edge_weight / out_weight);
                        *new_pagerank.entry(to_node.clone()).or_insert(0.0) += contribution;
                    }
                }
            }

            // 处理悬挂节点（没有出边的节点）
            let hanging_mass: f64 = self
                .nodes
                .iter()
                .filter(|node| out_weights.get(node).unwrap_or(&0.0) == &0.0)
                .map(|node| pagerank.get(node).unwrap_or(&0.0))
                .sum();

            // 将悬挂节点的质量按个性化向量重新分配
            for node in &self.nodes {
                let hanging_contrib =
                    alpha * hanging_mass * normalized_personalization.get(node).unwrap_or(&0.0);
                *new_pagerank.entry(node.clone()).or_insert(0.0) += hanging_contrib;
            }

            // 检查收敛
            let diff: f64 = self
                .nodes
                .iter()
                .map(|node| {
                    let old_val = pagerank.get(node).unwrap_or(&0.0);
                    let new_val = new_pagerank.get(node).unwrap_or(&0.0);
                    (old_val - new_val).abs()
                })
                .sum();

            pagerank = new_pagerank;

            if diff < tolerance {
                break;
            }
        }

        pagerank
    }
}

/// 打印并排序 HashMap 的内容（按 value 降序排列）
///
/// # 参数
/// * `map` - 要打印和排序的 HashMap
/// * `title` - 可选的标题，用于在打印结果前显示
///
/// # 泛型约束
/// * `K: Debug + Eq + std::hash::Hash + Copy` - 键类型必须可调试、可比较、可哈希且可复制
/// * `V: Debug + PartialOrd` - 值类型必须可调试且可比较
pub fn print_sorted_map<K, V>(map: &HashMap<K, V>, title: Option<&str>)
where
    K: Debug + Eq + std::hash::Hash + Copy,
    V: Debug + PartialOrd,
{
    // 如果有标题，先打印标题
    if let Some(t) = title {
        println!("{}", t);
    }

    // 将 map 转换为元组数组并排序
    let mut sorted_items: Vec<_> = map.iter().collect();
    sorted_items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 打印排序后的结果
    for (key, value) in sorted_items {
        println!("{:?}: {:?}", key, value);
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_simple_graph() {
        let mut graph = WeightedGraph::new();

        // 创建一个简单的图: A -> B -> C -> A
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6);

        // 在对称图中，所有节点的 PageRank 应该大致相等
        let expected = 1.0 / 3.0;
        for (_, value) in result.iter() {
            assert!((value - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_weighted_graph() {
        let mut graph = WeightedGraph::new();

        // 创建带权重的图
        graph.add_edge("A", "B", 2.0);
        graph.add_edge("A", "C", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6);

        // 验证结果合理性
        assert!(result.len() == 3);
        let sum: f64 = result.values().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_personalized_pagerank() {
        let mut graph = WeightedGraph::new();

        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.8);
        personalization.insert("B", 0.1);
        personalization.insert("C", 0.1);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);

        // A 应该有最高的 PageRank 值
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();

        assert!(a_rank > b_rank);
        assert!(a_rank > c_rank);
    }

    #[test]
    fn test_personalized_pagerank1() {
        let mut graph = WeightedGraph::new();

        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("B", 0.1);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);

        // A 应该有最高的 PageRank 值
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();

        println!("result: {:?}", result);

        assert!(b_rank > a_rank);
        assert!(a_rank > c_rank);
    }

    #[test]
    fn test_normal_pagerank1() {
        let mut graph = WeightedGraph::new();

        graph.add_edge("B", "A", 1.0);
        graph.add_edge("C", "A", 1.0);
        graph.add_edge("D", "A", 1.0);

        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "D", 1.0);

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.1);
        personalization.insert("B", 0.1);
        personalization.insert("C", 0.1);
        personalization.insert("D", 0.1);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);

        // A 应该有最高的 PageRank 值
        // let a_rank = result.get("A").unwrap();
        // let b_rank = result.get("B").unwrap();
        // let c_rank = result.get("C").unwrap();
        // let d_rank = result.get("D").unwrap();

        println!("result: {:?}", result);
    }

    #[test]
    fn test_normal_pagerank2() {
        let mut graph = WeightedGraph::new();

        graph.add_edge("B", "A", 1.0);
        graph.add_edge("C", "A", 1.0);
        graph.add_edge("D", "A", 1.0);

        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "D", 1.0);
        graph.add_edge("D", "C", 1.0);

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.1);
        personalization.insert("B", 0.1);
        personalization.insert("C", 0.1);
        personalization.insert("D", 0.1);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);

        // A 应该有最高的 PageRank 值
        // let a_rank = result.get("A").unwrap();
        // let b_rank = result.get("B").unwrap();
        // let c_rank = result.get("C").unwrap();
        // let c_rank = result.get("D").unwrap();

        println!("result: {:?}", result);
    }

    // >>>>>>>> basic >>>>>>>
    #[test]
    fn test_basic_pagerank() {
        let mut graph = WeightedGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6);

        // In a simple cycle, all nodes should have equal rank
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();

        assert_relative_eq!(a_rank, b_rank, epsilon = 1e-6);
        assert_relative_eq!(b_rank, c_rank, epsilon = 1e-6);
    }

    #[test]
    fn test_dangling_node() {
        let mut graph = WeightedGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        // C is a dangling node with no outgoing links

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6);
        println!("result: {:?}", result);

        // C should still have some rank from incoming link
        assert!(result.get("C").unwrap() > &0.0);
        // C should have higher rank than A
        assert!(result.get("C").unwrap() > result.get("A").unwrap());
    }

    #[test]
    fn test_disconnected_components() {
        let mut graph = WeightedGraph::new();
        // Component 1
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "A", 1.0);
        // Component 2
        graph.add_edge("C", "D", 1.0);
        graph.add_edge("D", "C", 1.0);

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6);
        println!("result: {:?}", result);

        // Within each component, ranks should be equal
        assert_relative_eq!(
            result.get("A").unwrap(),
            result.get("B").unwrap(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            result.get("C").unwrap(),
            result.get("D").unwrap(),
            epsilon = 1e-6
        );
    }

    // >>>>>>> personalized >>>>>>>>
    #[test]
    fn test_strong_personalization() {
        let mut graph = WeightedGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        // Strong personalization to node B
        let mut personalization = HashMap::new();
        personalization.insert("B", 1.0); // 100% preference for B

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);
        println!("result: {:?}", result);

        // B should have the highest rank
        assert!(result.get("B").unwrap() > result.get("A").unwrap());
        assert!(result.get("B").unwrap() > result.get("C").unwrap());
    }

    #[test]
    fn test_personalization_with_missing_node() {
        let mut graph = WeightedGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);

        // Personalization vector doesn't include C
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.7);
        personalization.insert("B", 0.3);

        let result: HashMap<&'static str, f64> =
            graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);
        println!("result: {:?}", result);
        print_sorted_map(&result, Some("Original result:"));

        // C should still have some rank (from teleportation)
        assert!(result.get("C").unwrap() > &0.0);
        // A should have higher rank than B due to personalization
        assert!(result.get("B").unwrap() > result.get("C").unwrap());
        assert!(result.get("C").unwrap() > result.get("A").unwrap());
    }

    #[test]
    fn test_personalization_with_extra_node() {
        let mut graph = WeightedGraph::new();
        graph.add_edge("B", "A", 1.0);

        // Personalization vector includes node not in graph
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.5);
        personalization.insert("B", 0.3);
        personalization.insert("C", 0.2); // C not in graph

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);
        println!("result: {:?}", result);

        // Should ignore C in personalization
        assert!(result.get("A").unwrap() > result.get("B").unwrap());
        assert!(result.contains_key("C") == false);
    }
}
