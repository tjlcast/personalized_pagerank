use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct EdgeData<I> {
    pub weight: f64,
    pub ident: I,
}

impl<I> EdgeData<I> {
    pub fn new(weight: f64, ident: I) -> Self {
        Self { weight, ident }
    }
}

#[derive(Debug, Clone)]
pub struct WeightedGraph<T, I = String> {
    /// 邻接表表示，存储 (节点, 边数据) 对
    adjacency: HashMap<T, Vec<(T, EdgeData<I>)>>,
    /// 所有节点的集合，保持插入顺序
    nodes: Vec<T>,
}

impl<T, I> WeightedGraph<T, I>
where
    T: Clone + Hash + Eq + std::fmt::Debug + Ord,
    I: Clone + std::fmt::Debug,
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

    /// 添加带权重和标识符的边
    pub fn add_edge(&mut self, from: T, to: T, weight: f64, ident: I) {
        // 确保节点存在
        self.add_node(from.clone());
        self.add_node(to.clone());

        // 添加边
        if let Some(edges) = self.adjacency.get_mut(&from) {
            edges.push((to, EdgeData::new(weight, ident)));
        }
    }

    /// 添加带权重的边（使用默认标识符）
    pub fn add_edge_simple(&mut self, from: T, to: T, weight: f64)
    where
        I: Default,
    {
        self.add_edge(from, to, weight, I::default());
    }

    /// 获取节点的出边信息
    /// 返回格式: Vec<(&T, &T, &EdgeData<I>)> 对应 (src, dst, data)
    pub fn out_edges<'a>(&'a self, node: &'a T) -> Vec<(&'a T, &'a T, &'a EdgeData<I>)> {
        if let Some(edges) = self.adjacency.get(node) {
            edges.iter().map(|(dst, data)| (node, dst, data)).collect()
        } else {
            Vec::new()
        }
    }

    /// 获取节点的出边（仅返回目标节点和边数据）
    pub fn get_edges(&self, node: &T) -> Option<&Vec<(T, EdgeData<I>)>> {
        self.adjacency.get(node)
    }

    /// 获取所有节点
    pub fn nodes(&self) -> &Vec<T> {
        &self.nodes
    }

    /// 获取所有边的迭代器
    pub fn edges(&self) -> impl Iterator<Item = (&T, &T, &EdgeData<I>)> {
        self.adjacency
            .iter()
            .flat_map(|(src, edges)| edges.iter().map(move |(dst, data)| (src, dst, data)))
    }

    /// 计算每个节点的出度权重和
    fn out_degree_weights(&self) -> HashMap<T, f64> {
        let mut out_weights = HashMap::new();

        for (node, edges) in &self.adjacency {
            let total_weight: f64 = edges.iter().map(|(_, edge_data)| edge_data.weight).sum();
            out_weights.insert(node.clone(), total_weight);
        }

        out_weights
    }

    /// 确定性的 Personalized PageRank 计算
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

        // 确保节点有确定的顺序
        let mut sorted_nodes = self.nodes.clone();
        sorted_nodes.sort(); // 关键：排序节点确保确定性

        // 初始化个性化向量
        let personalization = personalization.unwrap_or_else(|| {
            // 默认均匀分布
            let uniform_weight = 1.0 / n as f64;
            sorted_nodes
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
        let mut pagerank: HashMap<T, f64> = sorted_nodes
            .iter()
            .map(|node| (node.clone(), 1.0 / n as f64))
            .collect();

        // 迭代计算
        for _ in 0..max_iter {
            let mut new_pagerank = HashMap::new();

            // 初始化新的 PageRank 值为个性化向量的贡献
            for node in &sorted_nodes {
                let personalization_contrib =
                    (1.0 - alpha) * normalized_personalization.get(node).unwrap_or(&0.0);
                new_pagerank.insert(node.clone(), personalization_contrib);
            }

            // 按确定顺序计算来自其他节点的贡献
            for from_node in &sorted_nodes {
                if let Some(edges) = self.adjacency.get(from_node) {
                    let from_pagerank = pagerank.get(from_node).unwrap_or(&0.0);
                    let out_weight = out_weights.get(from_node).unwrap_or(&0.0);

                    if *out_weight > 0.0 {
                        // 对边也进行排序以确保确定性
                        let mut sorted_edges = edges.clone();
                        sorted_edges.sort_by(|a, b| a.0.cmp(&b.0));

                        for (to_node, edge_data) in sorted_edges {
                            let contribution =
                                alpha * from_pagerank * (edge_data.weight / out_weight);
                            *new_pagerank.entry(to_node.clone()).or_insert(0.0) += contribution;
                        }
                    }
                }
            }

            // 处理悬挂节点（没有出边的节点）
            let hanging_mass: f64 = sorted_nodes
                .iter()
                .filter(|node| out_weights.get(node).unwrap_or(&0.0) == &0.0)
                .map(|node| pagerank.get(node).unwrap_or(&0.0))
                .sum();

            // 将悬挂节点的质量按个性化向量重新分配
            for node in &sorted_nodes {
                let hanging_contrib =
                    alpha * hanging_mass * normalized_personalization.get(node).unwrap_or(&0.0);
                *new_pagerank.entry(node.clone()).or_insert(0.0) += hanging_contrib;
            }

            // 检查收敛
            let diff: f64 = sorted_nodes
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

#[cfg(test)]
mod tests {
    use crate::utils::print_sorted_map;

    use super::*;

    #[test]
    fn test_edge_with_ident() {
        let mut graph: WeightedGraph<&str, &str> = WeightedGraph::new();

        // 添加带标识符的边
        graph.add_edge("A", "B", 1.0, "ref1");
        graph.add_edge("B", "C", 2.0, "ref2");
        graph.add_edge("C", "A", 0.5, "ref3");

        // 测试 out_edges 功能
        let a_edges = graph.out_edges(&"A");
        assert_eq!(a_edges.len(), 1);
        let (src, dst, data) = &a_edges[0];
        assert_eq!(*src, &"A");
        assert_eq!(*dst, &"B");
        assert_eq!(data.weight, 1.0);
        assert_eq!(data.ident, "ref1");

        // 测试 B 的出边
        let b_edges = graph.out_edges(&"B");
        assert_eq!(b_edges.len(), 1);
        let (src, dst, data) = &b_edges[0];
        assert_eq!(*src, &"B");
        assert_eq!(*dst, &"C");
        assert_eq!(data.weight, 2.0);
        assert_eq!(data.ident, "ref2");
    }

    #[test]
    fn test_multiple_out_edges() {
        let mut graph: WeightedGraph<&str, String> = WeightedGraph::new();

        // A 有多条出边
        graph.add_edge("A", "B", 1.0, "edge_ab".to_string());
        graph.add_edge("A", "C", 2.0, "edge_ac".to_string());
        graph.add_edge("A", "D", 0.5, "edge_ad".to_string());

        let a_edges = graph.out_edges(&"A");
        assert_eq!(a_edges.len(), 3);

        // 验证所有边的信息
        for (src, dst, data) in &a_edges {
            assert_eq!(*src, &"A");
            match *dst {
                &"B" => {
                    assert_eq!(data.weight, 1.0);
                    assert_eq!(data.ident, "edge_ab");
                }
                &"C" => {
                    assert_eq!(data.weight, 2.0);
                    assert_eq!(data.ident, "edge_ac");
                }
                &"D" => {
                    assert_eq!(data.weight, 0.5);
                    assert_eq!(data.ident, "edge_ad");
                }
                _ => panic!("Unexpected destination node"),
            }
        }
    }

    #[test]
    fn test_edges_iterator() {
        let mut graph: WeightedGraph<i32, String> = WeightedGraph::new();

        graph.add_edge(1, 2, 1.5, "first".to_string());
        graph.add_edge(2, 3, 2.5, "second".to_string());
        graph.add_edge(3, 1, 0.8, "third".to_string());

        let all_edges: Vec<_> = graph.edges().collect();
        assert_eq!(all_edges.len(), 3);

        // 验证边的内容
        for (src, dst, data) in all_edges {
            match (src, dst) {
                (1, 2) => {
                    assert_eq!(data.weight, 1.5);
                    assert_eq!(data.ident, "first");
                }
                (2, 3) => {
                    assert_eq!(data.weight, 2.5);
                    assert_eq!(data.ident, "second");
                }
                (3, 1) => {
                    assert_eq!(data.weight, 0.8);
                    assert_eq!(data.ident, "third");
                }
                _ => panic!("Unexpected edge"),
            }
        }
    }

    #[test]
    fn test_pagerank_with_idents() {
        let mut graph: WeightedGraph<&str, &str> = WeightedGraph::new();

        graph.add_edge("A", "B", 1.0, "link1");
        graph.add_edge("B", "C", 1.0, "link2");
        graph.add_edge("C", "A", 1.0, "link3");

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6);

        // 在对称图中，所有节点的 PageRank 应该大致相等
        let expected = 1.0 / 3.0;
        for (_, value) in result.iter() {
            assert!((value - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_usage_example() {
        let mut graph: WeightedGraph<&str, String> = WeightedGraph::new();

        // 模拟您的使用场景
        let referencer = "main_func";
        let definer = "helper_func";
        let use_mul = 2.0;
        let num_refs = 3;
        let ident = format!("ref_{}_{}", referencer, definer);

        graph.add_edge(referencer, definer, use_mul * num_refs as f64, ident);

        // 获取出边信息
        for (src, dst, data) in graph.out_edges(&referencer) {
            println!(
                "Edge from {:?} to {:?}: weight={}, ident={}",
                src, dst, data.weight, data.ident
            );
            assert_eq!(data.weight, 6.0); // 2.0 * 3
        }
    }

    // 保持原有的其他测试，只需要在创建图时指定类型
    #[test]
    fn test_simple_graph() {
        let mut graph: WeightedGraph<&str, String> = WeightedGraph::new();

        // 创建一个简单的图: A -> B -> C -> A
        graph.add_edge("A", "B", 1.0, "e1".to_string());
        graph.add_edge("B", "C", 1.0, "e2".to_string());
        graph.add_edge("C", "A", 1.0, "e3".to_string());

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6);

        // 在对称图中，所有节点的 PageRank 应该大致相等
        let expected = 1.0 / 3.0;
        for (_, value) in result.iter() {
            assert!((value - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_normal_pagerank1() {
        let mut graph = WeightedGraph::new();

        graph.add_edge("B", "A", 1.0, "B-A".to_string());
        graph.add_edge("C", "A", 1.0, "C-A".to_string());
        graph.add_edge("D", "A", 1.0, "D-A".to_string());

        graph.add_edge("B", "C", 1.0, "B-C".to_string());
        graph.add_edge("C", "D", 1.0, "C-D".to_string());

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.1);
        personalization.insert("B", 0.1);
        personalization.insert("C", 0.1);
        personalization.insert("D", 0.1);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);
        print_sorted_map(&result, Some("123"));

        // A 应该有最高的 PageRank 值
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();
        let d_rank = result.get("D").unwrap();

        assert!(a_rank > b_rank);
        assert!(a_rank > c_rank);
        assert!(a_rank > d_rank);

        assert!(d_rank > c_rank);
        assert!(d_rank > b_rank);

        if let Some(get_edges) = graph.get_edges(&"B") {
            let target_nodes: Vec<_> = get_edges.iter().map(|edge| edge.0).collect();
            println!("{:?}", get_edges);
            assert!(target_nodes.len() == 2);
            assert!(target_nodes.contains(&"A"));
            assert!(target_nodes.contains(&"C"));
        }

        let out_edges = graph.out_edges(&"C");
        let target_nodes: Vec<_> = out_edges.iter().map(|edge| *edge.1).collect();
        println!("{:?}", out_edges);
        assert!(target_nodes.len() == 2);
        assert!(target_nodes.contains(&"A"));
        assert!(target_nodes.contains(&"D"));
    }

    #[test]
    fn test_normal_pagerank2() {
        let mut graph = WeightedGraph::new();

        graph.add_edge("B", "A", 1.0, "".to_string());
        graph.add_edge("C", "A", 1.0, "".to_string());
        graph.add_edge("D", "A", 1.0, "".to_string());

        graph.add_edge("B", "C", 1.0, "".to_string());
        graph.add_edge("C", "D", 1.0, "".to_string());
        graph.add_edge("D", "C", 1.0, "".to_string());

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.1);
        personalization.insert("B", 0.1);
        personalization.insert("C", 0.1);
        personalization.insert("D", 0.1);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);
        print_sorted_map(&result, None);

        // A 应该有最高的 PageRank 值
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();
        let d_rank = result.get("D").unwrap();

        assert!(a_rank > b_rank);
        assert!(a_rank > c_rank);
        assert!(a_rank > d_rank);

        assert!(c_rank > b_rank);
        assert!(c_rank > d_rank);
    }
}
