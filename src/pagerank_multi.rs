use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// 边的结构体，包含目标节点、权重、唯一ID和额外属性
#[derive(Debug, Clone)]
pub struct Edge<T> {
    pub to: T,
    pub weight: f64,
    pub id: usize,
    pub attributes: HashMap<String, String>,
}

impl<T> Edge<T> {
    pub fn new(to: T, weight: f64, id: usize) -> Self {
        Self {
            to,
            weight,
            id,
            attributes: HashMap::new(),
        }
    }

    pub fn with_attribute(mut self, key: String, value: String) -> Self {
        self.attributes.insert(key, value);
        self
    }
}

/// 多重有向图结构
#[derive(Debug, Clone)]
pub struct MultiDiGraph<T> {
    /// 邻接表表示，存储边的列表
    adjacency: HashMap<T, Vec<Edge<T>>>,
    /// 反向邻接表，用于快速查找入边
    reverse_adjacency: HashMap<T, Vec<Edge<T>>>,
    /// 所有节点的集合
    nodes: Vec<T>,
    /// 下一个边ID
    next_edge_id: usize,
    /// 节点属性
    node_attributes: HashMap<T, HashMap<String, String>>,
}

impl<T> MultiDiGraph<T>
where
    T: Clone + Hash + Eq + Debug,
{
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
            nodes: Vec::new(),
            next_edge_id: 0,
            node_attributes: HashMap::new(),
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, node: T) -> bool {
        if !self.adjacency.contains_key(&node) {
            self.adjacency.insert(node.clone(), Vec::new());
            self.reverse_adjacency.insert(node.clone(), Vec::new());
            self.nodes.push(node.clone());
            self.node_attributes.insert(node, HashMap::new());
            true
        } else {
            false
        }
    }

    /// 添加节点并设置属性
    pub fn add_node_with_attributes(
        &mut self,
        node: T,
        attributes: HashMap<String, String>,
    ) -> bool {
        let is_new = self.add_node(node.clone());
        if let Some(node_attrs) = self.node_attributes.get_mut(&node) {
            node_attrs.extend(attributes);
        }
        is_new
    }

    /// 添加带权重的边，支持多重边
    pub fn add_edge(&mut self, from: T, to: T, weight: f64) -> usize {
        // 确保节点存在
        self.add_node(from.clone());
        self.add_node(to.clone());

        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;

        // 添加到正向邻接表
        let edge = Edge::new(to.clone(), weight, edge_id);
        if let Some(edges) = self.adjacency.get_mut(&from) {
            edges.push(edge.clone());
        }

        // 添加到反向邻接表
        let reverse_edge = Edge::new(from.clone(), weight, edge_id);
        if let Some(reverse_edges) = self.reverse_adjacency.get_mut(&to) {
            reverse_edges.push(reverse_edge);
        }

        edge_id
    }

    /// 添加带属性的边
    pub fn add_edge_with_attributes(
        &mut self,
        from: T,
        to: T,
        weight: f64,
        attributes: HashMap<String, String>,
    ) -> usize {
        let edge_id = self.add_edge(from.clone(), to.clone(), weight);

        // 更新边属性
        if let Some(edges) = self.adjacency.get_mut(&from) {
            if let Some(edge) = edges.iter_mut().find(|e| e.id == edge_id) {
                edge.attributes.extend(attributes.clone());
            }
        }

        if let Some(reverse_edges) = self.reverse_adjacency.get_mut(&to) {
            if let Some(edge) = reverse_edges.iter_mut().find(|e| e.id == edge_id) {
                edge.attributes.extend(attributes);
            }
        }

        edge_id
    }

    /// 删除边
    pub fn remove_edge(&mut self, edge_id: usize) -> bool {
        let mut removed = false;

        // 从正向邻接表中删除
        for edges in self.adjacency.values_mut() {
            if let Some(pos) = edges.iter().position(|e| e.id == edge_id) {
                edges.remove(pos);
                removed = true;
                break;
            }
        }

        // 从反向邻接表中删除
        for edges in self.reverse_adjacency.values_mut() {
            if let Some(pos) = edges.iter().position(|e| e.id == edge_id) {
                edges.remove(pos);
                break;
            }
        }

        removed
    }

    /// 删除节点及其所有相关边
    pub fn remove_node(&mut self, node: &T) -> bool {
        if !self.adjacency.contains_key(node) {
            return false;
        }

        // 收集需要删除的边ID
        let mut edge_ids_to_remove = Vec::new();

        // 收集出边ID
        if let Some(out_edges) = self.adjacency.get(node) {
            edge_ids_to_remove.extend(out_edges.iter().map(|e| e.id));
        }

        // 收集入边ID
        if let Some(in_edges) = self.reverse_adjacency.get(node) {
            edge_ids_to_remove.extend(in_edges.iter().map(|e| e.id));
        }

        // 删除所有相关边
        for edge_id in edge_ids_to_remove {
            self.remove_edge(edge_id);
        }

        // 删除节点
        self.adjacency.remove(node);
        self.reverse_adjacency.remove(node);
        self.node_attributes.remove(node);
        self.nodes.retain(|n| n != node);

        true
    }

    /// 获取节点的出边
    pub fn get_out_edges(&self, node: &T) -> Option<&Vec<Edge<T>>> {
        self.adjacency.get(node)
    }

    /// 获取节点的入边
    pub fn get_in_edges(&self, node: &T) -> Option<&Vec<Edge<T>>> {
        self.reverse_adjacency.get(node)
    }

    /// 获取两个节点之间的所有边
    pub fn get_edges_between(&self, from: &T, to: &T) -> Vec<&Edge<T>> {
        if let Some(edges) = self.adjacency.get(from) {
            edges.iter().filter(|edge| &edge.to == to).collect()
        } else {
            Vec::new()
        }
    }

    /// 检查边是否存在
    pub fn has_edge(&self, from: &T, to: &T) -> bool {
        !self.get_edges_between(from, to).is_empty()
    }

    /// 获取所有节点
    pub fn nodes(&self) -> &Vec<T> {
        &self.nodes
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 获取边数量
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|edges| edges.len()).sum()
    }

    /// 获取节点的出度（边数）
    pub fn out_degree(&self, node: &T) -> usize {
        self.adjacency
            .get(node)
            .map(|edges| edges.len())
            .unwrap_or(0)
    }

    /// 获取节点的入度（边数）
    pub fn in_degree(&self, node: &T) -> usize {
        self.reverse_adjacency
            .get(node)
            .map(|edges| edges.len())
            .unwrap_or(0)
    }

    /// 获取节点的出度权重和
    pub fn out_degree_weight(&self, node: &T) -> f64 {
        self.adjacency
            .get(node)
            .map(|edges| edges.iter().map(|e| e.weight).sum())
            .unwrap_or(0.0)
    }

    /// 获取节点的入度权重和
    pub fn in_degree_weight(&self, node: &T) -> f64 {
        self.reverse_adjacency
            .get(node)
            .map(|edges| edges.iter().map(|e| e.weight).sum())
            .unwrap_or(0.0)
    }

    /// 获取所有节点的出度权重和
    fn out_degree_weights(&self) -> HashMap<T, f64> {
        let mut out_weights = HashMap::new();

        for (node, edges) in &self.adjacency {
            let total_weight: f64 = edges.iter().map(|e| e.weight).sum();
            out_weights.insert(node.clone(), total_weight);
        }

        out_weights
    }

    /// 获取节点属性
    pub fn get_node_attribute(&self, node: &T, key: &str) -> Option<&String> {
        self.node_attributes.get(node)?.get(key)
    }

    /// 设置节点属性
    pub fn set_node_attribute(&mut self, node: &T, key: String, value: String) -> bool {
        if let Some(attrs) = self.node_attributes.get_mut(node) {
            attrs.insert(key, value);
            true
        } else {
            false
        }
    }

    /// 个性化PageRank计算
    ///
    /// # 参数
    /// - `personalization`: 个性化向量，指定哪些节点作为"起始点"及其权重
    /// - `alpha`: 阻尼系数，通常为0.85
    /// - `max_iter`: 最大迭代次数
    /// - `tolerance`: 收敛容忍度
    /// - `weight_key`: 如果指定，使用边的指定属性作为权重而不是默认权重
    pub fn personalized_pagerank(
        &self,
        personalization: Option<HashMap<T, f64>>,
        alpha: f64,
        max_iter: usize,
        tolerance: f64,
        weight_key: Option<&str>,
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
        if total_personalization == 0.0 {
            return HashMap::new();
        }

        let normalized_personalization: HashMap<T, f64> = personalization
            .iter()
            .map(|(k, v)| (k.clone(), v / total_personalization))
            .collect();

        // 计算出度权重（支持使用边属性作为权重）
        let out_weights = if let Some(key) = weight_key {
            self.out_degree_weights_by_attribute(key)
        } else {
            self.out_degree_weights()
        };

        // 初始化PageRank值
        let mut pagerank: HashMap<T, f64> = self
            .nodes
            .iter()
            .map(|node| (node.clone(), 1.0 / n as f64))
            .collect();

        // 迭代计算
        for iteration in 0..max_iter {
            let mut new_pagerank = HashMap::new();

            // 初始化新的PageRank值为个性化向量的贡献
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
                    for edge in edges {
                        let edge_weight = if let Some(key) = weight_key {
                            edge.attributes
                                .get(key)
                                .and_then(|v| v.parse::<f64>().ok())
                                .unwrap_or(edge.weight)
                        } else {
                            edge.weight
                        };

                        let contribution = alpha * from_pagerank * (edge_weight / out_weight);
                        *new_pagerank.entry(edge.to.clone()).or_insert(0.0) += contribution;
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
                println!("个性化PageRank在第{}次迭代后收敛", iteration + 1);
                break;
            }
        }

        pagerank
    }

    /// 根据边属性计算出度权重
    fn out_degree_weights_by_attribute(&self, weight_key: &str) -> HashMap<T, f64> {
        let mut out_weights = HashMap::new();

        for (node, edges) in &self.adjacency {
            let total_weight: f64 = edges
                .iter()
                .map(|edge| {
                    edge.attributes
                        .get(weight_key)
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(edge.weight)
                })
                .sum();
            out_weights.insert(node.clone(), total_weight);
        }

        out_weights
    }

    /// 打印图的基本信息
    pub fn print_info(&self) {
        println!("多重有向图信息:");
        println!("  节点数: {}", self.node_count());
        println!("  边数: {}", self.edge_count());

        for node in &self.nodes {
            println!(
                "  节点 {:?}: 出度={}, 入度={}, 出度权重={:.2}, 入度权重={:.2}",
                node,
                self.out_degree(node),
                self.in_degree(node),
                self.out_degree_weight(node),
                self.in_degree_weight(node)
            );
        }
    }
}

impl<T> Default for MultiDiGraph<T>
where
    T: Clone + Hash + Eq + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_digraph_basic_operations() {
        let mut graph = MultiDiGraph::new();

        // 添加节点
        assert!(graph.add_node("A"));
        assert!(graph.add_node("B"));
        assert!(graph.add_node("C"));
        assert!(!graph.add_node("A")); // 重复添加应该返回false

        // 添加边
        let edge1 = graph.add_edge("A", "B", 1.0);
        let _edge2 = graph.add_edge("A", "B", 2.0); // 多重边
        let _edge3 = graph.add_edge("B", "C", 1.5);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
        assert_eq!(graph.out_degree(&"A"), 2);
        assert_eq!(graph.in_degree(&"B"), 2);
        assert_eq!(graph.out_degree_weight(&"A"), 3.0);

        // 检查多重边
        let edges_ab = graph.get_edges_between(&"A", &"B");
        assert_eq!(edges_ab.len(), 2);

        // 删除边
        assert!(graph.remove_edge(edge1));
        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.out_degree(&"A"), 1);
    }

    #[test]
    fn test_personalized_pagerank() {
        let mut graph = MultiDiGraph::new();

        // 创建一个简单的图: A -> B -> C -> A
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        // 测试默认PageRank
        let pr1 = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);
        println!("默认PageRank: {:?}", pr1);

        // 测试个性化PageRank，偏向节点A
        let mut personalization = HashMap::new();
        personalization.insert("A", 1.0);
        personalization.insert("B", 0.0);
        personalization.insert("C", 0.0);

        let pr2 = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);
        println!("个性化PageRank (偏向A): {:?}", pr2);

        // A应该有最高的PageRank值
        assert!(pr2[&"A"] > pr2[&"B"]);
        assert!(pr2[&"A"] > pr2[&"C"]);
    }

    #[test]
    fn test_edge_attributes() {
        let mut graph = MultiDiGraph::new();

        let mut attrs = HashMap::new();
        attrs.insert("type".to_string(), "important".to_string());
        attrs.insert("custom_weight".to_string(), "5.0".to_string());

        let _edge_id = graph.add_edge_with_attributes("A", "B", 1.0, attrs);

        // 使用自定义权重属性进行PageRank计算
        let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, Some("custom_weight"));
        println!("使用自定义权重的PageRank: {:?}", pr);
    }
}

#[cfg(test)]
mod tests1 {
    use crate::utils::print_sorted_map;

    use super::*;
    use std::collections::HashMap;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_uniform_pagerank() {
        let mut graph = MultiDiGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);

        // 由于是均匀环图，期望每个节点的值大致相等
        let avg = 1.0 / 3.0;
        for v in &["A", "B", "C"] {
            assert!(approx_eq(*pr.get(v).unwrap(), avg, 1e-4));
        }
    }

    #[test]
    fn test_personalized_bias() {
        let mut graph = MultiDiGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        let mut personalization = HashMap::new();
        personalization.insert("A", 1.0);
        personalization.insert("B", 0.0);
        personalization.insert("C", 0.0);

        let pr = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);

        // A 拥有更高权重
        assert!(pr.get("A").unwrap() > pr.get("B").unwrap());
        assert!(pr.get("A").unwrap() > pr.get("C").unwrap());
    }

    #[test]
    fn test_multi_edges_with_personalization() {
        let mut graph = MultiDiGraph::new();

        // 多重边 A -> B
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("A", "B", 2.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        // 个性化权重：强偏向 C
        let personalization = HashMap::from([("A", 0.0), ("B", 0.0), ("C", 1.0)]);

        let pr = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);

        assert_eq!(pr.len(), 3);

        // 验证 C 应该拥有最高的 PageRank 值（因为它是唯一的 personalization 节点）
        let c_score = pr.get("C").copied().unwrap_or(0.0);
        let a_score = pr.get("A").copied().unwrap_or(0.0);
        let b_score = pr.get("B").copied().unwrap_or(0.0);

        assert!(c_score > a_score);
        assert!(c_score > b_score);

        println!(
            "PageRank scores with personalization bias to C: A={a_score:.4}, B={b_score:.4}, C={c_score:.4}"
        );
    }

    #[test]
    fn test_edge_attribute_weights() {
        let mut graph = MultiDiGraph::new();
        let mut attr1 = HashMap::new();
        attr1.insert("importance".to_string(), "2.0".to_string());
        graph.add_edge_with_attributes("A", "B", 1.0, attr1);

        let mut attr2 = HashMap::new();
        attr2.insert("importance".to_string(), "1.0".to_string());
        graph.add_edge_with_attributes("B", "C", 1.0, attr2);

        let mut attr3 = HashMap::new();
        attr3.insert("importance".to_string(), "0.5".to_string());
        graph.add_edge_with_attributes("C", "A", 1.0, attr3);

        let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, Some("importance"));

        // A的出边权重大，应体现影响力差异
        for val in pr.values() {
            assert!(*val >= 0.0);
        }
    }

    #[test]
    fn test_dangling_node() {
        let mut graph = MultiDiGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_node("C"); // 悬挂节点

        let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);

        // 所有节点都应有值
        assert_eq!(pr.len(), 3);
        for v in &["A", "B", "C"] {
            assert!(pr.get(v).is_some());
        }
    }

    #[test]
    fn test_empty_graph() {
        let graph: MultiDiGraph<&str> = MultiDiGraph::new();
        let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);
        assert!(pr.is_empty());
    }

    #[test]
    fn test_zero_personalization() {
        let mut graph = MultiDiGraph::new();
        graph.add_edge("A", "B", 1.0);

        let personalization = HashMap::from([("A", 0.0), ("B", 0.0)]);
        let pr = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);

        assert!(pr.is_empty()); // 全为0无法归一化
    }

    #[test]
    fn test_multi_edges() {
        let mut graph = MultiDiGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("A", "B", 2.0);
        graph.add_edge("B", "C", 1.0);
        graph.add_edge("C", "A", 1.0);

        let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);

        assert_eq!(pr.len(), 3);
        for v in &["A", "B", "C"] {
            assert!(pr.contains_key(v));
        }
    }

    #[test]
    fn test_convergence_behavior() {
        let mut graph = MultiDiGraph::new();
        graph.add_edge("A", "B", 1.0);
        graph.add_edge("B", "A", 1.0);

        let pr = graph.personalized_pagerank(None, 0.85, 200, 1e-12, None);

        let sum: f64 = pr.values().sum();
        assert!(approx_eq(sum, 1.0, 1e-6)); // 最终的 PageRank 应该归一化到 1
    }

    #[test]
    fn test_normal_pagerank1() {
        let mut graph = MultiDiGraph::new();

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

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);
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

    #[test]
    fn test_normal_pagerank2() {
        let mut graph = MultiDiGraph::new();

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

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);
        print_sorted_map(&result, None);

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
    }
}
