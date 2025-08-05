use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// 边的结构体，包含目标节点、权重、唯一ID、标识符和额外属性
#[derive(Debug, Clone)]
pub struct Edge<T, I = String> {
    pub to: T,
    pub weight: f64,
    pub id: usize,
    pub ident: I,
    pub attributes: HashMap<String, String>,
}

impl<T, I> Edge<T, I> {
    pub fn new(to: T, weight: f64, id: usize, ident: I) -> Self {
        Self {
            to,
            weight,
            id,
            ident,
            attributes: HashMap::new(),
        }
    }

    pub fn with_attribute(mut self, key: String, value: String) -> Self {
        self.attributes.insert(key, value);
        self
    }
}

/// 边数据结构，用于返回边信息
#[derive(Debug, Clone)]
pub struct EdgeData<I> {
    pub weight: f64,
    pub ident: I,
    pub id: usize,
    pub attributes: HashMap<String, String>,
}

impl<I: Clone> EdgeData<I> {
    fn from_edge<T>(edge: &Edge<T, I>) -> Self {
        Self {
            weight: edge.weight,
            ident: edge.ident.clone(),
            id: edge.id,
            attributes: edge.attributes.clone(),
        }
    }
}

/// 多重有向图结构
#[derive(Debug, Clone)]
pub struct MultiDiGraph<T, I = String> {
    /// 邻接表表示，存储边的列表
    adjacency: HashMap<T, Vec<Edge<T, I>>>,
    /// 反向邻接表，用于快速查找入边
    reverse_adjacency: HashMap<T, Vec<Edge<T, I>>>,
    /// 所有节点的集合
    nodes: Vec<T>,
    /// 下一个边ID
    next_edge_id: usize,
    /// 节点属性
    node_attributes: HashMap<T, HashMap<String, String>>,
}

impl<T, I> MultiDiGraph<T, I>
where
    T: Clone + Hash + Eq + Debug,
    I: Clone + Debug,
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

    /// 添加带权重和标识符的边
    pub fn add_edge(&mut self, from: T, to: T, weight: f64, ident: I) -> usize {
        // 确保节点存在
        self.add_node(from.clone());
        self.add_node(to.clone());

        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;

        // 添加到正向邻接表
        let edge = Edge::new(to.clone(), weight, edge_id, ident.clone());
        if let Some(edges) = self.adjacency.get_mut(&from) {
            edges.push(edge);
        }

        // 添加到反向邻接表
        let reverse_edge = Edge::new(from.clone(), weight, edge_id, ident);
        if let Some(reverse_edges) = self.reverse_adjacency.get_mut(&to) {
            reverse_edges.push(reverse_edge);
        }

        edge_id
    }

    /// 添加带权重的边（使用默认标识符）
    pub fn add_edge_simple(&mut self, from: T, to: T, weight: f64) -> usize
    where
        I: Default,
    {
        self.add_edge(from, to, weight, I::default())
    }

    /// 添加带属性的边
    pub fn add_edge_with_attributes(
        &mut self,
        from: T,
        to: T,
        weight: f64,
        ident: I,
        attributes: HashMap<String, String>,
    ) -> usize {
        let edge_id = self.add_edge(from.clone(), to.clone(), weight, ident);

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

    /// 类似NetworkX的out_edges方法，支持data参数
    /// 当data=true时，返回(src, dst, data)三元组
    /// 当data=false时，返回(src, dst)二元组
    pub fn out_edges<'a>(&'a self, node: &'a T, data: bool) -> Vec<OutEdgeResult<'a, T, I>> {
        if let Some(edges) = self.adjacency.get(node) {
            edges
                .iter()
                .map(|edge| {
                    if data {
                        OutEdgeResult::WithData(node, &edge.to, EdgeData::from_edge(edge))
                    } else {
                        OutEdgeResult::WithoutData(node, &edge.to)
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// 类似NetworkX的in_edges方法
    pub fn in_edges<'a>(&'a self, node: &'a T, data: bool) -> Vec<InEdgeResult<'a, T, I>> {
        if let Some(edges) = self.reverse_adjacency.get(node) {
            edges
                .iter()
                .map(|edge| {
                    if data {
                        InEdgeResult::WithData(&edge.to, node, EdgeData::from_edge(edge))
                    } else {
                        InEdgeResult::WithoutData(&edge.to, node)
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// 获取所有边的迭代器
    pub fn edges(&self, data: bool) -> Vec<EdgeResult<T, I>> {
        let mut result = Vec::new();
        for (src, edges) in &self.adjacency {
            for edge in edges {
                if data {
                    result.push(EdgeResult::WithData(
                        src,
                        &edge.to,
                        EdgeData::from_edge(edge),
                    ));
                } else {
                    result.push(EdgeResult::WithoutData(src, &edge.to));
                }
            }
        }
        result
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

    /// 获取节点的出边（原始方法）
    pub fn get_out_edges(&self, node: &T) -> Option<&Vec<Edge<T, I>>> {
        self.adjacency.get(node)
    }

    /// 获取节点的入边（原始方法）
    pub fn get_in_edges(&self, node: &T) -> Option<&Vec<Edge<T, I>>> {
        self.reverse_adjacency.get(node)
    }

    /// 获取两个节点之间的所有边
    pub fn get_edges_between(&self, from: &T, to: &T) -> Vec<&Edge<T, I>> {
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
            let uniform_weight = 1.0 / n as f64;
            self.nodes
                .iter()
                .map(|node| (node.clone(), uniform_weight))
                .collect()
        });

        let total_personalization: f64 = personalization.values().sum();
        // 归一化个性化向量
        let normalized_personalization: HashMap<T, f64> = if total_personalization != 0.0 {
            personalization
                .iter()
                .map(|(k, v)| (k.clone(), v / total_personalization))
                .collect()
        } else {
            let uniform_weight = 1.0 / n as f64;
            self.nodes
                .iter()
                .map(|node| (node.clone(), uniform_weight))
                .collect()
        };

        // 计算出度权重
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

            // 初始化新的PageRank值
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

            // 处理悬挂节点
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

impl<T, I> Default for MultiDiGraph<T, I>
where
    T: Clone + Hash + Eq + Debug,
    I: Clone + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

/// 出边结果枚举，支持带数据和不带数据的情况
#[derive(Debug)]
pub enum OutEdgeResult<'a, T, I> {
    WithData(&'a T, &'a T, EdgeData<I>),
    WithoutData(&'a T, &'a T),
}

impl<'a, T, I> OutEdgeResult<'a, T, I> {
    /// 获取源节点
    pub fn src(&self) -> &'a T {
        match self {
            OutEdgeResult::WithData(src, _, _) => src,
            OutEdgeResult::WithoutData(src, _) => src,
        }
    }

    /// 获取目标节点
    pub fn dst(&self) -> &'a T {
        match self {
            OutEdgeResult::WithData(_, dst, _) => dst,
            OutEdgeResult::WithoutData(_, dst) => dst,
        }
    }

    /// 获取边数据（如果有的话）
    pub fn data(&self) -> Option<&EdgeData<I>> {
        match self {
            OutEdgeResult::WithData(_, _, data) => Some(data),
            OutEdgeResult::WithoutData(_, _) => None,
        }
    }
}

/// 入边结果枚举
#[derive(Debug)]
pub enum InEdgeResult<'a, T, I> {
    WithData(&'a T, &'a T, EdgeData<I>),
    WithoutData(&'a T, &'a T),
}

/// 边结果枚举
#[derive(Debug)]
pub enum EdgeResult<'a, T, I> {
    WithData(&'a T, &'a T, EdgeData<I>),
    WithoutData(&'a T, &'a T),
}

#[cfg(test)]
mod tests {
    use crate::utils::print_sorted_map;

    use super::*;

    /// 节点存在且有出边
    #[test]
    fn test_get_out_edges_some() {
        let mut graph = MultiDiGraph::new();
        let src = "A";
        graph.add_edge(src, "B", 1.0, "edge_ab".to_string());
        graph.add_edge(src, "C", 2.0, "edge_ac".to_string());
        graph.add_edge(src, "D", 3.0, "edge_ad".to_string());
        graph.add_edge(src, "A", 3.0, "edge_aa".to_string());

        let edges = graph.get_out_edges(&"A").unwrap();
        println!("get_out_edges: {:?}", edges);
        for edge in edges {
            println!("Edge: {:?}", edge);
        }

        let out_edges_1 = graph.out_edges(&src, true);
        println!("out_edges_1: {:?}", out_edges_1);
        for edge in out_edges_1 {
            if let OutEdgeResult::WithData(src_rel_fname, dist_rel_fname, data) = edge {
                println!("Edge from {:?} to {:?}: weight={}, ident={}", src_rel_fname, dist_rel_fname, data.weight, data.ident);
            }
        }

        let out_edges_2 = graph.out_edges(&src, false);
        println!("out_edges_2: {:?}", out_edges_2);

        // 断言 get_out_edges 的结果
        assert_eq!(edges.len(), 4, "Should have 4 outgoing edges from A");
    }

    #[test]
    fn test_add_edge_with_ident() {
        let mut graph: MultiDiGraph<&str, String> = MultiDiGraph::new();

        // 模拟您的使用场景
        let referencer = "main_func";
        let definer = "helper_func";
        let use_mul = 2.0;
        let num_refs = 3;
        let ident = format!("ref_{}_{}", referencer, definer);

        let edge_id = graph.add_edge(
            referencer,
            definer,
            use_mul * num_refs as f64,
            ident.clone(),
        );

        // 验证边被正确添加
        assert_eq!(graph.edge_count(), 1);

        // 测试out_edges方法
        let out_edges = graph.out_edges(&referencer, true);
        assert_eq!(out_edges.len(), 1);

        if let OutEdgeResult::WithData(src, dst, data) = &out_edges[0] {
            assert_eq!(*src, &referencer);
            assert_eq!(*dst, &definer);
            assert_eq!(data.weight, 6.0); // 2.0 * 3
            assert_eq!(data.ident, ident);
            assert_eq!(data.id, edge_id);
        } else {
            panic!("Expected WithData result");
        }
    }

    #[test]
    fn test_out_edges_iteration() {
        let mut graph: MultiDiGraph<&str, String> = MultiDiGraph::new();

        let src = "A";
        graph.add_edge(src, "B", 1.0, "edge_ab".to_string());
        graph.add_edge(src, "C", 2.0, "edge_ac".to_string());
        graph.add_edge(src, "D", 3.0, "edge_ad".to_string());

        // 类似于Python中的 for src, dst, data in G.out_edges(src, data=True):
        for edge_result in graph.out_edges(&src, true) {
            if let OutEdgeResult::WithData(src_node, dst_node, data) = edge_result {
                println!(
                    "Edge from {:?} to {:?}: weight={}, ident={}",
                    src_node, dst_node, data.weight, data.ident
                );

                // 验证源节点是正确的
                assert_eq!(src_node, &src);

                // 验证边数据
                match dst_node {
                    &"B" => {
                        assert_eq!(data.weight, 1.0);
                        assert_eq!(data.ident, "edge_ab");
                    }
                    &"C" => {
                        assert_eq!(data.weight, 2.0);
                        assert_eq!(data.ident, "edge_ac");
                    }
                    &"D" => {
                        assert_eq!(data.weight, 3.0);
                        assert_eq!(data.ident, "edge_ad");
                    }
                    _ => panic!("Unexpected destination"),
                }
            }
        }
    }

    #[test]
    fn test_out_edges_without_data() {
        let mut graph: MultiDiGraph<&str, String> = MultiDiGraph::new();

        graph.add_edge("A", "B", 1.0, "edge1".to_string());
        graph.add_edge("A", "C", 2.0, "edge2".to_string());

        // 测试不带数据的情况
        let out_edges = graph.out_edges(&"A", false);
        assert_eq!(out_edges.len(), 2);

        for edge_result in out_edges {
            if let OutEdgeResult::WithoutData(src, dst) = edge_result {
                assert_eq!(src, &"A");
                assert!(dst == &"B" || dst == &"C");
            } else {
                panic!("Expected WithoutData result");
            }
        }
    }

    #[test]
    fn test_multiple_edges_between_same_nodes() {
        let mut graph: MultiDiGraph<&str, String> = MultiDiGraph::new();

        // 在同一对节点间添加多条边
        graph.add_edge("A", "B", 1.0, "first_ref".to_string());
        graph.add_edge("A", "B", 2.0, "second_ref".to_string());
        graph.add_edge("A", "B", 0.5, "third_ref".to_string());

        let out_edges = graph.out_edges(&"A", true);
        assert_eq!(out_edges.len(), 3);

        let mut total_weight = 0.0;
        for edge_result in out_edges {
            if let OutEdgeResult::WithData(src, dst, data) = edge_result {
                assert_eq!(src, &"A");
                assert_eq!(dst, &"B");
                total_weight += data.weight;
                assert!(data.ident.contains("ref"));
            }
        }

        assert_eq!(total_weight, 3.5); // 1.0 + 2.0 + 0.5
    }

    #[test]
    fn test_pagerank_with_idents() {
        let mut graph: MultiDiGraph<&str, String> = MultiDiGraph::new();

        graph.add_edge("A", "B", 1.0, "link1".to_string());
        graph.add_edge("B", "C", 1.0, "link2".to_string());
        graph.add_edge("C", "A", 1.0, "link3".to_string());

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);

        // 验证PageRank结果
        let expected = 1.0 / 3.0;
        for (_, value) in result.iter() {
            assert!((value - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_normal_pagerank1() {
        let mut graph = MultiDiGraph::new();

        graph.add_edge("B", "A", 1.0, "B-A".to_string());
        graph.add_edge("C", "A", 1.0, "C-A".to_string());
        graph.add_edge("D", "A", 1.0, "D-A".to_string());

        graph.add_edge("B", "C", 1.0, "B-C".to_string());
        graph.add_edge("C", "D", 1.0, "C-D".to_string());
        graph.add_edge("D", "C", 1.0, "D-C".to_string());

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

    #[test]
    fn test_cycle_pagerank() {
        let mut graph = MultiDiGraph::new();

        graph.add_edge("B", "A", 1.0, "B-A".to_string());
        graph.add_edge("C", "A", 1.0, "C-A".to_string());
        graph.add_edge("D", "A", 1.0, "D-A".to_string());

        graph.add_edge("B", "C", 1.0, "B-C".to_string());
        graph.add_edge("C", "D", 1.0, "C-D".to_string());

        graph.add_edge("E", "E", 1.0, "E-E".to_string());

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("A", 0.1);
        personalization.insert("B", 0.1);
        personalization.insert("C", 0.1);
        personalization.insert("D", 0.1);
        personalization.insert("E", 0.1);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);
        print_sorted_map(&result, None);

        // A 应该有最高的 PageRank 值
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();
        let d_rank = result.get("D").unwrap();
        let e_rank = result.get("E").unwrap();

        assert!(e_rank > a_rank);
        assert!(a_rank > d_rank);
        assert!(d_rank > c_rank);

        assert!(c_rank > b_rank);
    }

    #[test]
    fn test_cycle_pagerank_with_none_personalization() {
        let mut graph = MultiDiGraph::new();

        graph.add_edge("B", "A", 1.0, "B-A".to_string());
        graph.add_edge("C", "A", 1.0, "C-A".to_string());
        graph.add_edge("D", "A", 1.0, "D-A".to_string());

        graph.add_edge("B", "C", 1.0, "B-C".to_string());
        graph.add_edge("C", "D", 1.0, "C-D".to_string());

        graph.add_edge("E", "E", 1.0, "E-E".to_string());

        let result = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);
        print_sorted_map(&result, None);

        // A 应该有最高的 PageRank 值
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();
        let d_rank = result.get("D").unwrap();
        let e_rank = result.get("E").unwrap();

        assert!(e_rank > a_rank);
        assert!(a_rank > d_rank);
        assert!(d_rank > c_rank);

        assert!(c_rank > b_rank);
    }

    #[test]
    fn test_cycle_pagerank_with_zero_personalization() {
        let mut graph = MultiDiGraph::new();

        graph.add_edge("B", "A", 1.0, "B-A".to_string());
        graph.add_edge("C", "A", 1.0, "C-A".to_string());
        graph.add_edge("D", "A", 1.0, "D-A".to_string());

        graph.add_edge("B", "C", 1.0, "B-C".to_string());
        graph.add_edge("C", "D", 1.0, "C-D".to_string());

        graph.add_edge("E", "E", 1.0, "E-E".to_string());

        // 个性化向量：更偏向节点 A
        let mut personalization = HashMap::new();
        personalization.insert("A", 0 as f64);
        personalization.insert("B", 0 as f64);
        personalization.insert("C", 0 as f64);
        personalization.insert("D", 0 as f64);
        personalization.insert("E", 0 as f64);

        let result = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);
        print_sorted_map(&result, None);

        // A 应该有最高的 PageRank 值
        let a_rank = result.get("A").unwrap();
        let b_rank = result.get("B").unwrap();
        let c_rank = result.get("C").unwrap();
        let d_rank = result.get("D").unwrap();
        let e_rank = result.get("E").unwrap();

        assert!(e_rank > a_rank);
        assert!(a_rank > d_rank);
        assert!(d_rank > c_rank);

        assert!(c_rank > b_rank);
    }
}
