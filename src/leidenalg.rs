// src/leidenalg1.rs
use crate::pagerank_multi1::MultiDiGraph;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// 社区划分结果
#[derive(Debug, Clone)]
pub struct LeidenPartition<T> {
    /// 每个节点对应的社区ID
    pub membership: HashMap<T, usize>,
    /// 社区数量
    pub num_communities: usize,
}

/// Leiden算法实现
pub struct LeidenCommunityDetection;

impl LeidenCommunityDetection {
    /// 使用Leiden算法进行社区检测
    pub fn find_partition<T>(
        graph: &MultiDiGraph<T, String>,
        resolution: Option<f64>,
        randomness: Option<f64>,
        iterations: Option<usize>,
    ) -> LeidenPartition<T>
    where
        T: Clone + Hash + Eq + std::fmt::Debug + Ord,
    {
        let resolution = resolution.unwrap_or(1.0);
        let randomness = randomness.unwrap_or(0.01);
        let iterations = iterations.unwrap_or(2);

        // 初始化：每个节点属于自己的社区
        let mut membership: HashMap<T, usize> = HashMap::new();
        let mut community_nodes: HashMap<usize, HashSet<T>> = HashMap::new();

        for (index, node) in graph.nodes().iter().enumerate() {
            membership.insert(node.clone(), index);
            community_nodes
                .entry(index)
                .or_default()
                .insert(node.clone());
        }

        let mut rng = rand::thread_rng();

        // 计算总边权重
        let total_weight = Self::calculate_total_weight(graph);

        // 进行多次迭代
        for iteration in 0..iterations {
            println!(
                "Leiden迭代 {}: 初始社区数量 = {}",
                iteration + 1,
                community_nodes.len()
            );

            // 第一阶段：局部移动节点以优化模块度
            let mut improvement = true;
            let mut moves = 0;

            while improvement {
                improvement = false;

                // 随机打乱节点顺序
                let mut nodes: Vec<T> = graph.nodes().clone();
                shuffle(&mut nodes, &mut rng);

                for node in &nodes {
                    let current_community = *membership.get(node).unwrap();

                    // 计算移动到邻居社区的质量增益
                    let best_community = Self::find_best_community_move(
                        graph,
                        node,
                        &membership,
                        &community_nodes,
                        resolution,
                        randomness,
                        total_weight,
                        &mut rng,
                    );

                    if best_community != current_community {
                        // 移动节点到新社区
                        membership.insert(node.clone(), best_community);

                        // 更新社区节点集合
                        if let Some(nodes_in_old_community) =
                            community_nodes.get_mut(&current_community)
                        {
                            nodes_in_old_community.remove(node);
                            if nodes_in_old_community.is_empty() {
                                community_nodes.remove(&current_community);
                            }
                        }
                        community_nodes
                            .entry(best_community)
                            .or_default()
                            .insert(node.clone());

                        improvement = true;
                        moves += 1;
                    }
                }
            }

            println!("  局部移动阶段完成，移动了 {} 个节点", moves);

            // 第二阶段：聚合社区并递归应用
            if iteration < iterations - 1 {
                let (aggregated_graph, node_to_supernode) =
                    Self::aggregate_communities(graph, &membership);

                // 递归应用Leiden算法到聚合图
                let super_partition = Self::find_partition(
                    &aggregated_graph,
                    Some(resolution),
                    Some(randomness),
                    Some(1),
                );

                // 更新原始图的社区分配
                let mut new_membership = HashMap::new();
                let mut new_community_nodes: HashMap<usize, HashSet<T>> = HashMap::new();
                let mut next_community_id = 0;
                let mut community_mapping: HashMap<usize, usize> = HashMap::new();

                for (node, old_super_community) in membership {
                    if let Some(supernode) = node_to_supernode.get(&old_super_community) {
                        if let Some(&new_super_community) =
                            super_partition.membership.get(supernode)
                        {
                            let final_community = *community_mapping
                                .entry(new_super_community)
                                .or_insert_with(|| {
                                    let id = next_community_id;
                                    next_community_id += 1;
                                    id
                                });

                            new_membership.insert(node.clone(), final_community);
                            new_community_nodes
                                .entry(final_community)
                                .or_default()
                                .insert(node);
                        }
                    }
                }

                membership = new_membership;
                community_nodes = new_community_nodes;

                println!("  聚合阶段完成，新社区数量 = {}", community_nodes.len());
            }
        }

        // 重新编号社区以确保连续性
        let mut unique_communities: Vec<usize> = membership.values().cloned().collect();
        unique_communities.sort_unstable();
        unique_communities.dedup();

        let community_mapping: HashMap<usize, usize> = unique_communities
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id))
            .collect();

        let final_membership: HashMap<T, usize> = membership
            .into_iter()
            .map(|(node, community)| (node, *community_mapping.get(&community).unwrap()))
            .collect();

        let num_communities = community_mapping.len();
        println!("Leiden算法完成: 最终社区数量 = {}", num_communities);

        LeidenPartition {
            membership: final_membership,
            num_communities,
        }
    }

    /// 计算图的总边权重
    fn calculate_total_weight<T>(graph: &MultiDiGraph<T, String>) -> f64
    where
        T: Clone + Hash + Eq + std::fmt::Debug + Ord,
    {
        let mut total = 0.0;
        for node in graph.nodes() {
            if let Some(edges) = graph.get_out_edges(node) {
                for edge in edges {
                    total += edge.weight;
                }
            }
        }
        total
    }

    /// 找到节点移动到的最佳社区
    fn find_best_community_move<T, R>(
        graph: &MultiDiGraph<T, String>,
        node: &T,
        membership: &HashMap<T, usize>,
        community_nodes: &HashMap<usize, HashSet<T>>,
        resolution: f64,
        randomness: f64,
        total_weight: f64,
        rng: &mut R,
    ) -> usize
    where
        T: Clone + Hash + Eq + std::fmt::Debug + Ord,
        R: Rng,
    {
        let current_community = *membership.get(node).unwrap();

        // 收集所有邻居社区（包括当前社区）
        let mut candidate_communities = HashSet::new();
        candidate_communities.insert(current_community);

        // 添加所有邻居节点所在的社区
        if let Some(edges) = graph.get_out_edges(node) {
            for edge in edges {
                let neighbor = &edge.to;
                if let Some(&neighbor_community) = membership.get(neighbor) {
                    candidate_communities.insert(neighbor_community);
                }
            }
        }

        // 计算移动到各社区的模块度增益
        let mut best_community = current_community;
        let mut best_gain = f64::NEG_INFINITY;

        for &community in &candidate_communities {
            let gain = Self::calculate_modularity_gain(
                graph,
                node,
                community,
                membership,
                community_nodes,
                resolution,
                total_weight,
            ) + randomness * rng.gen_range(-1.0..1.0);

            if gain > best_gain {
                best_gain = gain;
                best_community = community;
            }
        }

        best_community
    }

    /// 计算模块度增益 - 使用正确的模块度公式
    fn calculate_modularity_gain<T>(
        graph: &MultiDiGraph<T, String>,
        node: &T,
        target_community: usize,
        membership: &HashMap<T, usize>,
        community_nodes: &HashMap<usize, HashSet<T>>,
        resolution: f64,
        total_weight: f64,
    ) -> f64
    where
        T: Clone + Hash + Eq + std::fmt::Debug + Ord,
    {
        if total_weight == 0.0 {
            return 0.0;
        }

        let current_community = *membership.get(node).unwrap();

        if target_community == current_community {
            return 0.0;
        }

        // 计算节点与目标社区的连接权重
        let mut connection_to_target = 0.0;
        // 计算节点与当前社区的连接权重
        let mut connection_to_current = 0.0;

        if let Some(edges) = graph.get_out_edges(node) {
            for edge in edges {
                let neighbor = &edge.to;
                if let Some(&neighbor_community) = membership.get(neighbor) {
                    if neighbor_community == target_community {
                        connection_to_target += edge.weight;
                    }
                    if neighbor_community == current_community {
                        connection_to_current += edge.weight;
                    }
                }
            }
        }

        // 计算节点的总出度权重
        let node_out_degree = graph.out_degree_weight(node);
        // 计算目标社区的总出度权重
        let target_community_out_degree: f64 = community_nodes
            .get(&target_community)
            .map(|nodes| nodes.iter().map(|n| graph.out_degree_weight(n)).sum())
            .unwrap_or(0.0);
        // 计算当前社区的总出度权重
        let current_community_out_degree: f64 = community_nodes
            .get(&current_community)
            .map(|nodes| nodes.iter().map(|n| graph.out_degree_weight(n)).sum())
            .unwrap_or(0.0);

        // 正确的模块度增益计算
        let gain_to_target = connection_to_target / total_weight
            - resolution * (node_out_degree * target_community_out_degree)
                / (total_weight * total_weight);

        let gain_from_current = connection_to_current / total_weight
            - resolution * (node_out_degree * current_community_out_degree)
                / (total_weight * total_weight);

        gain_to_target - gain_from_current
    }

    /// 聚合社区：将同一社区的节点合并为超节点
    fn aggregate_communities<T>(
        graph: &MultiDiGraph<T, String>,
        membership: &HashMap<T, usize>,
    ) -> (MultiDiGraph<usize, String>, HashMap<usize, usize>)
    where
        T: Clone + Hash + Eq + std::fmt::Debug + Ord,
    {
        let mut aggregated_graph = MultiDiGraph::new();
        let mut node_to_supernode: HashMap<usize, usize> = HashMap::new();
        let mut supernode_counter = 0;

        // 创建社区到超节点的映射
        let communities: HashSet<usize> = membership.values().cloned().collect();
        for &community in &communities {
            node_to_supernode.insert(community, supernode_counter);
            aggregated_graph.add_node(supernode_counter);
            supernode_counter += 1;
        }

        // 计算社区间的连接权重
        let mut community_edges: HashMap<(usize, usize), f64> = HashMap::new();

        for node in graph.nodes() {
            let source_community = membership.get(node).unwrap();
            let source_supernode = node_to_supernode.get(source_community).unwrap();

            if let Some(edges) = graph.get_out_edges(node) {
                for edge in edges {
                    let target = &edge.to;
                    let weight = edge.weight;

                    if let Some(&target_community) = membership.get(target) {
                        let target_supernode = node_to_supernode.get(&target_community).unwrap();
                        let edge_key = (*source_supernode, *target_supernode);
                        *community_edges.entry(edge_key).or_insert(0.0) += weight;
                    }
                }
            }
        }

        // 添加社区间的边到聚合图
        for ((from, to), weight) in community_edges {
            if weight > 0.0 {
                aggregated_graph.add_edge(
                    from,
                    to,
                    weight,
                    format!("community_{}_to_{}", from, to),
                );
            }
        }

        (aggregated_graph, node_to_supernode)
    }
}

/// 随机打乱数组
fn shuffle<T, R>(vec: &mut [T], rng: &mut R)
where
    R: Rng,
{
    for i in (1..vec.len()).rev() {
        let j = rng.gen_range(0..=i);
        vec.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leiden_community_detection() {
        // 创建测试图：两个明显的社区
        let mut graph = MultiDiGraph::new();

        // 第一个社区 (A, B, C) - 强连接
        graph.add_edge("A", "B", 3.0, "A-B".to_string());
        graph.add_edge("B", "A", 3.0, "B-A".to_string());
        graph.add_edge("B", "C", 3.0, "B-C".to_string());
        graph.add_edge("C", "B", 3.0, "C-B".to_string());
        graph.add_edge("C", "A", 3.0, "C-A".to_string());
        graph.add_edge("A", "C", 3.0, "A-C".to_string());

        // 第二个社区 (D, E, F) - 强连接
        graph.add_edge("D", "E", 3.0, "D-E".to_string());
        graph.add_edge("E", "D", 3.0, "E-D".to_string());
        graph.add_edge("E", "F", 3.0, "E-F".to_string());
        graph.add_edge("F", "E", 3.0, "F-E".to_string());
        graph.add_edge("F", "D", 3.0, "F-D".to_string());
        graph.add_edge("D", "F", 3.0, "D-F".to_string());

        // 社区间的弱连接
        graph.add_edge("A", "D", 0.1, "A-D".to_string());
        graph.add_edge("C", "F", 0.1, "C-F".to_string());

        println!("图信息:");
        graph.print_info();

        // 运行Leiden算法
        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        println!("Leiden社区检测结果:");
        println!("社区数量: {}", partition.num_communities);
        println!("节点社区分配:");

        let mut communities: HashMap<usize, Vec<&str>> = HashMap::new();
        for (node, community) in &partition.membership {
            communities.entry(*community).or_default().push(node);
        }

        for (community_id, nodes) in communities {
            println!("  社区 {}: {:?}", community_id, nodes);
        }

        // 验证结果
        assert!(partition.num_communities >= 2, "应该至少检测到2个社区");

        // 验证同一社区内的节点应该被分到同一个社区
        let community_a = partition.membership.get("A").unwrap();
        let community_b = partition.membership.get("B").unwrap();
        let community_c = partition.membership.get("C").unwrap();

        let community_d = partition.membership.get("D").unwrap();
        let community_e = partition.membership.get("E").unwrap();
        let community_f = partition.membership.get("F").unwrap();

        // A, B, C 应该在同一个社区
        assert_eq!(community_a, community_b);
        assert_eq!(community_b, community_c);

        // D, E, F 应该在同一个社区
        assert_eq!(community_d, community_e);
        assert_eq!(community_e, community_f);

        // 两个主要社区应该不同
        assert_ne!(community_a, community_d);
    }

    #[test]
    fn test_leiden_single_community() {
        // 测试完全连接的图（应该只有一个社区）
        let mut graph = MultiDiGraph::new();

        graph.add_edge("A", "B", 1.0, "A-B".to_string());
        graph.add_edge("B", "C", 1.0, "B-C".to_string());
        graph.add_edge("C", "A", 1.0, "C-A".to_string());
        graph.add_edge("A", "C", 1.0, "A-C".to_string());
        graph.add_edge("B", "A", 1.0, "B-A".to_string());
        graph.add_edge("C", "B", 1.0, "C-B".to_string());

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        println!("单社区测试结果:");
        println!("社区数量: {}", partition.num_communities);

        let communities: HashSet<usize> = partition.membership.values().cloned().collect();
        assert_eq!(communities.len(), 1, "完全连接的图应该只有一个社区");
    }

    #[test]
    fn test_leiden_disconnected_components() {
        // 测试完全不连接的组件
        let mut graph = MultiDiGraph::new();

        // 三个孤立的节点
        graph.add_node("A");
        graph.add_node("B");
        graph.add_node("C");

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        println!("断开连接测试结果:");
        println!("社区数量: {}", partition.num_communities);

        // 每个孤立节点应该在自己的社区
        assert_eq!(partition.num_communities, 3);

        let communities: HashSet<usize> = partition.membership.values().cloned().collect();
        assert_eq!(communities.len(), 3);
    }

    #[test]
    fn test_modularity_calculation() {
        // 测试模块度计算
        let mut graph = MultiDiGraph::new();

        // 创建两个明显社区
        graph.add_edge("A", "B", 2.0, "A-B".to_string());
        graph.add_edge("B", "A", 2.0, "B-A".to_string());
        graph.add_edge("B", "C", 2.0, "B-C".to_string());
        graph.add_edge("C", "B", 2.0, "C-B".to_string());

        graph.add_edge("D", "E", 2.0, "D-E".to_string());
        graph.add_edge("E", "D", 2.0, "E-D".to_string());
        graph.add_edge("E", "F", 2.0, "E-F".to_string());
        graph.add_edge("F", "E", 2.0, "F-E".to_string());

        let total_weight = LeidenCommunityDetection::calculate_total_weight(&graph);
        assert!(total_weight > 0.0, "总权重应该大于0");
    }
}

#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    /// 测试辅助函数：验证社区划分的合理性
    fn validate_partition<T: Clone + Hash + Eq + std::fmt::Debug>(
        graph: &MultiDiGraph<T, String>,
        partition: &LeidenPartition<T>,
    ) -> bool {
        // 1. 每个节点都应该有社区分配
        for node in graph.nodes() {
            if !partition.membership.contains_key(node) {
                return false;
            }
        }

        // 2. 社区ID应该是连续的
        let mut community_ids: Vec<usize> = partition.membership.values().cloned().collect();
        community_ids.sort();
        community_ids.dedup();

        if community_ids.len() != partition.num_communities {
            return false;
        }

        // 3. 社区ID应该从0开始连续编号
        for (expected, &actual) in (0..partition.num_communities).zip(community_ids.iter()) {
            if expected != actual {
                return false;
            }
        }

        true
    }

    #[test]
    fn test_basic_community_structure() {
        // 测试基本的社区结构检测
        let mut graph = MultiDiGraph::new();

        // 社区1: 强连接的三角结构
        graph.add_edge("A", "B", 5.0, "A-B".to_string());
        graph.add_edge("B", "A", 5.0, "B-A".to_string());
        graph.add_edge("B", "C", 5.0, "B-C".to_string());
        graph.add_edge("C", "B", 5.0, "C-B".to_string());
        graph.add_edge("C", "A", 5.0, "C-A".to_string());
        graph.add_edge("A", "C", 5.0, "A-C".to_string());

        // 社区2: 强连接的链式结构
        graph.add_edge("D", "E", 5.0, "D-E".to_string());
        graph.add_edge("E", "D", 5.0, "E-D".to_string());
        graph.add_edge("E", "F", 5.0, "E-F".to_string());
        graph.add_edge("F", "E", 5.0, "F-E".to_string());
        graph.add_edge("F", "G", 5.0, "F-G".to_string());
        graph.add_edge("G", "F", 5.0, "G-F".to_string());

        // 社区间的弱连接
        graph.add_edge("A", "D", 0.5, "A-D".to_string());
        graph.add_edge("C", "G", 0.5, "C-G".to_string());

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        assert!(validate_partition(&graph, &partition));
        assert_eq!(partition.num_communities, 2);

        // 验证社区划分的合理性
        let comm_a = partition.membership.get("A").unwrap();
        let comm_b = partition.membership.get("B").unwrap();
        let comm_c = partition.membership.get("C").unwrap();
        let comm_d = partition.membership.get("D").unwrap();
        let comm_e = partition.membership.get("E").unwrap();
        let comm_f = partition.membership.get("F").unwrap();
        let comm_g = partition.membership.get("G").unwrap();

        assert_eq!(comm_a, comm_b);
        assert_eq!(comm_b, comm_c);
        assert_eq!(comm_d, comm_e);
        assert_eq!(comm_e, comm_f);
        assert_eq!(comm_f, comm_g);
        assert_ne!(comm_a, comm_d);
    }

    #[test]
    fn test_weighted_edges_community() {
        // 测试带权边对社区检测的影响
        let mut graph = MultiDiGraph::new();

        // 社区1: 强权重连接
        graph.add_edge("A", "B", 10.0, "A-B-strong".to_string());
        graph.add_edge("B", "A", 10.0, "B-A-strong".to_string());

        // 社区2: 中等权重连接
        graph.add_edge("C", "D", 5.0, "C-D-medium".to_string());
        graph.add_edge("D", "C", 5.0, "D-C-medium".to_string());

        // 跨社区的弱权重连接
        graph.add_edge("A", "C", 1.0, "A-C-weak".to_string());
        graph.add_edge("B", "D", 1.0, "B-D-weak".to_string());

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        assert!(validate_partition(&graph, &partition));
        assert_eq!(partition.num_communities, 2);

        // 验证权重影响
        let comm_a = partition.membership.get("A").unwrap();
        let comm_b = partition.membership.get("B").unwrap();
        let comm_c = partition.membership.get("C").unwrap();
        let comm_d = partition.membership.get("D").unwrap();

        assert_eq!(comm_a, comm_b);
        assert_eq!(comm_c, comm_d);
        assert_ne!(comm_a, comm_c);
    }

    #[test]
    fn test_directed_graph_communities() {
        // 测试有向图的社区检测
        let mut graph = MultiDiGraph::new();

        // 创建有向社区结构
        // 社区1: A -> B -> C -> A (环状)
        graph.add_edge("A", "B", 3.0, "A->B".to_string());
        graph.add_edge("B", "C", 3.0, "B->C".to_string());
        graph.add_edge("C", "A", 3.0, "C->A".to_string());

        // 社区2: D -> E -> F -> D (环状)
        graph.add_edge("D", "E", 3.0, "D->E".to_string());
        graph.add_edge("E", "F", 3.0, "E->F".to_string());
        graph.add_edge("F", "D", 3.0, "F->D".to_string());

        // 有向的跨社区连接
        graph.add_edge("A", "D", 0.5, "A->D".to_string());
        graph.add_edge("C", "F", 0.5, "C->F".to_string());

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        assert!(validate_partition(&graph, &partition));
        assert_eq!(partition.num_communities, 2);

        let comm_a = partition.membership.get("A").unwrap();
        let comm_b = partition.membership.get("B").unwrap();
        let comm_c = partition.membership.get("C").unwrap();
        let comm_d = partition.membership.get("D").unwrap();
        let comm_e = partition.membership.get("E").unwrap();
        let comm_f = partition.membership.get("F").unwrap();

        assert_eq!(comm_a, comm_b);
        assert_eq!(comm_b, comm_c);
        assert_eq!(comm_d, comm_e);
        assert_eq!(comm_e, comm_f);
        assert_ne!(comm_a, comm_d);
    }

    #[test]
    fn test_multiple_edges_between_nodes() {
        // 测试多重边的情况
        let mut graph = MultiDiGraph::new();

        // 节点A和B之间有多个连接（模拟强关系）
        graph.add_edge("A", "B", 2.0, "A-B-1".to_string());
        graph.add_edge("A", "B", 2.0, "A-B-2".to_string());
        graph.add_edge("B", "A", 2.0, "B-A-1".to_string());
        graph.add_edge("B", "A", 2.0, "B-A-2".to_string());

        graph.add_edge("C", "D", 2.0, "C-D-1".to_string());
        graph.add_edge("C", "D", 2.0, "C-D-2".to_string());
        graph.add_edge("D", "C", 2.0, "D-C-1".to_string());
        graph.add_edge("D", "C", 2.0, "D-C-2".to_string());

        // 跨社区的单个弱连接
        graph.add_edge("A", "C", 0.5, "A-C".to_string());

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        assert!(validate_partition(&graph, &partition));
        assert_eq!(partition.num_communities, 2);

        // 多重边应该增强社区内部连接
        let comm_a = partition.membership.get("A").unwrap();
        let comm_b = partition.membership.get("B").unwrap();
        let comm_c = partition.membership.get("C").unwrap();
        let comm_d = partition.membership.get("D").unwrap();

        assert_eq!(comm_a, comm_b);
        assert_eq!(comm_c, comm_d);
        assert_ne!(comm_a, comm_c);
    }

    #[test]
    fn test_resolution_parameter_effect() {
        // 测试分辨率参数对社区检测的影响
        let mut graph = MultiDiGraph::new();

        // 创建层次化社区结构
        // 大社区内的子社区
        graph.add_edge("A", "B", 8.0, "A-B-strong".to_string());
        graph.add_edge("B", "A", 8.0, "B-A-strong".to_string());
        graph.add_edge("B", "C", 8.0, "B-C-strong".to_string());
        graph.add_edge("C", "B", 8.0, "C-B-strong".to_string());

        graph.add_edge("D", "E", 8.0, "D-E-strong".to_string());
        graph.add_edge("E", "D", 8.0, "E-D-strong".to_string());
        graph.add_edge("E", "F", 8.0, "E-F-strong".to_string());
        graph.add_edge("F", "E", 8.0, "F-E-strong".to_string());

        // 连接两个子社区
        graph.add_edge("C", "D", 3.0, "C-D-medium".to_string());
        graph.add_edge("D", "C", 3.0, "D-C-medium".to_string());

        // 低分辨率：应该检测到1个大社区
        let partition_low_res = LeidenCommunityDetection::find_partition(
            &graph,
            Some(0.5), // 低分辨率
            None,
            None,
        );

        // 高分辨率：应该检测到2个子社区
        let partition_high_res = LeidenCommunityDetection::find_partition(
            &graph,
            Some(2.0), // 高分辨率
            None,
            None,
        );

        assert!(validate_partition(&graph, &partition_low_res));
        assert!(validate_partition(&graph, &partition_high_res));

        // 低分辨率应该产生更少的社区
        assert!(partition_low_res.num_communities <= partition_high_res.num_communities);
    }

    #[test]
    fn test_empty_graph() {
        // 测试空图
        let graph: MultiDiGraph<&str, String> = MultiDiGraph::new();

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        assert!(validate_partition(&graph, &partition));
        assert_eq!(partition.num_communities, 0);
        assert!(partition.membership.is_empty());
    }

    #[test]
    fn test_single_node_graph() {
        // 测试单节点图
        let mut graph = MultiDiGraph::new();
        graph.add_node("A");

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        assert!(validate_partition(&graph, &partition));
        assert_eq!(partition.num_communities, 1);
        assert_eq!(partition.membership.get("A"), Some(&0));
    }

    #[test]
    fn test_self_loops() {
        // 测试自环的情况
        let mut graph = MultiDiGraph::new();

        graph.add_edge("A", "A", 2.0, "A-self".to_string()); // 自环
        graph.add_edge("A", "B", 3.0, "A-B".to_string());
        graph.add_edge("B", "A", 3.0, "B-A".to_string());
        graph.add_edge("B", "C", 3.0, "B-C".to_string());
        graph.add_edge("C", "B", 3.0, "C-B".to_string());

        graph.add_edge("D", "E", 3.0, "D-E".to_string());
        graph.add_edge("E", "D", 3.0, "E-D".to_string());

        let partition = LeidenCommunityDetection::find_partition(&graph, None, None, None);

        assert!(validate_partition(&graph, &partition));

        // 自环不应该阻止合理的社区检测
        let comm_a = partition.membership.get("A").unwrap();
        let comm_b = partition.membership.get("B").unwrap();
        let comm_c = partition.membership.get("C").unwrap();

        assert_eq!(comm_a, comm_b);
        assert_eq!(comm_b, comm_c);
    }
}
