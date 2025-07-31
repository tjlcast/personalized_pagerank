use std::collections::HashMap;

use personalized_pagerank::pagerank::WeightedGraph;

fn main() {
    // 创建加权图
    let mut graph = WeightedGraph::new();
    graph.add_edge("A", "B", 1.5);
    graph.add_edge("B", "C", 2.0);

    // 标准 PageRank
    let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6);
    println!("Standard PageRank: {:?}", pr);

    // 个性化 PageRank
    let mut personalization = HashMap::new();
    personalization.insert("A", 0.8);
    personalization.insert("B", 0.2);

    let ppr = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6);
    println!("Personalized PageRank: {:?}", ppr);

    main1();
}



// 示例用法
fn main1() {
    let mut graph = WeightedGraph::new();
    
    // 构建示例图
    graph.add_edge("A", "B", 1.5);
    graph.add_edge("A", "C", 2.0);
    graph.add_edge("B", "C", 1.0);
    graph.add_edge("B", "D", 0.5);
    graph.add_edge("C", "D", 2.5);
    graph.add_edge("D", "A", 1.0);

    // 标准 PageRank
    println!("标准 PageRank:");
    let standard_pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6);
    for (node, rank) in &standard_pr {
        println!("{:?}: {:.6}", node, rank);
    }

    // 个性化 PageRank
    println!("\n个性化 PageRank (偏向节点 A):");
    let mut personalization = HashMap::new();
    personalization.insert("A", 0.7);
    personalization.insert("B", 0.1);
    personalization.insert("C", 0.1);
    personalization.insert("D", 0.1);

    let personalized_pr = graph.personalized_pagerank(
        Some(personalization), 
        0.85, 
        100, 
        1e-6
    );
    
    for (node, rank) in &personalized_pr {
        println!("{:?}: {:.6}", node, rank);
    }

    // 验证概率和为1
    let sum: f64 = personalized_pr.values().sum();
    println!("\nPageRank 值总和: {:.10}", sum);
}