use std::collections::HashMap;

use personalized_pagerank::{pagerank::WeightedGraph, pagerank_multi::MultiDiGraph};

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
    main2();
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



// 使用示例
fn main2() {
    let mut graph = MultiDiGraph::new();

    // 添加节点
    graph.add_node("页面A");
    graph.add_node("页面B");
    graph.add_node("页面C");
    graph.add_node("页面D");

    // 添加多重边（模拟多个链接）
    graph.add_edge("页面A", "页面B", 1.0);
    graph.add_edge("页面A", "页面B", 0.5); // 第二个链接，权重较小
    graph.add_edge("页面A", "页面C", 2.0);
    graph.add_edge("页面B", "页面C", 1.0);
    graph.add_edge("页面B", "页面D", 1.5);
    graph.add_edge("页面C", "页面A", 1.0);
    graph.add_edge("页面C", "页面D", 1.0);
    graph.add_edge("页面D", "页面A", 0.5);

    // 打印图信息
    graph.print_info();

    // 计算标准PageRank
    println!("\n=== 标准PageRank ===");
    let standard_pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, None);
    for (node, score) in &standard_pr {
        println!("{}: {:.4}", node, score);
    }

    // 计算个性化PageRank（偏向页面A和页面C）
    println!("\n=== 个性化PageRank (偏向页面A和页面C) ===");
    let mut personalization = HashMap::new();
    personalization.insert("页面A", 0.7);
    personalization.insert("页面B", 0.1);
    personalization.insert("页面C", 0.2);
    personalization.insert("页面D", 0.0);

    let personalized_pr = graph.personalized_pagerank(Some(personalization), 0.85, 100, 1e-6, None);

    for (node, score) in &personalized_pr {
        println!("{}: {:.4}", node, score);
    }

    // 使用边属性作为权重
    println!("\n=== 使用边属性权重的示例 ===");
    let mut graph2 = MultiDiGraph::new();

    let mut attrs1 = HashMap::new();
    attrs1.insert("importance".to_string(), "3.0".to_string());
    graph2.add_edge_with_attributes("X", "Y", 1.0, attrs1);

    let mut attrs2 = HashMap::new();
    attrs2.insert("importance".to_string(), "1.0".to_string());
    graph2.add_edge_with_attributes("Y", "X", 1.0, attrs2);

    let attr_pr = graph2.personalized_pagerank(None, 0.85, 100, 1e-6, Some("importance"));
    for (node, score) in &attr_pr {
        println!("{}: {:.4}", node, score);
    }
}
