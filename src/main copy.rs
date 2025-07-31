use std::collections::HashMap;
use personalized_pagerank::personalized_pagerank;

fn main() {
    // 创建边列表 (source, target, weight)
    let edges = vec![
        (0, 1, 2.0),
        (0, 2, 1.0),
        (1, 2, 1.0),
        (2, 0, 1.0),
    ];
    
    // 创建个性化向量
    let mut personalization = HashMap::new();
    personalization.insert(0, 1.0);
    
    // 计算 Personalized PageRank
    let result = personalized_pagerank(
        &edges,
        Some(&personalization),
        0.85,  // alpha (阻尼因子)
        100,   // 最大迭代次数
        1e-6,  // 收敛阈值
    );
    
    println!("PageRank 结果: {:?}", result);
}