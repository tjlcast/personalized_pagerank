# Personalized PageRank 算法实现

这是一个 Rust 实现的个性化 PageRank 算法库，包含两种图结构实现：
1. 加权图 (`WeightedGraph`) - 支持带权重的边
2. 多重有向图 (`MultiDiGraph`) - 支持多重边和边属性

## 功能特性

- **标准 PageRank 计算**
- **个性化 PageRank 计算** - 支持自定义起始点分布
- **多种图结构支持**：
  - 加权图
  - 多重有向图（支持多条边连接相同节点）
- **边属性支持** - 可以使用边属性作为权重计算
- **确定性结果** - 通过排序确保算法确定性
- **高效实现** - 使用邻接表存储图结构

## 快速开始

### 安装

将以下内容添加到你的 `Cargo.toml`:

```toml
[dependencies]
personalized_pagerank = { path = "./path_to_crate" }
```

### 使用示例

#### 1. 加权图示例

```rust
use std::collections::HashMap;
use personalized_pagerank::pagerank::WeightedGraph;

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
```

#### 2. 多重有向图示例

```rust
use std::collections::HashMap;
use personalized_pagerank::pagerank_multi::MultiDiGraph;

let mut graph = MultiDiGraph::new();

// 添加节点和边
graph.add_edge("页面A", "页面B", 1.0);
graph.add_edge("页面A", "页面B", 0.5); // 第二个链接
graph.add_edge("页面A", "页面C", 2.0);

// 使用边属性作为权重
let mut attrs = HashMap::new();
attrs.insert("importance".to_string(), "3.0".to_string());
graph.add_edge_with_attributes("X", "Y", 1.0, attrs);

// 计算 PageRank
let pr = graph.personalized_pagerank(None, 0.85, 100, 1e-6, Some("importance"));
```

## API 文档

### WeightedGraph

#### 主要方法

- `new()` - 创建新图
- `add_node(node)` - 添加节点
- `add_edge(from, to, weight)` - 添加带权重的边
- `personalized_pagerank(personalization, alpha, max_iter, tolerance)` - 计算 PageRank

#### 设计
WeightedGraph 使用邻接表存储图结构，每个节点包含一个邻接节点列表和权重列表。

Personalized PageRank 计算过程
``` mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '10px'}, 'config': {'width': 400, 'height': 300}}}%%
flowchart TD
    A[开始] --> B[排序所有节点和边]
    B --> C[初始化个性化向量]
    C --> D[归一化个性化向量]
    D --> E[计算出度权重和]
    E --> F[初始化PageRank值为均匀分布]
    F --> G[开始迭代]
    
    G --> H{迭代次数\n< max_iter?}
    H -->|是| I[初始化新PR值为个性化贡献]
    I --> J[按顺序处理每个源节点]
    J --> K[获取源节点的PR值和出度权重]
    K --> L{出度权重>0?}
    L -->|是| M[排序并处理每条出边]
    M --> N[计算转移贡献:\nalpha*PR*weight/out_weight]
    N --> O[累加到目标节点的PR值]
    L -->|否| P[累计悬挂节点质量]
    
    O --> Q[处理下一个源节点]
    P --> Q
    Q --> R[所有源节点处理完毕?]
    R -->|否| J
    R -->|是| S[按个性化向量分配悬挂质量]
    S --> T[计算PR值变化量]
    T --> U{变化量 < tolerance?}
    U -->|是| V[返回最终PR值]
    U -->|否| G
    H -->|否| V
```

### MultiDiGraph

#### 主要方法

- `new()` - 创建新图
- `add_node(node)` - 添加节点
- `add_edge(from, to, weight)` - 添加边
- `add_edge_with_attributes(from, to, weight, attributes)` - 添加带属性的边
- `personalized_pagerank(personalization, alpha, max_iter, tolerance, weight_key)` - 计算 PageRank
  - `weight_key`: 可选，指定使用哪个边属性作为权重

## 测试

项目包含全面的单元测试，覆盖以下场景：

- 基本 PageRank 计算
- 悬挂节点处理
- 不连通组件
- 个性化 PageRank
- 边属性权重
- 多重边处理

运行测试：

```bash
cargo test
```

## 性能考虑

- 使用邻接表存储图结构，空间效率高
- 迭代计算时使用确定性排序，确保结果可重现
- 支持提前终止（当变化小于容差时）

## 贡献

欢迎提交 issue 和 pull request。请确保所有变更都包含相应的测试。

## 许可证

Apache2.0 License
