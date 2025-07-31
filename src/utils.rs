use std::collections::HashMap;
use std::fmt::Debug;


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