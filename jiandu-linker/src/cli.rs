use std::io::{self, BufRead, Write};
use crate::models::{BambooSlip, LinkResult};
use crate::scoring::ScoringEngine;
use crate::fragment_matching::FragmentMatcher;

pub struct InteractiveCli {
    engine: ScoringEngine,
    slips: Vec<BambooSlip>,
    current_order: Vec<BambooSlip>,
}

impl InteractiveCli {
    pub fn new(engine: ScoringEngine, slips: Vec<BambooSlip>) -> Self {
        let current_order = slips.clone();
        InteractiveCli {
            engine,
            slips,
            current_order,
        }
    }

    pub fn run(&mut self) -> LinkResult {
        println!("=== 简牍编连交互模式 ===");
        println!("输入 'help' 查看命令列表");
        println!();

        let stdin = io::stdin();
        loop {
            print!("jiandu> ");
            io::stdout().flush().unwrap();

            let mut line = String::new();
            stdin.lock().read_line(&mut line).unwrap();
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            let cmd = parts[0];
            let arg = if parts.len() > 1 { parts[1] } else { "" };

            match cmd {
                "help" => self.print_help(),
                "list" | "ls" => self.print_current_order(),
                "show" => self.show_slip_detail(arg),
                "score" => self.print_current_score(),
                "move" => self.move_slip(arg),
                "insert-after" => self.insert_after(arg),
                "insert-before" => self.insert_before(arg),
                "swap" => self.swap_slips(arg),
                "auto" => self.auto_order(),
                "reset" => self.reset_order(),
                "weights" => self.show_weights(),
                "set-weights" => self.set_weights(arg),
                "top-scores" => self.show_top_pair_scores(arg),
                "fragment-match" | "fm" => self.show_fragment_matches(arg),
                "done" | "exit" | "quit" => {
                    println!("退出交互模式，输出当前编连方案");
                    break;
                }
                _ => println!("未知命令: {cmd}，输入 'help' 查看帮助"),
            }
            println!();
        }

        self.engine.compute_order_score(&self.current_order)
    }

    fn print_help(&self) {
        println!("可用命令:");
        println!("  list / ls        - 显示当前简牍顺序");
        println!("  show <简号>       - 显示指定简牍的详细信息");
        println!("  score            - 显示当前编连方案的评分");
        println!("  move <简号> <目标位置> - 将简牍移动到指定位置（从1开始计数）");
        println!("  insert-after <简号1> <简号2> - 将简1插入到简2之后");
        println!("  insert-before <简号1> <简号2> - 将简1插入到简2之前");
        println!("  swap <简号1> <简号2> - 交换两个简牍的位置");
        println!("  auto             - 使用自动编连算法重新排序");
        println!("  reset            - 重置为原始顺序");
        println!("  weights          - 显示当前评分权重");
        println!("  set-weights <字形> <残笔> <语法> - 设置评分权重");
        println!("  top-scores [N]   - 显示前N对最高评分的相邻组合（默认10）");
        println!("  fragment-match [N] / fm [N] - 显示前N对残片特征高级匹配结果（默认10）");
        println!("  done / exit      - 退出并输出结果");
    }

    fn print_current_order(&self) {
        println!("当前编连顺序（共 {} 枚简）:", self.current_order.len());
        for (i, slip) in self.current_order.iter().enumerate() {
            println!("  {:3}. [{}] {} ({})",
                i + 1,
                slip.slip_id,
                slip.glyph_features,
                slip.grammar_tag
            );
        }
    }

    fn show_slip_detail(&self, slip_id: &str) {
        if slip_id.is_empty() {
            println!("请指定简号");
            return;
        }
        if let Some(slip) = self.slips.iter().find(|s| s.slip_id == slip_id) {
            println!("简号: {}", slip.slip_id);
            println!("字形特征: {}", slip.glyph_features);
            println!("残笔描述: {}", slip.remnant_strokes);
            println!("语法标签: {}", slip.grammar_tag);
        } else {
            println!("未找到简号为 '{}' 的简牍", slip_id);
        }
    }

    fn print_current_score(&self) {
        let result = self.engine.compute_order_score(&self.current_order);
        println!("当前编连方案评分:");
        println!("  总分: {:.4}", result.total_score);
        if !result.link_scores.is_empty() {
            let avg = result.total_score / result.link_scores.len() as f64;
            println!("  平均相邻分: {:.4}", avg);
        }
        if !result.link_scores.is_empty() {
            println!("  相邻详情:");
            for (i, score) in result.link_scores.iter().enumerate() {
                println!("    {:2}. [{}] -> [{}]: 总分={:.4} (字形={:.4}, 残笔={:.4}, 语法={:.4})",
                    i + 1,
                    score.from_id,
                    score.to_id,
                    score.total_score,
                    score.glyph_score,
                    score.stroke_score,
                    score.grammar_score
                );
            }
        }
    }

    fn find_slip_index(&self, slip_id: &str) -> Option<usize> {
        self.current_order.iter().position(|s| s.slip_id == slip_id)
    }

    fn print_current_total_score(&self) {
        let result = self.engine.compute_order_score(&self.current_order);
        println!("  当前编连总分: {:.4}", result.total_score);
    }

    fn move_slip(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.len() != 2 {
            println!("用法: move <简号> <目标位置>");
            return;
        }

        let slip_id = parts[0];
        let target_pos: usize = match parts[1].parse::<usize>() {
            Ok(n) if n >= 1 => n - 1,
            _ => {
                println!("无效的目标位置");
                return;
            }
        };

        if let Some(from_idx) = self.find_slip_index(slip_id) {
            let target_idx = target_pos.min(self.current_order.len().saturating_sub(1));
            let slip = self.current_order.remove(from_idx);
            let insert_idx = if target_idx > from_idx { target_idx - 1 } else { target_idx };
            self.current_order.insert(insert_idx, slip);
            println!("已将 [{}] 移动到第 {} 位", slip_id, insert_idx + 1);
            self.print_current_total_score();
        } else {
            println!("未找到简号为 '{}' 的简牍", slip_id);
        }
    }

    fn insert_after(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.len() != 2 {
            println!("用法: insert-after <要移动的简> <目标简>");
            return;
        }

        let move_id = parts[0];
        let target_id = parts[1];

        if let (Some(from_idx), Some(target_idx)) = (self.find_slip_index(move_id), self.find_slip_index(target_id)) {
            let slip = self.current_order.remove(from_idx);
            let insert_idx = if from_idx < target_idx { target_idx } else { target_idx + 1 };
            self.current_order.insert(insert_idx, slip);
            println!("已将 [{}] 插入到 [{}] 之后", move_id, target_id);
            self.print_current_total_score();
        } else {
            println!("请检查简号是否正确");
        }
    }

    fn insert_before(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.len() != 2 {
            println!("用法: insert-before <要移动的简> <目标简>");
            return;
        }

        let move_id = parts[0];
        let target_id = parts[1];

        if let (Some(from_idx), Some(target_idx)) = (self.find_slip_index(move_id), self.find_slip_index(target_id)) {
            let slip = self.current_order.remove(from_idx);
            let insert_idx = if from_idx < target_idx { target_idx - 1 } else { target_idx };
            self.current_order.insert(insert_idx, slip);
            println!("已将 [{}] 插入到 [{}] 之前", move_id, target_id);
            self.print_current_total_score();
        } else {
            println!("请检查简号是否正确");
        }
    }

    fn swap_slips(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.len() != 2 {
            println!("用法: swap <简号1> <简号2>");
            return;
        }

        let id1 = parts[0];
        let id2 = parts[1];

        if let (Some(idx1), Some(idx2)) = (self.find_slip_index(id1), self.find_slip_index(id2)) {
            self.current_order.swap(idx1, idx2);
            println!("已交换 [{}] 和 [{}] 的位置", id1, id2);
            self.print_current_total_score();
        } else {
            println!("请检查简号是否正确");
        }
    }

    fn auto_order(&mut self) {
        let result = self.engine.find_optimal_order(&self.current_order);
        self.current_order = result.order.clone();
        println!("自动编连完成，总分: {:.4}", result.total_score);
    }

    fn reset_order(&mut self) {
        self.current_order = self.slips.clone();
        println!("已重置为原始顺序");
        self.print_current_total_score();
    }

    fn show_weights(&self) {
        println!("当前评分权重:");
        println!("  字形特征权重: {:.2}", self.engine.weights.glyph_weight);
        println!("  残笔描述权重: {:.2}", self.engine.weights.stroke_weight);
        println!("  语法标签权重: {:.2}", self.engine.weights.grammar_weight);
        println!("  权重总和: {:.2}",
            self.engine.weights.glyph_weight + self.engine.weights.stroke_weight + self.engine.weights.grammar_weight
        );
    }

    fn set_weights(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        if parts.len() != 3 {
            println!("用法: set-weights <字形权重> <残笔权重> <语法权重>");
            println!("示例: set-weights 0.5 0.3 0.2");
            return;
        }

        let glyph_weight: f64 = match parts[0].parse() {
            Ok(n) => n,
            Err(_) => {
                println!("无效的字形权重");
                return;
            }
        };
        let stroke_weight: f64 = match parts[1].parse() {
            Ok(n) => n,
            Err(_) => {
                println!("无效的残笔权重");
                return;
            }
        };
        let grammar_weight: f64 = match parts[2].parse() {
            Ok(n) => n,
            Err(_) => {
                println!("无效的语法权重");
                return;
            }
        };

        self.engine.weights.glyph_weight = glyph_weight;
        self.engine.weights.stroke_weight = stroke_weight;
        self.engine.weights.grammar_weight = grammar_weight;
        println!("权重已更新");
        self.print_current_total_score();
    }

    fn show_top_pair_scores(&self, arg: &str) {
        let n: usize = if arg.is_empty() {
            10
        } else {
            match arg.parse() {
                Ok(n) => n,
                Err(_) => {
                    println!("无效的数字");
                    return;
                }
            }
        };

        let mut all_scores = self.engine.compute_all_pair_scores(&self.slips);
        all_scores.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap_or(std::cmp::Ordering::Equal));

        println!("前 {} 对最高评分的相邻组合:", n.min(all_scores.len()));
        for (i, score) in all_scores.iter().take(n).enumerate() {
            println!("  {:2}. [{}] -> [{}]: 总分={:.4} (字形={:.4}, 残笔={:.4}, 语法={:.4})",
                i + 1,
                score.from_id,
                score.to_id,
                score.total_score,
                score.glyph_score,
                score.stroke_score,
                score.grammar_score
            );
        }
    }

    fn show_fragment_matches(&self, arg: &str) {
        let n: usize = if arg.is_empty() {
            10
        } else {
            match arg.parse() {
                Ok(n) => n,
                Err(_) => {
                    println!("无效的数字");
                    return;
                }
            }
        };

        let matcher = FragmentMatcher::new();
        let matches = matcher.find_top_matches(&self.slips, n);

        println!("前 {} 对最高置信度的残片匹配组合:", matches.len());
        for (i, m) in matches.iter().enumerate() {
            println!("  {:2}. [{}] -> [{}]: 置信度={:.4} (边缘={:.4}, 残笔互补={:.4}, 字形连续={:.4})",
                i + 1,
                m.left_id,
                m.right_id,
                m.confidence,
                m.edge_similarity,
                m.stroke_complement,
                m.glyph_continuity
            );
        }
    }
}
