use crate::models::{LinkResult, VisualizationConfig, BambooSlip};
use std::fmt::Write;

pub struct Visualizer {
    config: VisualizationConfig,
}

impl Default for Visualizer {
    fn default() -> Self {
        Visualizer {
            config: VisualizationConfig::default(),
        }
    }
}

impl Visualizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: VisualizationConfig) -> Self {
        Visualizer { config }
    }

    pub fn render_tree(&self, result: &LinkResult) -> String {
        let mut output = String::new();

        if result.order.is_empty() {
            return "编连方案为空".to_string();
        }

        writeln!(output, "{}", self.render_header(result)).unwrap();

        let indent = " ".repeat(self.config.indent_width);

        for (i, slip) in result.order.iter().enumerate() {
            let is_last = i == result.order.len() - 1;
            let connector = if is_last { "└──" } else { "├──" };
            let link_score = result.link_scores.get(i);

            writeln!(
                output,
                "{}{} {}",
                indent,
                connector,
                self.render_slip_node(slip, link_score, i)
            )
            .unwrap();

            if !is_last && self.config.show_details {
                if let Some(score) = link_score {
                    writeln!(
                        output,
                        "{}│   {} {}",
                        indent,
                        self.render_score_arrow(score),
                        self.render_score_detail(score)
                    )
                    .unwrap();
                }
            }
        }

        writeln!(output, "{}", self.render_footer(result)).unwrap();

        output
    }

    pub fn render_compact_tree(&self, result: &LinkResult) -> String {
        let mut output = String::new();

        if result.order.is_empty() {
            return "编连方案为空".to_string();
        }

        let mut compact = String::new();
        for (i, slip) in result.order.iter().enumerate() {
            if i > 0 {
                let score = result.link_scores.get(i - 1);
                let arrow = match score {
                    Some(s) if s.total_score >= 0.8 => " ═══► ",
                    Some(s) if s.total_score >= 0.6 => " ───► ",
                    Some(s) if s.total_score >= 0.4 => " ─┈─► ",
                    _ => " ┄┄┄► ",
                };
                let score_str = score
                    .map(|s| format!("({:.2})", s.total_score))
                    .unwrap_or_default();
                compact.push_str(&format!("{}{}", arrow, score_str));
            }
            compact.push_str(&format!("[{}]", slip.slip_id));
        }

        writeln!(output, "编连路径图:").unwrap();
        writeln!(output, "{}", compact).unwrap();
        writeln!(output).unwrap();
        writeln!(output, "总分: {:.4}", result.total_score).unwrap();
        if !result.link_scores.is_empty() {
            let avg = result.total_score / result.link_scores.len() as f64;
            writeln!(output, "平均分: {:.4}", avg).unwrap();
        }

        output
    }

    pub fn render_detailed_report(&self, result: &LinkResult) -> String {
        let mut output = String::new();

        writeln!(output, "{}", "═".repeat(self.config.max_width)).unwrap();
        writeln!(output, "{}", self.render_header(result)).unwrap();
        writeln!(output, "{}", "═".repeat(self.config.max_width)).unwrap();
        writeln!(output).unwrap();

        writeln!(output, "{}", self.render_tree(result)).unwrap();

        writeln!(output, "{}", "─".repeat(self.config.max_width)).unwrap();
        writeln!(output, "评分详情:").unwrap();
        writeln!(output).unwrap();

        for (i, score) in result.link_scores.iter().enumerate() {
            writeln!(
                output,
                "  {:2}. [{}] → [{}]",
                i + 1,
                score.from_id,
                score.to_id
            )
            .unwrap();
            writeln!(
                output,
                "     总分: {:.4}  |  字形: {:.4}  |  残笔: {:.4}  |  语法: {:.4}",
                score.total_score,
                score.glyph_score,
                score.stroke_score,
                score.grammar_score
            )
            .unwrap();

            let bar_len = (score.total_score * 20.0) as usize;
            let bar: String = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
            writeln!(output, "     置信度: [{}] {:.1}%", bar, score.total_score * 100.0).unwrap();
            writeln!(output).unwrap();
        }

        writeln!(output, "{}", "═".repeat(self.config.max_width)).unwrap();

        output
    }

    fn render_header(&self, result: &LinkResult) -> String {
        format!(
            "简牍编连方案 (共 {} 枚简, 总分: {:.4})",
            result.order.len(),
            result.total_score
        )
    }

    fn render_footer(&self, result: &LinkResult) -> String {
        if !result.link_scores.is_empty() {
            let avg = result.total_score / result.link_scores.len() as f64;
            let quality = if avg >= 0.8 {
                "优秀"
            } else if avg >= 0.6 {
                "良好"
            } else if avg >= 0.4 {
                "一般"
            } else {
                "较差"
            };
            format!(
                "平均连接分: {:.4}  |  编连质量: {}",
                avg, quality
            )
        } else {
            String::new()
        }
    }

    fn render_slip_node(
        &self,
        slip: &BambooSlip,
        link_score: Option<&crate::models::LinkScore>,
        index: usize,
    ) -> String {
        let glyph_preview = if slip.glyph_features.chars().count() > 10 {
            let truncated: String = slip.glyph_features.chars().take(10).collect();
            format!("{}…", truncated)
        } else {
            slip.glyph_features.clone()
        };

        let score_tag = if self.config.show_scores {
            link_score
                .map(|s| format!("  <{:.2}>", s.total_score))
                .unwrap_or_default()
        } else {
            String::new()
        };

        format!(
            "{:3}. [{}]  {:<12}  ({}){}",
            index + 1,
            slip.slip_id,
            glyph_preview,
            slip.grammar_tag,
            score_tag
        )
    }

    fn render_score_arrow(&self, score: &crate::models::LinkScore) -> String {
        if score.total_score >= 0.85 {
            "═══════════════►"
        } else if score.total_score >= 0.7 {
            "───────────────►"
        } else if score.total_score >= 0.55 {
            "─┄─┄─┄─┄─┄─┄─►"
        } else if score.total_score >= 0.4 {
            "┄┄┄┄┄┄┄┄┄┄┄┄┄┄►"
        } else {
            "· · · · · · · ·►"
        }
    }

    fn render_score_detail(&self, score: &crate::models::LinkScore) -> String {
        format!(
            "字:{:.2} 残:{:.2} 语:{:.2}",
            score.glyph_score, score.stroke_score, score.grammar_score
        )
    }
}

pub fn render_confidence_bar(score: f64, width: usize) -> String {
    let filled = (score * width as f64).round() as usize;
    let filled = filled.min(width);
    let (fill_char, empty_char) = if score >= 0.8 {
        ('█', '░')
    } else if score >= 0.6 {
        ('▓', '░')
    } else if score >= 0.4 {
        ('▒', '░')
    } else {
        ('░', '░')
    };
    format!(
        "[{}{}] {:.1}%",
        fill_char.to_string().repeat(filled),
        empty_char.to_string().repeat(width - filled),
        score * 100.0
    )
}
