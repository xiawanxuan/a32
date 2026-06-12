use clap::{Parser, Subcommand};

mod models;
mod data_loader;
mod distance;
mod scoring;
mod cli;
mod csv_export;
mod fragment_matching;
mod sift_matching;
mod fragment_recommender;
mod visualization;

use models::Weights;
use scoring::ScoringEngine;
use data_loader::load_slips_from_json;
use csv_export::export_result_to_csv;
use fragment_matching::FragmentMatcher;
use fragment_recommender::FragmentRecommender;
use visualization::Visualizer;

#[derive(Parser)]
#[command(name = "jiandu-linker")]
#[command(version = "0.1.0")]
#[command(about = "简牍编连工具 - 基于 Jaro-Winkler + 语法距离加权的简牍自动编连系统", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, default_value_t = 0.4)]
    #[arg(help = "字形特征权重")]
    glyph_weight: f64,

    #[arg(long, default_value_t = 0.3)]
    #[arg(help = "残笔描述权重")]
    stroke_weight: f64,

    #[arg(long, default_value_t = 0.3)]
    #[arg(help = "语法标签权重")]
    grammar_weight: f64,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "自动编连模式 - 直接计算最优编连方案并输出 CSV")]
    Auto {
        #[arg(short, long)]
        #[arg(help = "输入 JSON 文件路径")]
        input: String,

        #[arg(short, long, default_value = "output.csv")]
        #[arg(help = "输出 CSV 文件路径")]
        output: String,
    },

    #[command(about = "交互式模式 - 手动调整简牍顺序")]
    Interactive {
        #[arg(short, long)]
        #[arg(help = "输入 JSON 文件路径")]
        input: String,

        #[arg(short, long, default_value = "output.csv")]
        #[arg(help = "输出 CSV 文件路径")]
        output: String,
    },

    #[command(about = "查看所有简牍对的评分排名")]
    TopScores {
        #[arg(short, long)]
        #[arg(help = "输入 JSON 文件路径")]
        input: String,

        #[arg(short, long, default_value_t = 10)]
        #[arg(help = "显示前 N 对")]
        top: usize,
    },

    #[command(about = "基于残片特征的高级匹配分析")]
    FragmentMatch {
        #[arg(short, long)]
        #[arg(help = "输入 JSON 文件路径")]
        input: String,

        #[arg(short, long, default_value_t = 10)]
        #[arg(help = "显示前 N 对")]
        top: usize,
    },

    #[command(about = "残简拼合推荐 - 基于 SIFT 特征的断裂边缘匹配")]
    FragmentRecommend {
        #[arg(short, long)]
        #[arg(help = "输入 JSON 文件路径")]
        input: String,

        #[arg(short, long, default_value_t = 10)]
        #[arg(help = "显示前 N 对")]
        top: usize,

        #[arg(long, default_value_t = 0.6)]
        #[arg(help = "置信度阈值")]
        threshold: f64,
    },

    #[command(about = "编连方案可视化 - ASCII 树形图展示")]
    Visualize {
        #[arg(short, long)]
        #[arg(help = "输入 JSON 文件路径")]
        input: String,

        #[arg(short, long, default_value = "tree")]
        #[arg(help = "可视化模式: tree/compact/report")]
        mode: String,

        #[arg(short, long, default_value = "output.csv")]
        #[arg(help = "输出 CSV 文件路径（可选）")]
        output: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let weights = Weights {
        glyph_weight: cli.glyph_weight,
        stroke_weight: cli.stroke_weight,
        grammar_weight: cli.grammar_weight,
    };

    let engine = ScoringEngine::new(weights);

    match &cli.command {
        Commands::Auto { input, output } => {
            run_auto_mode(&engine, input, output);
        }
        Commands::Interactive { input, output } => {
            run_interactive_mode(engine, input, output);
        }
        Commands::TopScores { input, top } => {
            run_top_scores(&engine, input, *top);
        }
        Commands::FragmentMatch { input, top } => {
            run_fragment_match(input, *top);
        }
        Commands::FragmentRecommend { input, top, threshold } => {
            run_fragment_recommend(input, *top, *threshold);
        }
        Commands::Visualize { input, mode, output } => {
            run_visualize(&engine, input, mode, output);
        }
    }
}

fn run_auto_mode(engine: &ScoringEngine, input: &str, output: &str) {
    println!("=== 简牍编连 - 自动模式 ===");
    println!("输入文件: {}", input);

    let slips = match load_slips_from_json(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    println!("加载简牍数量: {}", slips.len());
    println!("正在计算最优编连方案...");

    let result = engine.find_optimal_order(&slips);

    println!();
    println!("=== 编连结果 ===");
    println!("总分: {:.4}", result.total_score);
    if !result.link_scores.is_empty() {
        let avg = result.total_score / result.link_scores.len() as f64;
        println!("平均相邻分: {:.4}", avg);
        println!();
        println!("相邻详情:");
        for (i, score) in result.link_scores.iter().enumerate() {
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
    println!();
    println!("编连顺序:");
    for (i, slip) in result.order.iter().enumerate() {
        println!("  {:3}. [{}] {} ({})", i + 1, slip.slip_id, slip.glyph_features, slip.grammar_tag);
    }

    match export_result_to_csv(&result, output) {
        Ok(_) => println!("\n结果已导出到: {}", output),
        Err(e) => eprintln!("导出 CSV 失败: {}", e),
    }
}

fn run_interactive_mode(engine: ScoringEngine, input: &str, output: &str) {
    let slips = match load_slips_from_json(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    let mut interactive_cli = cli::InteractiveCli::new(engine, slips);
    let result = interactive_cli.run();

    println!();
    println!("=== 最终编连方案 ===");
    println!("总分: {:.4}", result.total_score);
    for (i, slip) in result.order.iter().enumerate() {
        println!("  {:3}. [{}] {} ({})", i + 1, slip.slip_id, slip.glyph_features, slip.grammar_tag);
    }

    match export_result_to_csv(&result, output) {
        Ok(_) => println!("\n结果已导出到: {}", output),
        Err(e) => eprintln!("导出 CSV 失败: {}", e),
    }
}

fn run_top_scores(engine: &ScoringEngine, input: &str, top: usize) {
    println!("=== 简牍编连 - 评分排名 ===");
    println!("输入文件: {}", input);

    let slips = match load_slips_from_json(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    println!("加载简牍数量: {}", slips.len());
    println!();

    let mut all_scores = engine.compute_all_pair_scores(&slips);
    all_scores.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap_or(std::cmp::Ordering::Equal));

    println!("前 {} 对最高评分的相邻组合:", top.min(all_scores.len()));
    for (i, score) in all_scores.iter().take(top).enumerate() {
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

fn run_fragment_match(input: &str, top: usize) {
    println!("=== 简牍编连 - 残片特征高级匹配 ===");
    println!("输入文件: {}", input);

    let slips = match load_slips_from_json(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    println!("加载简牍数量: {}", slips.len());
    println!();

    let matcher = FragmentMatcher::new();
    let matches = matcher.find_top_matches(&slips, top);

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

fn run_fragment_recommend(input: &str, top: usize, threshold: f64) {
    println!("=== 简牍编连 - 残简拼合推荐 ===");
    println!("输入文件: {}", input);
    println!("置信度阈值: {:.2}", threshold);

    let slips = match load_slips_from_json(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    println!("加载简牍数量: {}", slips.len());
    println!();
    println!("正在进行 SIFT 特征离线预计算和断裂边缘匹配...");
    println!();

    let mut recommender = FragmentRecommender::new();
    recommender.confidence_threshold = threshold;
    let recommendation = recommender.recommend_matches(&slips);

    println!("SIFT 特征提取完成，共分析 {} 对组合", recommendation.matches.len());
    println!();

    if !recommendation.recommended_merges.is_empty() {
        println!("╔{}╗", "═".repeat(60));
        println!("║{:^60}║", "★ 推荐优先拼合的组合 ★");
        println!("╠{}╣", "═".repeat(60));
        for (i, (left, right, conf)) in recommendation.recommended_merges.iter().enumerate() {
            let level = FragmentRecommender::get_confidence_level(*conf);
            let bar = visualization::render_confidence_bar(*conf, 20);
            println!("║ {:2}. [{:>4}] ════► [{:>4}]  {:<28} {:>6} ║",
                i + 1, left, right, bar, level
            );
        }
        println!("╚{}╝", "═".repeat(60));
        println!();
    }

    println!("前 {} 对候选拼合组合:", top.min(recommendation.matches.len()));
    for (i, m) in recommendation.matches.iter().take(top).enumerate() {
        let level = FragmentRecommender::get_confidence_level(m.confidence);
        let bar = visualization::render_confidence_bar(m.confidence, 15);
        println!("  {:2}. [{}] -> [{}]  {:<20}  [{:>4}]",
            i + 1,
            m.left_id,
            m.right_id,
            bar,
            level
        );
        println!("      SIFT:{:.4}  几何:{:.4}  残笔:{:.4}  字形:{:.4}  匹配点:{}个",
            m.sift_similarity,
            m.edge_geometry_score,
            m.stroke_continuity,
            m.glyph_overlap_score,
            m.matched_keypoints
        );
    }
}

fn run_visualize(engine: &ScoringEngine, input: &str, mode: &str, output: &str) {
    println!("=== 简牍编连 - 方案可视化 ===");
    println!("输入文件: {}", input);
    println!("可视化模式: {}", mode);
    println!();

    let slips = match load_slips_from_json(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    println!("加载简牍数量: {}", slips.len());
    println!("正在计算最优编连方案...");
    println!();

    let result = engine.find_optimal_order(&slips);
    let visualizer = Visualizer::new();

    println!("{}", "═".repeat(72));
    match mode {
        "compact" | "c" => {
            println!("{}", visualizer.render_compact_tree(&result));
        }
        "report" | "r" => {
            println!("{}", visualizer.render_detailed_report(&result));
        }
        "tree" | "t" | _ => {
            println!("{}", visualizer.render_tree(&result));
        }
    }
    println!("{}", "═".repeat(72));

    if output != "output.csv" || mode == "report" {
        match export_result_to_csv(&result, output) {
            Ok(_) => println!("\n编连方案已导出到: {}", output),
            Err(e) => eprintln!("导出 CSV 失败: {}", e),
        }
    }
}
