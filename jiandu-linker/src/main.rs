use clap::{Parser, Subcommand};

mod models;
mod data_loader;
mod distance;
mod scoring;
mod cli;
mod csv_export;

use models::Weights;
use scoring::ScoringEngine;
use data_loader::load_slips_from_json;
use csv_export::export_result_to_csv;

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
