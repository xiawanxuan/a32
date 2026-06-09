use std::fs::File;
use std::path::Path;
use csv::WriterBuilder;
use crate::models::{LinkResult, LinkScore};

pub fn export_result_to_csv<P: AsRef<Path>>(result: &LinkResult, path: P) -> Result<(), String> {
    let file = File::create(path.as_ref())
        .map_err(|e| format!("创建 CSV 文件失败: {}", e))?;

    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_writer(file);

    wtr.write_record(["序号", "简号", "字形特征", "残笔描述", "语法标签", "相邻总分", "字形分", "残笔分", "语法分"])
        .map_err(|e| format!("写入 CSV 表头失败: {}", e))?;

    for (i, slip) in result.order.iter().enumerate() {
        let link_score = result.link_scores.get(i);

        let total_score = link_score.map(|s| format!("{:.4}", s.total_score)).unwrap_or_default();
        let glyph_score = link_score.map(|s| format!("{:.4}", s.glyph_score)).unwrap_or_default();
        let stroke_score = link_score.map(|s| format!("{:.4}", s.stroke_score)).unwrap_or_default();
        let grammar_score = link_score.map(|s| format!("{:.4}", s.grammar_score)).unwrap_or_default();

        wtr.write_record([
            (i + 1).to_string(),
            slip.slip_id.clone(),
            slip.glyph_features.clone(),
            slip.remnant_strokes.clone(),
            slip.grammar_tag.clone(),
            total_score,
            glyph_score,
            stroke_score,
            grammar_score,
        ]).map_err(|e| format!("写入 CSV 行失败: {}", e))?;
    }

    wtr.flush().map_err(|e| format!("刷新 CSV 缓冲区失败: {}", e))?;

    Ok(())
}

pub fn export_link_scores_to_csv<P: AsRef<Path>>(scores: &[LinkScore], path: P) -> Result<(), String> {
    let file = File::create(path.as_ref())
        .map_err(|e| format!("创建 CSV 文件失败: {}", e))?;

    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_writer(file);

    wtr.write_record(["起始简号", "目标简号", "总分", "字形分", "残笔分", "语法分"])
        .map_err(|e| format!("写入 CSV 表头失败: {}", e))?;

    for score in scores {
        wtr.write_record([
            score.from_id.clone(),
            score.to_id.clone(),
            format!("{:.4}", score.total_score),
            format!("{:.4}", score.glyph_score),
            format!("{:.4}", score.stroke_score),
            format!("{:.4}", score.grammar_score),
        ]).map_err(|e| format!("写入 CSV 行失败: {}", e))?;
    }

    wtr.flush().map_err(|e| format!("刷新 CSV 缓冲区失败: {}", e))?;

    Ok(())
}
