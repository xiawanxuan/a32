use std::fs;
use std::path::Path;
use std::collections::HashSet;
use serde_json;
use crate::models::BambooSlip;

pub fn load_slips_from_json<P: AsRef<Path>>(path: P) -> Result<Vec<BambooSlip>, String> {
    let file_content = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("读取文件失败: {}", e))?;

    load_slips_from_str(&file_content)
}

pub fn load_slips_from_str(json_str: &str) -> Result<Vec<BambooSlip>, String> {
    let slips: Vec<BambooSlip> = serde_json::from_str(json_str)
        .map_err(|e| format!("JSON 解析失败: {}", e))?;

    validate_slips(&slips)?;

    Ok(slips)
}

fn validate_slips(slips: &[BambooSlip]) -> Result<(), String> {
    if slips.is_empty() {
        return Err("简牍数据为空".to_string());
    }

    let mut seen_ids = HashSet::new();
    let mut warnings = Vec::new();

    for (i, slip) in slips.iter().enumerate() {
        if slip.slip_id.is_empty() {
            warnings.push(format!("第 {} 枚简的简号为空", i + 1));
        } else if !seen_ids.insert(slip.slip_id.clone()) {
            return Err(format!("简号重复: {}", slip.slip_id));
        }

        if slip.glyph_features.is_empty() {
            warnings.push(format!("简 [{}] 的字形特征为空", slip.slip_id));
        }

        if slip.grammar_tag.is_empty() {
            warnings.push(format!("简 [{}] 的语法标签为空", slip.slip_id));
        }
    }

    for w in &warnings {
        eprintln!("警告: {}", w);
    }

    Ok(())
}
