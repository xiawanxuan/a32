use std::fs;
use std::path::Path;
use serde_json;
use crate::models::BambooSlip;

pub fn load_slips_from_json<P: AsRef<Path>>(path: P) -> Result<Vec<BambooSlip>, String> {
    let file_content = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("读取文件失败: {}", e))?;

    let slips: Vec<BambooSlip> = serde_json::from_str(&file_content)
        .map_err(|e| format!("JSON 解析失败: {}", e))?;

    Ok(slips)
}

pub fn load_slips_from_str(json_str: &str) -> Result<Vec<BambooSlip>, String> {
    let slips: Vec<BambooSlip> = serde_json::from_str(json_str)
        .map_err(|e| format!("JSON 解析失败: {}", e))?;

    Ok(slips)
}
